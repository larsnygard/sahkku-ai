"""Self-play data generation (AlphaZero-lite on GPU).

For every decision the agent takes, the ROOT position is evaluated by
`rollouts` parallel policy-sampled playouts (temperature `temp`):
  * z   = mean leaf outcome (root-to-move POV)   -> value target
  * pi  = visit distribution over root actions   -> policy target
          + Dirichlet root noise (exploration)
The move played is the most-visited root action.

The `rollouts` playouts run SIMULTANEOUSLY: one batched GPU forward per ply
over all active playouts (the pure-Python engine steps each state). Root
policy is sampled once per rollout from the shared root policy vector.
Output: runs/<name>/shard_<worker>.npz (data.py shard format).
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.multiprocessing as mp

from . import engine as E
from . import features as FT
from .features import _square_attackable
from .model import Net
from .data import save_shard

MAX_PLIES = 300


@torch.no_grad()
def net_pol(net, device, states):
    """Batched policy for a list of states."""
    x = torch.from_numpy(np.stack([FT.encode_pair(s) for s in states]))
    pl = x[:, :FT.N_PLANES * 45].to(device)
    g = x[:, FT.N_PLANES * 45:].to(device)
    pol, _ = net(pl, g)
    return pol.cpu().numpy()


@torch.no_grad()
def net_pv(net, device, states):
    x = torch.from_numpy(np.stack([FT.encode_pair(s) for s in states]))
    pl = x[:, :FT.N_PLANES * 45].to(device)
    g = x[:, FT.N_PLANES * 45:].to(device)
    pol, val = net(pl, g)
    return pol.cpu().numpy(), val.cpu().numpy()


def sample_policy(pol_vec, legal, temp, rng):
    lg = np.asarray(legal, dtype=np.int64)
    z = pol_vec[lg] / max(temp, 1e-3)
    z = z - z.max()
    w = np.exp(z)
    w /= w.sum()
    return int(lg[rng.choices(range(len(lg)), weights=w.tolist(), k=1)[0]])


def apply_choice(st, ids, aid, rng):
    """Apply action id `aid` (from ids list) to state st."""
    for a, mv in ids:
        if a == aid:
            if mv.get("kind") == "pass":
                E.pass_remaining(st)
            elif mv.get("kind") == "reroll":
                E.reroll_x(st, rng.random)
                st.phase = "move"
            else:
                E.apply_move(st, mv)
                E.settle(st)
            return True
    return False


def rollout_batch(net, device, children, temp, rng):
    """Policy-sampled playouts for a list of child states, batched per ply.
    Returns list of z (outcome from EACH CHILD-to-move POV; draw/unfinished 0)."""
    side0 = [c.turn for c in children]
    z = [0.0] * len(children)
    # Children already decided by the ROOT move (queen taken, annihilation):
    # their outcome is known before any playout and must not be scored as a
    # draw — these are the sharpest value targets in the whole dataset.
    for i, ch in enumerate(children):
        if ch.winner is not None:
            z[i] = 1.0 if ch.winner == side0[i] else -1.0
    alive = [i for i in range(len(children)) if children[i].winner is None]
    for _ in range(MAX_PLIES):
        if not alive:
            break
        states = [children[i] for i in alive]
        # states needing dice get a roll (pure python, no net cost)
        need_net = []
        for k, i in enumerate(alive):
            st = children[i]
            if st.winner is not None:
                continue
            if st.phase == "roll":
                E.roll(st, rng.random)
                st.phase = "move"
        # legal pools
        pools = {i: FT.legal_ids(children[i]) for i in alive}
        to_step = [i for i in alive if children[i].winner is None and pools[i]]
        # positions with no legal action: pass and continue next ply
        for i in alive:
            if children[i].winner is None and not pools[i]:
                E.pass_remaining(children[i])
        if not to_step:
            continue
        pols = net_pol(net, device, [children[i] for i in to_step])
        for k, i in enumerate(to_step):
            legal = [a for a, _ in pools[i]]
            aid = sample_policy(pols[k], legal, temp, rng)
            apply_choice(children[i], pools[i], aid, rng)
        # collect finished
        for i in alive:
            st = children[i]
            if st.winner is not None:
                z[i] = 1.0 if st.winner == side0[i] else -1.0
        alive = [i for i in alive if children[i].winner is None]
    return z


# ---------- «sein dronning»: handikap for éi side i sjølvspel ----------
# Lars 2026-10-27: AIL0.6 skal trenast AIL0.5 mot AIL0.5, men éi side får
# ALDRI lov til å vekkje dronninga før ho har berre N soldatar att. Då ser
# nettet kva som faktisk skjer når dronninga kjem seint — noko sjølvspel
# elles aldri utforskar, fordi begge sider vekkjer henne tidleg.
#
# Handikapet er eit TREKKVAL, ikkje ei endra regel: den handikappa sida
# spelar berre andre lovlege trekk. Er ho blokkert utan dronningvekking
# (kan skje når ho berre har dronninga att), får ho vekkje likevel — elles
# ville partiet stått fast og datagenereringa gjeve søppel.

def qd_side_for_game(cfg, rng):
    """Kva side er handikappa i dette partiet (-1 = ingen)."""
    qd = cfg.get("queen_delay", -1)
    if qd is None or qd < 0:
        return -1
    if qd == 2:                      # «alt»: byt side frå parti til parti
        return rng.randint(0, 1)
    return 1 if qd == 1 else 0


def queen_activation_blocked(st, mv, side, min_soldiers):
    """Er dette ei dronning-vekking som handikapet forbyr nett no?"""
    if side is None or side < 0 or st.turn != side:
        return False
    if mv.get("kind") != "rel":          # berre vekking (X), ikkje vanlege trekk
        return False
    p = E.piece(st, mv["pid"])
    if p is None or p.t != "q":
        return False
    return len(E.alive_soldiers(st, side)) > min_soldiers


def filter_qd(st, ids, side, min_soldiers):
    """Fjern dronningvekking frå trekkmengda; aldri tøm heile mengda."""
    if side is None or side < 0:
        return ids
    keep = [(a, mv) for a, mv in ids
            if not queen_activation_blocked(st, mv, side, min_soldiers)]
    return keep or ids                   # fallback: heller vekkje enn å stå fast


# ---------- materialklausa i sjølvspel ---------------------------------
# Verdhovudet undervurderer gratisslag (målt: eit gratis soldatkast vart
# vurdert 0,08 dårlegare enn eit roleg dronningtrekk, på alle søkedjupner).
# Policy-måla kjem frå søket, so når søket aldri tek slaget, lærer nettet
# aldri at slaget er bra. Difor let vi sjølvspelet spele trygge slag når dei
# er i nærleiken av det beste trekket — då inneheld måla slag, og verdihovudet
# får sjå at slike game faktisk vinn.
CAP_RATIO = 0.5          # trygt slag med minst denne delen av beste π vert spelt


def _victims(st, mv):
    land = E.move_dest(st, mv)
    me = st.turn
    return [q for q in E.occ(st, land)
            if q.o is not None and q.o != me and q.id != mv["pid"]]


def _square_attackable_unused(st, sq, side):
    """Kan `side` slå ein brikke som står på `sq`, med terning 1-3?

    Billig og eksakt: vi går gjennom brikkene og sjekkar kvar dei kan lande
    (path_for for konge/dronning, soldatane berre framover i syklusen), i
    staden for å byggje tre fulle trekk-mengder per kandidat. Det var denne
    kostnaden som gjorde AIL0.8-sjølvspelet 2,4x tregare enn AIL0.7.
    """
    for p in st.pieces:
        if p.captured or p.o != side:
            continue
        if p.t == "s":
            if not p.released:
                continue
            for v in (1, 2, 3):
                if E.soldier_at(side, p.at + v) == sq and E.can_land(st, p, sq):
                    return True
            continue
        for v in (1, 2, 3):
            for d in ("R", "L", "F", "B"):
                ok, land, _ = E.path_for(p.at, d, v)
                if ok and land == sq and E.can_land(st, p, sq):
                    return True
    return False


def _can_take_back(st, mv):
    """Kan motstandaren slå den brikka vi netto flytta, med terning 1-3?"""
    p = E.piece(st, mv["pid"])
    if p is None or p.captured:
        return True
    sq = E.soldier_at(p.o, p.at) if p.t == "s" else p.at
    return _square_attackable(st, sq, 1 - p.o)


# ---------- «aggressiv side» i sjølvspel --------------------------------
# Lars 2026-10-29: ny treningsøkt der ÉI side spelar meir aggressivt. Vi gjer
# det som ei vekting av policyen for den sida: slag får meir vekt, og trekk
# som flyttar ein soldat inn på motstandarens halvdel får litt meir. Sida
# vekslar frå parti til parti, so nettet lærer både å angripe og å straffe
# angrep — elles vart det berre éin spelestil i data.
AGR_CAP = 2.5        # vektfaktor på slag
AGR_ADV = 1.4        # vektfaktor på framrykk mot fienden


def aggressive_side_for_game(cfg, rng):
    if not cfg.get("aggressive"):
        return -1
    if rng.random() >= float(cfg.get("aggressive_fraction", 1.0)):
        return -1
    return rng.randint(0, 1)


def reweight_aggressive(st, ids, pi, cap=AGR_CAP, adv=AGR_ADV):
    """Vekt π for den aggressive sida: slag og framrykk tel meir."""
    out = dict(pi)
    me = st.turn
    for a, mv in ids:
        if mv.get("kind") in ("pass", "reroll") or a not in out:
            continue
        land = E.move_dest(st, mv)
        w = 1.0
        if _victims(st, mv):
            w *= cap
        p = E.piece(st, mv["pid"])
        if p.t == "s":
            # soldat framover: kjem han nærare/inn på fienden si side?
            row = E.pt(land)[1]
            home_row = 0 if me == 0 else 2
            if row != home_row:
                w *= adv
        out[a] = out[a] * w
    tot = sum(out.values())
    if tot > 0:
        out = {a: v / tot for a, v in out.items()}
    return out


# ------------------------------------------------------------------ stil-eksperiment
# NAIS.6: fire taktiske stilar, køyrde som DUGELEGAR (kvar kamp: éi side spelar
# stil A, den andre stil B, og sidene byter kvar kamp). Då svarar køyringa på
# to spørsmål samstundes: kva taktikk vinn (duellane er eit kontrollert
# forsøk), og nettet lærer båe sidene av kvar taktikk.
#
#   few        aktiver FÅ soldatar, og dytt dei som alt er ute hardt fram
#   many       aktiver SÅ MANGE som mogleg, spreidd over breidda
#   kingfwd    etter kongekapring: bruk kongen offensivt, inn mot fienden
#   kingguard  etter kongekapring: hald kongen attende, send soldatane fremst
PAIRS = [("few", "many"), ("kingfwd", "kingguard")]
REL_FEW = 0.30        # løysing vert lite attraktiv når ≥2 alt er ute
REL_MANY = 2.5        # løysing vert sterkt attraktiv
PUSH = 1.6            # soldat som alt er ute, framover
KING_FWD = 2.5        # kongetrekk som kjem nærare fienden
KING_BACK = 2.5       # kongetrekk som går bort frå fienden
SOLDIER_SEND = 1.6    # soldatframrykk når kongen skal vernast


def style_pair_for_game(cfg, rng):
    """Kva stil kvar side spelar i dette partiet (None = vanleg)."""
    if not cfg.get("styles"):
        return (None, None)
    if rng.random() >= float(cfg.get("style_fraction", 1.0)):
        return (None, None)
    pair = PAIRS[rng.randint(0, len(PAIRS) - 1)]
    return pair if rng.randint(0, 1) == 0 else (pair[1], pair[0])


def _enemy_home_row(side):
    return 2 if side == 0 else 0


def _king_rank(st, side):
    """Kor nær fienden sin heimrad kongen står (lågare = lenger fram)."""
    k = E.king(st)
    if k is None or k.captured or k.o != side:
        return None
    return abs(E.pt(E.at_idx(st, k))[1] - _enemy_home_row(side))


def reweight_style(st, ids, pi, style):
    """Vekt π for éin stil. Vert køyrd FØR policy-målet vert skrive, so både
    trekket som vert spelt og målet nettet lærer av ber stilen."""
    out = dict(pi)
    me = st.turn
    released = sum(1 for q in E.soldiers(st, me) if q.released and not q.captured)
    krank = _king_rank(st, me)
    eh = _enemy_home_row(me)
    for a, mv in ids:
        if mv.get("kind") in ("pass", "reroll") or a not in out:
            continue
        p = E.piece(st, mv["pid"])
        w = 1.0
        if style == "few":
            if mv.get("kind") == "rel":
                if released >= 2:
                    w *= REL_FEW
            elif p.t == "s":
                w *= PUSH
        elif style == "many":
            if mv.get("kind") == "rel":
                w *= REL_MANY
        elif style in ("kingfwd", "kingguard") and krank is not None:
            if p.t == "k":
                adv = abs(E.pt(E.move_dest(st, mv))[1] - eh) - krank
                if style == "kingfwd":
                    w *= KING_FWD if adv < 0 else (0.4 if adv > 0 else 1.0)
                else:
                    w *= KING_BACK if adv > 0 else (0.4 if adv < 0 else 1.0)
            elif p.t == "s" and style == "kingguard":
                w *= SOLDIER_SEND
        out[a] = out[a] * w
    tot = sum(out.values())
    return {a: v / tot for a, v in out.items()} if tot > 0 else dict(pi)


def style_summary(cfg):
    """Kort oppsummering av stil-duellane for éin arbeidar (til loggen)."""
    games = cfg.get("_style_games") or []
    if not games:
        return ""
    out = []
    for a, b in PAIRS:
        wa = wb = dr = 0
        for g in games:
            s0, s1, w = g["s0"], g["s1"], g["winner"]
            if {s0, s1} != {a, b}:
                continue
            if w is None:
                dr += 1
                continue
            who = s0 if w == 0 else s1
            if who == a:
                wa += 1
            else:
                wb += 1
        n = wa + wb
        if n:
            out.append(f"{a} {wa}-{wb} {b}" + (f" ({dr} draw)" if dr else ""))
    # åtferd: snitt løyste soldatar ved trekk 40, og kor nær fienden kongen enda
    def style_mean(style, key):
        vals = []
        for g in games:
            for s in (0, 1):
                if (g["s0"] if s == 0 else g["s1"]) != style:
                    continue
                v = (g.get(key) or {}).get(s)
                if v is not None:
                    vals.append(v)
        return sum(vals) / len(vals) if vals else None

    bits = []
    few, many = style_mean("few", "rel40"), style_mean("many", "rel40")
    if few is not None and many is not None:
        bits.append(f"løyste@40: few {few:.1f} vs many {many:.1f}")
    kf, kg = style_mean("kingfwd", "krank"), style_mean("kingguard", "krank")
    if kf is not None and kg is not None:
        bits.append(f"kongeavstand: fwd {kf:.1f} vs guard {kg:.1f}")
    txt = ", ".join(out + bits)
    return (", stil: " + txt) if txt else ""


def safe_capture(st, ids, pi):
    """Trygt slag med høgast π, om det er minst CAP_RATIO av beste π."""
    if not ids:
        return None
    best_a = max(pi, key=pi.get)
    best_p = pi.get(best_a, 0.0)
    cand = None
    for a, mv in ids:
        if mv.get("kind") in ("pass", "reroll"):
            continue
        if not _victims(st, mv):
            continue
        probe = st.clone()
        E.apply_move(probe, mv)
        if _can_take_back(probe, mv):
            continue
        if a not in pi:
            continue
        if cand is None or pi[a] > pi[cand]:
            cand = a
    if cand is not None and pi.get(cand, 0.0) >= CAP_RATIO * best_p:
        return cand
    return None


def dirichlet(rng, alpha, k):
    gs = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    s = sum(gs) + 1e-12
    return [g / s for g in gs]


def softmax_prior(pol_vec, legal):
    """Policy prior over the legal action ids (float list, sums to 1)."""
    z = np.asarray([pol_vec[a] for a in legal], dtype=np.float64)
    z -= z.max()
    w = np.exp(z)
    return (w / w.sum()).tolist()


def evaluate_root(net, device, st, cfg, rng):
    """One-ply policy improvement + value target for the root position.

    `rollouts` policy-sampled first moves are each played out to the end
    (batched). Per action a we get an empirical value  q[a]  = mean outcome
    (root-to-move POV, +1 / -1 / 0) and a visit count  n[a].

    Policy target  π[a] ∝ (n[a] + prior_eps·N·prior[a]) · exp(q[a]/tau_q):
    the actions that actually WON in the playouts get more weight than the
    ones that lost. Without this term the target is just the net's own prior
    re-normalised — self-distillation that can never get stronger; with it
    the policy head absorbs what the playouts discovered.

    Returns (π: dict aid->prob, z_root: float, legal: [aid]).
    """
    ids = FT.legal_ids(st)
    legal = [a for a, _ in ids]
    pol = net_pol(net, device, [st])[0]
    prior = softmax_prior(pol, legal)
    picks = [sample_policy(pol, legal, cfg["temp"], rng)
             for _ in range(cfg["rollouts"])]
    children = []
    for a in picks:
        ch = st.clone()
        apply_choice(ch, ids, a, rng)
        children.append(ch)
    zc = rollout_batch(net, device, children, cfg["temp"], rng)
    # zc is CHILD-to-move POV; convert to ROOT-to-move POV. A child that the
    # root move already finished keeps turn == acting side, so the same
    # comparison still holds.
    zr = []
    for ch, z in zip(children, zc):
        zr.append(z if ch.turn == st.turn else -z)

    n = {a: 0 for a in legal}
    sz = {a: 0.0 for a in legal}
    for a, z in zip(picks, zr):
        n[a] += 1
        sz[a] += z
    q = {a: (sz[a] / n[a] if n[a] else 0.0) for a in legal}

    N = max(1, len(picks))
    eps = cfg.get("prior_eps", 0.2)
    tau_q = max(1e-3, cfg.get("tau_q", 1.0))
    pi, tot = {}, 0.0
    for a, pr in zip(legal, prior):
        w = (n[a] + eps * N * pr) * math.exp(q[a] / tau_q)
        pi[a] = w
        tot += w
    for a in legal:
        pi[a] /= max(tot, 1e-12)

    z_root = float(sum(zr) / max(1, len(zr)))
    if cfg["noise_alpha"] > 0:
        d = dirichlet(rng, cfg["noise_alpha"], len(legal))
        for a, dv in zip(legal, d):
            pi[a] = (1 - cfg["noise_eps"]) * pi[a] + cfg["noise_eps"] * dv
    return pi, z_root, legal


def play_selfplay_game(net, device, cfg, rng, samples_out, pred=None):
    # Regelmodus per parti: med --nesseby-fraction 0.5 vert halvparten av
    # partia spela med Nesseby og halvparten utan, so EITT nett lærer båe
    # variantane og flaget g[15] faktisk blir meiningsfullt. (Berre-Nesseby
    # data gjer nettet dårleg i vanleg sáhkku og omvendt.)
    _nb = bool(cfg.get("nesseby", False))
    if _nb:
        _frac = float(cfg.get("nesseby_fraction", 1.0))
        if _frac < 1.0:
            _nb = rng.random() < _frac
    agg_side = aggressive_side_for_game(cfg, rng)
    cfg["_agg_side"] = agg_side          # so køyringa kan dokumentere effekten
    style_p0, style_p1 = style_pair_for_game(cfg, rng)
    cfg["_style_pair"] = (style_p0, style_p1)
    # Diagnostikk for Nesseby: kven kaprar kongen, og når? (Asymmetrien vi
    # fann: Giera-setet vann 22 av 30 i benken medan tilfeldig spel gjev Menn
    # 57 % — då må vi måle kven som faktisk tek kongen, ikkje gjette.)
    _nb_caps = cfg.get("_nb_caps") if _nb else None
    _ply_no = 0
    _snap40 = {}          # side -> tal løyste soldatar ved trekk 40
    st = E.State(first=rng.randint(0, 1), nesseby=_nb)
    _op = E.opening_roll(st, rng.random)
    qd_side = qd_side_for_game(cfg, rng)
    qd_min = int(cfg.get("queen_delay_soldiers", 2))
    rec = cfg.get("_record")        # valfritt opptak -> SPG-eksport
    if isinstance(rec, dict):
        rec.clear()
        rec.update({"first": _op["first"],
                    "opening": [{"p0": _op["p0"], "p1": _op["p1"]}],
                    "plies": [], "qd_side": qd_side})
    rec_ply = None
    outcome_idx = []      # (indeks i samples_out, sida-som-trekte) for MCTS
    for _ in range(int(cfg.get("max_plies", MAX_PLIES))):
        if st.winner is not None:
            break
        if st.phase == "roll":
            E.roll(st, rng.random)
            st.phase = "move"
            if isinstance(rec, dict):
                rec_ply = {"dice": [list(st.dice)], "moves": []}
                rec["plies"].append(rec_ply)
            continue
        ids = filter_qd(st, FT.legal_ids(st), qd_side, qd_min)
        if not ids:
            E.pass_remaining(st)
            continue
        planes, g = FT.encode_state(st)
        if pred is not None and cfg.get("sims", 0) > 0:
            # MCTS: π frå vitjingstal, z frå søket (verdien vert likevel
            # overstyrt av partresultatet nedanfor, som før)
            from . import mcts as MCTS
            pi, z_search = MCTS.search(pred, st, cfg, rng, root_noise=True)
            legal = list(pi.keys())
            if not legal:
                E.pass_remaining(st)
                continue
            z = z_search
        else:
            pi, z, legal = evaluate_root(net, device, st, cfg, rng)
        # Handikapet må gjelde SØKET òg: søket kan ha vitja eit forbode
        # dronningtrekk (det kjenner ikkje filteret), so π vert maskert her.
        # Står alt att som forbode (skal ikkje skje — filter_qd let alltid
        # noko stå), held vi på søket sitt svar.
        if qd_side >= 0:
            ok = {a for a, _ in ids}
            pi2 = {a: p for a, p in pi.items() if a in ok}
            tot2 = sum(pi2.values())
            if pi2 and tot2 > 0:
                pi = {a: p / tot2 for a, p in pi2.items()}
                legal = list(pi.keys())
        # Stilen må inn FØR målet vert skrive. (Den gamle aggressive-køyringa
        # vekta etterpå, so berre trekket — ikkje policy-målet — bar stilen;
        # nettet lærte då berre av stillingane som oppstod.)
        my_style = style_p0 if st.turn == 0 else style_p1
        if my_style:
            pi = reweight_style(st, ids, pi, my_style)
            legal = list(pi.keys())
        pol = np.zeros(FT.N_ACTIONS, np.float32)
        for a in legal:
            pol[a] = pi[a]
        samples_out.append((planes.reshape(-1), g, pol, np.float32(z)))
        if pred is not None and cfg.get("sims", 0) > 0:
            outcome_idx.append((len(samples_out) - 1, st.turn))
        # Play the outcome-weighted best action (π), not the raw prior.
        best = max(legal, key=lambda a: pi[a])
        # Aggressiv side: vekt policyen hennar mot slag og framrykk, og vel
        # det trekket som då veg tyngst. (Policy-målet under er det SAME
        # vekta, so nettet lærer stilen — ikkje berre spelet.)
        if agg_side >= 0 and st.turn == agg_side:
            pi = reweight_aggressive(st, ids, pi)
            legal = list(pi.keys())
            best = max(legal, key=lambda a: pi[a])
        else:
            # Materialklausa: eit trygt slag som er nær nok beste trekk vert
            # spelt, so policy-måla inneheld slag (sjå safe_capture over).
            _sc = safe_capture(st, ids, pi)
            if _sc is not None:
                best = _sc
        # Tel brot på handikapet: skal vere 0 utanom fallbacken (der ho må
        # vekkje for ikkje å stå fast). Eit tal i treningsloggen er betre
        # prov enn at filteret «ser rett ut».
        if qd_side >= 0:
            mv_best = next((m for a, m in ids if a == best), None)
            if mv_best is not None and queen_activation_blocked(st, mv_best, qd_side, qd_min):
                stats = cfg.get("_qd_stats")
                if isinstance(stats, dict):
                    stats["viol"] = stats.get("viol", 0) + 1
                stats_soldiers = len(E.alive_soldiers(st, qd_side))
                if isinstance(stats, dict):
                    stats.setdefault("fallback_soldiers", []).append(stats_soldiers)
        _before = list(st.dice)
        if _ply_no == 40 and not _snap40:
            for _s in (0, 1):
                _snap40[_s] = sum(1 for q in E.soldiers(st, _s) if q.released and not q.captured)
        _k_before = E.king(st).o
        apply_choice(st, ids, best, rng)
        _ply_no += 1
        if _nb_caps is not None:
            _k_now = E.king(st).o
            if _k_before is None and _k_now is not None:
                _nb_caps.append((_k_now, _ply_no))
        if isinstance(rec, dict) and rec_ply is not None:
            _mv = dict(next((m for a, m in ids if a == best), {}))
            _kind = _mv.get("kind")
            if _kind in ("pass", "reroll"):
                # spesialhandling: ikkje eit trekk med pid — gøym henne for seg
                rec_ply["special"] = _kind
                if len(st.dice) == 3 and list(st.dice) != _before:
                    rec_ply["dice"].append(list(st.dice))   # nytt kast etter omkast
            else:
                rec_ply["moves"].append(_mv)
    cfg["_last_state"] = st          # feilsøking: kva enda partiet i?
    # Stil-duell: logg kven som spela kva, kven som vann og korleis åtferda
    # vart — utan dette kan vi ikkje seie om stilen batt eller berre var støy.
    if style_p0 or style_p1:
        g_ = cfg.setdefault("_style_games", [])
        g_.append({
            "s0": style_p0, "s1": style_p1, "winner": st.winner,
            "plies": _ply_no, "reason": st.win_reason or "",
            "rel40": _snap40,            # {0: n, 1: n} løyste soldatar ved trekk 40
            "krank": {0: _king_rank(st, 0), 1: _king_rank(st, 1)},
            "kowner": E.king(st).o,
        })
    # MCTS: verdimålet er PARTIRESULTATET (AlphaZero-konvensjonen), ikkje
    # søkets eige verdianslag — elles vert måla mjuke tal som ikkje lærer
    # nettet kven som faktisk vann.
    if outcome_idx and st.winner is not None:
        for i, side in outcome_idx:
            pl_, g_, pol_, _ = samples_out[i]
            samples_out[i] = (pl_, g_, pol_,
                              np.float32(1.0 if st.winner == side else -1.0))
    return st.winner



def play_vs_opponent(net, device, cfg, rng, samples_out, pred=None):
    """Agenten (nettet) spelar mot klassisk CPU; sida vert bytt kvar kamp.

    Treningssignalet er partiresultatet mot ein STERK, deterministisk
    motstandar — ikkje agenten sine eigne utspelingar. Det er det som gjer at
    ho kan lære noko ho ikkje alt kan (t.d. ikkje å henge dronninga).
    """
    from . import classic as CL
    level = cfg.get("opponent_level", 3)
    net_side = rng.randint(0, 1)
    st = E.State(first=rng.randint(0, 1))
    E.opening_roll(st, rng.random)
    outcome_idx = []
    for _ in range(MAX_PLIES):
        if st.winner is not None:
            break
        if st.phase == "roll":
            E.roll(st, rng.random)
            st.phase = "move"
        ids = FT.legal_ids(st)
        if not ids:
            if E.can_reroll_x(st):
                E.reroll_x(st, rng.random)
                st.phase = "move"
                continue
            E.pass_remaining(st)
            continue
        if st.turn != net_side:
            mv = CL.pick(st, level, rng.random)
            if mv is None:
                E.pass_remaining(st)
                continue
            E.apply_move(st, mv)
            E.settle(st)
            continue
        planes, g = FT.encode_state(st)
        if pred is not None and cfg.get("sims", 0) > 0:
            from . import mcts as MCTS
            pi, z = MCTS.search(pred, st, cfg, rng, root_noise=True)
            legal = list(pi.keys())
            if not legal:
                E.pass_remaining(st)
                continue
        else:
            pi, z, legal = evaluate_root(net, device, st, cfg, rng)
        pol = np.zeros(FT.N_ACTIONS, np.float32)
        for a in legal:
            pol[a] = pi[a]
        samples_out.append((planes.reshape(-1), g, pol, np.float32(z)))
        outcome_idx.append((len(samples_out) - 1, st.turn))
        best = max(legal, key=lambda a: pi[a])
        apply_choice(st, ids, best, rng)
    if outcome_idx and st.winner is not None:
        for i, side in outcome_idx:
            pl_, g_, pol_, _ = samples_out[i]
            samples_out[i] = (pl_, g_, pol_,
                              np.float32(1.0 if st.winner == side else -1.0))
    return st.winner


def worker(args):
    wid, args_json = args
    cfg = json.loads(args_json)
    # MERK: ikkje rør torch.set_grad_enabled() globalt her. Med --workers 1
    # køyrer worker() i same prosess som treninga (pipeline), og ein global
    # «grad av» gjer at neste loss.backward() feilar med «does not require
    # grad». Alle nettkalla under er alt @torch.no_grad-dekorerte.
    device = torch.device(cfg["device"])
    # Net.load() si map_location flyttar berre checkpoint-tensorane; modulen
    # sjølv vert verande på CPU (load_state_dict kopierer data inn i dei
    # eksisterande CPU-parametrane). Difor: last på CPU, flytt så modulen.
    net = Net.load(cfg["net"], map_location="cpu").to(device).eval()
    rng = random.Random(cfg["seed"] * 1000 + wid)
    pred = None
    if cfg.get("sims", 0) > 0:
        from .fastplay import Predictor
        pred = Predictor(net, device, cfg.get("batch", 64))
    samples = []
    wins = [0, 0]
    if cfg.get("queen_delay", -1) >= 0:
        cfg["_qd_stats"] = {"viol": 0}
    if cfg.get("nesseby"):
        cfg["_nb_caps"] = []          # [(side, ply)] — kven kapra kongen
    t0 = time.time()
    for g in range(cfg["games_per_worker"]):
        if cfg.get("opponent", "").startswith("classic"):
            w = play_vs_opponent(net, device, cfg, rng, samples, pred)
        else:
            w = play_selfplay_game(net, device, cfg, rng, samples, pred)
        if w is not None:
            wins[w] += 1
        if (g + 1) % int(cfg.get("report_every", 1)) == 0:
            el = max(1e-9, time.time() - t0)
            print(f"[w{wid}] {g+1}/{cfg['games_per_worker']} games, "
                  f"{len(samples)} samples, {el:.0f}s "
                  f"({(g+1)/el*60:.1f} games/min)", flush=True)
    out = os.path.join(cfg["outdir"], f"shard_{wid:03d}.npz")
    save_shard(samples, out)
    nbc = cfg.get("_nb_caps")
    nbtxt = ""
    if nbc is not None:
        m = sum(1 for s, _ in nbc if s == 0)
        g = sum(1 for s, _ in nbc if s == 1)
        avg = (sum(p for _, p in nbc) / len(nbc)) if nbc else 0
        nbtxt = f", Nesseby: king captured by Menn {m} / Giera {g} (mean ply {avg:.0f})"
    qds = cfg.get("_qd_stats")
    qdtxt = ""
    if qds is not None:
        qdtxt = (f", queen handicap: {qds.get('viol', 0)} fallback activations"
                 f" (of {len(qds.get('fallback_soldiers', []))})")
    styletxt = style_summary(cfg)
    print(f"[w{wid}] done: {cfg['games_per_worker']} games, {len(samples)} samples -> {out} "
          f"(Menn wins {wins[0]}, Giera wins {wins[1]}{qdtxt}{nbtxt}{styletxt})", flush=True)
    return len(samples)


def run_selfplay(net_path, outdir, device="cuda", games=64, workers=8,
                 rollouts=6, temp=1.0, noise_alpha=0.3, noise_eps=0.25,
                 seed=1, prior_eps=0.2, tau_q=1.0, k_games=1, batch=64,
                 sims=0, c_puct=1.6, leaf_batch=8, opponent="", opponent_level=3,
                 queen_delay=-1, queen_delay_soldiers=2, max_plies=MAX_PLIES,
                 nesseby=False, nesseby_fraction=1.0,
                 aggressive=False, aggressive_fraction=1.0,
                 styles=False, style_fraction=1.0):
    os.makedirs(outdir, exist_ok=True)
    cfg = {"net": net_path, "device": device, "outdir": outdir,
           "games_per_worker": max(1, games // workers),
           "rollouts": rollouts, "temp": temp,
           "noise_alpha": noise_alpha, "noise_eps": noise_eps, "seed": seed,
           "prior_eps": prior_eps, "tau_q": tau_q,
           "k_games": k_games, "batch": batch,
           "sims": sims, "c_puct": c_puct, "leaf_batch": leaf_batch,
           "opponent": opponent, "opponent_level": opponent_level,
           "queen_delay": queen_delay,
           "queen_delay_soldiers": queen_delay_soldiers,
           "max_plies": max_plies, "nesseby": bool(nesseby),
           "nesseby_fraction": float(nesseby_fraction),
           "aggressive": bool(aggressive),
           "aggressive_fraction": float(aggressive_fraction),
           "styles": bool(styles),
           "style_fraction": float(style_fraction)}
    jobs = [(w, json.dumps(cfg)) for w in range(workers)]
    # Vektorisert løype (K parti i lufta + CUDA-graph) når --k-games > 1
    fn = worker
    if cfg.get("k_games", 1) > 1:
        from . import fastplay
        fn = fastplay.worker_batch
    t0 = time.time()
    if workers == 1:
        tot = fn(jobs[0])
    else:
        # spawn (ikkje fork): far-prosessen kan alt ha ein CUDA-kontekst
        # (treningsløkka), og CUDA kan ikkje re-initialiserast i eit born
        # etter fork. spawn gjev kvar arbeidar sin eigen kontekst.
        with mp.get_context("spawn").Pool(workers) as pool:
            tot = sum(pool.map(fn, jobs))
    print(f"selfplay: ~{games} games, {tot} samples in {time.time()-t0:.0f}s -> {outdir}")
    return tot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="checkpoint .pt (random-init net OK)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rollouts", type=int, default=6)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--noise-alpha", type=float, default=0.3)
    ap.add_argument("--noise-eps", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prior-eps", type=float, default=0.2,
                    help="weight on the net prior in the policy target (0 = outcome only)")
    ap.add_argument("--tau-q", type=float, default=1.0,
                    help="temperature on the outcome weighting, pi ~ n*exp(q/tau)")
    ap.add_argument("--k-games", type=int, default=1,
                    help="vectorised path: K games in flight per worker (1 = classic)")
    ap.add_argument("--batch", type=int, default=64,
                    help="fast batch for CUDA-graph-inferens")
    ap.add_argument("--sims", type=int, default=0,
                    help="MCTS simulations per decision (0 = plain playouts)")
    ap.add_argument("--c-puct", type=float, default=1.6)
    ap.add_argument("--leaf-batch", type=int, default=8)
    ap.add_argument("--queen-delay", type=int, default=-1,
                    help="handicap: -1 off, 0/1 fixed side, 2 = random side per game "
                         "(that side may not activate its queen until --queen-delay-soldiers soldiers remain)")
    ap.add_argument("--queen-delay-soldiers", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=MAX_PLIES,
                    help="ply cap per game in data generation (handicapped games run long)")
    ap.add_argument("--nesseby", action="store_true",
                    help="play with the Nesseby rule (king cannot be captured by landing)")
    ap.add_argument("--nesseby-fraction", type=float, default=1.0,
                    help="share of games with Nesseby when --nesseby is set")
    ap.add_argument("--styles", action="store_true",
                    help="tactic duels: one side 'few'/'kingfwd' vs 'many'/'kingguard'")
    ap.add_argument("--style-fraction", type=float, default=1.0)
    ap.add_argument("--aggressive", action="store_true",
                    help="one side plays aggressively (weight on captures and advances)")
    ap.add_argument("--aggressive-fraction", type=float, default=1.0)
    ap.add_argument("--opponent", default="",
                    help="train against this instead of self-play (e.g. \"classic3\")")
    ap.add_argument("--opponent-level", type=int, default=3)
    a = ap.parse_args()
    run_selfplay(a.net, a.outdir, a.device, a.games, a.workers,
                 a.rollouts, a.temp, a.noise_alpha, a.noise_eps, a.seed,
                 a.prior_eps, a.tau_q, a.k_games, a.batch,
                 a.sims, a.c_puct, a.leaf_batch, a.opponent, a.opponent_level,
                 a.queen_delay, a.queen_delay_soldiers, a.max_plies,
                 a.nesseby, a.nesseby_fraction,
                 a.aggressive, a.aggressive_fraction,
                 a.styles, a.style_fraction)
