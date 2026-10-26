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
    st = E.State(first=rng.randint(0, 1))
    E.opening_roll(st, rng.random)
    outcome_idx = []      # (indeks i samples_out, sida-som-trekte) for MCTS
    for _ in range(MAX_PLIES):
        if st.winner is not None:
            break
        if st.phase == "roll":
            E.roll(st, rng.random)
            st.phase = "move"
            continue
        ids = FT.legal_ids(st)
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
        pol = np.zeros(FT.N_ACTIONS, np.float32)
        for a in legal:
            pol[a] = pi[a]
        samples_out.append((planes.reshape(-1), g, pol, np.float32(z)))
        if pred is not None and cfg.get("sims", 0) > 0:
            outcome_idx.append((len(samples_out) - 1, st.turn))
        # Play the outcome-weighted best action (π), not the raw prior.
        best = max(legal, key=lambda a: pi[a])
        apply_choice(st, ids, best, rng)
    # MCTS: verdimålet er PARTIRESULTATET (AlphaZero-konvensjonen), ikkje
    # søkets eige verdianslag — elles vert måla mjuke tal som ikkje lærer
    # nettet kven som faktisk vann.
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
    t0 = time.time()
    for g in range(cfg["games_per_worker"]):
        w = play_selfplay_game(net, device, cfg, rng, samples, pred)
        if w is not None:
            wins[w] += 1
        if (g + 1) % 5 == 0:
            el = max(1e-9, time.time() - t0)
            print(f"[w{wid}] {g+1}/{cfg['games_per_worker']} parti, "
                  f"{len(samples)} prøver, {el:.0f}s "
                  f"({(g+1)/el*60:.1f} parti/min)", flush=True)
    out = os.path.join(cfg["outdir"], f"shard_{wid:03d}.npz")
    save_shard(samples, out)
    print(f"[w{wid}] ferdig: {cfg['games_per_worker']} parti, {len(samples)} prøver -> {out} "
          f"(Menn-vinn {wins[0]}, Giera-vinn {wins[1]})", flush=True)
    return len(samples)


def run_selfplay(net_path, outdir, device="cuda", games=64, workers=8,
                 rollouts=6, temp=1.0, noise_alpha=0.3, noise_eps=0.25,
                 seed=1, prior_eps=0.2, tau_q=1.0, k_games=1, batch=64,
                 sims=0, c_puct=1.6, leaf_batch=8):
    os.makedirs(outdir, exist_ok=True)
    cfg = {"net": net_path, "device": device, "outdir": outdir,
           "games_per_worker": max(1, games // workers),
           "rollouts": rollouts, "temp": temp,
           "noise_alpha": noise_alpha, "noise_eps": noise_eps, "seed": seed,
           "prior_eps": prior_eps, "tau_q": tau_q,
           "k_games": k_games, "batch": batch,
           "sims": sims, "c_puct": c_puct, "leaf_batch": leaf_batch}
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
    print(f"selfplay: ~{games} parti, {tot} prøver på {time.time()-t0:.0f}s -> {outdir}")
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
                    help="vekt på nett-prioren i policy-målet (0 = berre utfall)")
    ap.add_argument("--tau-q", type=float, default=1.0,
                    help="temperatur på utfallsvektinga, π ∝ n·exp(q/tau)")
    ap.add_argument("--k-games", type=int, default=1,
                    help="vektorisert løype: K parti i lufta per arbeidar (1 = klassisk)")
    ap.add_argument("--batch", type=int, default=64,
                    help="fast batch for CUDA-graph-inferens")
    ap.add_argument("--sims", type=int, default=0,
                    help="MCTS-simuleringar per avgjerd (0 = dagens utspelingar)")
    ap.add_argument("--c-puct", type=float, default=1.6)
    ap.add_argument("--leaf-batch", type=int, default=8)
    a = ap.parse_args()
    run_selfplay(a.net, a.outdir, a.device, a.games, a.workers,
                 a.rollouts, a.temp, a.noise_alpha, a.noise_eps, a.seed,
                 a.prior_eps, a.tau_q, a.k_games, a.batch,
                 a.sims, a.c_puct, a.leaf_batch)
