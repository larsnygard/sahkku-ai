"""Taktikk-stilane for NAIS.6: few/many og kingfwd/kingguard.

Testen pinnar (a) at vektinga faktisk flyttar policyen i rett retning, (b) at
sidene byter stil kvar kamp, og (c) at stilen hamnar i POLICY-MÅLET — ikkje
berre i trekket som vert spelt. Det siste var ein ekte feil i den gamle
aggressive-køyringa: ho vekta π etter at målet var lagra, so nettet lærte
berre av stillingane som oppstod.

Køyr: .venv/bin/python -m tests.test_styles
"""
from __future__ import annotations
import random
import sys

from sahkku_ai import engine as E
from sahkku_ai import features as FT
from sahkku_ai import selfplay as SP

PASS = FAIL = 0


def ok(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" :: {extra}" if extra else ""))


def fresh(turn=0):
    st = E.State(first=0, nesseby=True)
    rnd = random.Random(5)
    E.opening_roll(st, rnd.random)
    st.turn = turn
    E.roll(st, rnd.random)
    st.phase = "move"
    return st


def weights(st, style):
    """π jamt over alle lovlege trekk, før og etter stilen."""
    ids = FT.legal_ids(st)
    base = {a: 1.0 / len(ids) for a, _ in ids}
    return base, SP.reweight_style(st, ids, base, style), ids


def main():
    # ---- 1. few vs many: løysing -----------------------------------------
    st = fresh(0)
    # Merk to soldatar som løyste (det er talet «few»-regelen les), og set
    # terningane til X III II so det finst ei løysing å vekte.
    sold = E.soldiers(st, 0)
    for q in sold[:2]:
        q.released = True
    st.dice = ["X", "III", "II"]
    st.used = [False, False, False]
    st.phase = "move"
    base, few, ids = weights(st, "few")
    _, many, _ = weights(st, "many")
    rel_ids = [a for a, m in ids if m.get("kind") == "rel"]
    ok("det finst ei løysing å måle på", bool(rel_ids), "ingen release-trekk")
    if rel_ids:
        # Sannsynet vert renormalisert over alle lovlege trekk, so vi samanliknar
        # løysinga mot eit anna trekk i staden for mot sin eigen basisvekt:
        # forholdet er fritt for normaliseringa.
        r = rel_ids[0]
        other = next((a for a, m in ids if m.get("kind") != "rel"), None)
        ok("det finst eit «anna trekk» å jamføre med", other is not None)
        if other is not None:
            b_ratio = base[r] / base[other]
            ok("«few» gjer løysing relativt mindre attraktiv",
               few[r] / few[other] < b_ratio * 0.5,
               f"{few[r] / few[other]:.2f} mot {b_ratio:.2f}")
            ok("«many» gjer løysing relativt meir attraktiv",
               many[r] / many[other] > b_ratio * 2.0,
               f"{many[r] / many[other]:.2f} mot {b_ratio:.2f}")

    # ---- 2. kongestilane -------------------------------------------------
    st = fresh(0)
    king = E.king(st)
    king.o = 0                     # Menn eig kongen
    ids = FT.legal_ids(st)
    kmoves = [(a, m) for a, m in ids if E.piece(st, m["pid"]).t == "k"]
    ok("kongen har trekk å vekte", len(kmoves) >= 2, len(kmoves))
    if len(kmoves) >= 2:
        base, fwd, _ = weights(st, "kingfwd")
        _, guard, _ = weights(st, "kingguard")
        eh = 2                                    # Menn går mot rad 2
        rank0 = abs(E.pt(E.at_idx(st, king))[1] - eh)
        adv = {a: abs(E.pt(E.move_dest(st, m))[1] - eh) - rank0 for a, m in kmoves}
        fwd_ids = [a for a in adv if adv[a] < 0]
        back_ids = [a for a in adv if adv[a] > 0]
        if fwd_ids:
            ok("«kingfwd» lyfter kongetrekk mot fienden",
               all(fwd[a] > base[a] for a in fwd_ids), {a: round(fwd[a], 4) for a in fwd_ids})
            ok("«kingguard» dempar same trekk",
               all(guard[a] < base[a] for a in fwd_ids), {a: round(guard[a], 4) for a in fwd_ids})
        if back_ids:
            ok("«kingguard» lyfter kongetrekk bort frå fienden",
               all(guard[a] > base[a] for a in back_ids), {a: round(guard[a], 4) for a in back_ids})

    # ---- 3. sidene byter stil -------------------------------------------
    cfg = {"styles": True, "style_fraction": 1.0}
    rnd = random.Random(3)
    pairs = [SP.style_pair_for_game(cfg, rnd) for _ in range(200)]
    ok("alle kampar har eit stil-par", all(p[0] and p[1] for p in pairs))
    gyldige = {frozenset(x) for x in SP.PAIRS}
    ok("para kjem frå PAIRS", all(frozenset(p) in gyldige for p in pairs), pairs[:3])
    few_as_p0 = sum(1 for p in pairs if p[0] in ("few", "many"))
    ok("sidene er bytte om lag halvparten av gongene", 60 <= few_as_p0 <= 140, few_as_p0)
    cfg_off = {"styles": False}
    ok("utan --styles vert ingen stil sett",
       all(SP.style_pair_for_game(cfg_off, rnd) == (None, None) for _ in range(20)))

    # ---- 4. stilen MÅ liggje i policy-målet ------------------------------
    # Den gamle aggressive-køyringa vekta π ETTER at målet var lagra. Her
    # pinnar vi rekkjefylgja i kjelda: vektinga må kome før «pol[a] = pi[a]».
    src = open(SP.__file__, encoding="utf-8").read()
    i_vekt = src.index("pi = reweight_style(")
    i_pol = src.index("pol[a] = pi[a]")
    ok("stilen vert vekta FØR policy-målet vert skrive", i_vekt < i_pol,
       f"reweight@{i_vekt} vs pol@{i_pol}")
    import numpy as np
    class StubNet:                        # nett som gjev jamn prior
        device = "cpu"
        def __call__(self, x, g):
            import torch
            n = x.shape[0]
            return torch.zeros(n, FT.N_ACTIONS), torch.zeros(n, 1)
    # Ein full kamp med MCTS er for tungt for ein einingstest; vi sjekkar
    # i staden at play_selfplay_game med sims=0 og stilar produserer prøver
    # (regresjonsvern for at hooken ikkje krasjar) og at stilen vart protokollført.
    cfg = {"net": "x", "device": "cpu", "outdir": "/tmp", "games_per_worker": 1,
           "rollouts": 2, "temp": 1.0, "noise_alpha": 0.0, "noise_eps": 0.0,
           "seed": 1, "prior_eps": 0.0, "tau_q": 1.0, "k_games": 1, "batch": 4,
           "sims": 0, "c_puss": 1.6, "c_puct": 1.6, "leaf_batch": 1,
           "opponent": "", "opponent_level": 3, "queen_delay": -1,
           "queen_delay_soldiers": 2, "max_plies": 120, "nesseby": True,
           "nesseby_fraction": 1.0, "aggressive": False, "aggressive_fraction": 0.0,
           "styles": True, "style_fraction": 1.0}
    out = []
    try:
        SP.play_selfplay_game(StubNet(), "cpu", cfg, random.Random(11), out, None)
        ok("ein kamp med stilar køyrer", True)
        ok("kampen vart protokollført som stil-duell", bool(cfg.get("_style_games")),
           list(cfg.keys())[-4:])
        if cfg.get("_style_games"):
            g = cfg["_style_games"][0]
            ok("loggen har stil, vinnar og åtferd",
               all(k in g for k in ("s0", "s1", "winner", "rel40", "krank")), list(g))
    except Exception as e:                # noqa: BLE001
        ok("ein kamp med stilar køyrer", False, f"{type(e).__name__}: {e}")

    print(f"test_styles: {PASS} pass, {FAIL} fail")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
