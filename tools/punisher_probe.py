"""Kor mange brikker mistar nettet mot straffar-motstandaren?

Sjølvspel kan ikkje måle dette: der spelar båe sider likt, so dei straffar kvar
andre sine feil like lite. `few` slo `many` 322-48 i spegelduell og var likevel
lett å slå for eit menneske. Denne proben let straffaren (menneskestrategien:
aktiver mange, hald dei attende, plukk brikka som står framme) spele mot eit
nett, og tel kor mange UTLØYSTE soldatar nettet mistar.

  .venv/bin/python tools/punisher_probe.py --net engines/nais5-nn.pt --games 20
"""
from __future__ import annotations

import argparse
import random
import sys
import time

import torch

from sahkku_ai import engine as E
from sahkku_ai import fastplay as FP
from sahkku_ai import features as FT
from sahkku_ai import mcts as MCTS
from sahkku_ai import punisher as PU
from sahkku_ai.model import Net


def probe(net_path, games, sims, seed=0, max_plies=2500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Net.load() si map_location flyttar berre tensorane; modulen må flyttast
    # etterpå (same grunn som i selfplay.py).
    net = Net.load(net_path, map_location="cpu").to(device).eval()
    pred = FP.Predictor(net, device)
    cfg = {"sims": sims, "nesseby": True, "nesseby_fraction": 1.0,
           "max_plies": max_plies, "c_puct": 1.4}
    rng = random.Random(seed)
    score = 0.0
    lost_released = lost_all = plies_total = 0
    t0 = time.time()

    for g in range(games):
        net_side = g % 2
        st = E.State(first=rng.randint(0, 1), nesseby=True)
        E.opening_roll(st, rng.random)
        was = {p.id: (p.o, p.released) for p in st.pieces}

        for _ in range(max_plies):
            if st.winner is not None:
                break
            if st.phase == "roll":
                E.roll(st, rng.random)
                st.phase = "move"
            if st.turn == net_side:
                pi, _z = MCTS.search(pred, st, cfg, rng, root_noise=False)
                byid = {FT.move_id(st, m): m for a, m in FT.legal_ids(st)
                        if m.get("pid") is not None}
                mv = byid.get(max(pi, key=pi.get)) if pi else None
                if mv is None:
                    E.pass_remaining(st)
                    continue
                E.apply_move(st, mv)
            else:
                act = PU.pick_action(st)
                if act is None or act.get("kind") in ("pass", "reroll"):
                    if E.can_reroll_x(st):
                        E.reroll_x(st, rng.random)
                        continue
                    E.pass_remaining(st)
                    continue
                E.apply_move(st, act)
            E.settle(st)
            plies_total += 1

        for p in st.pieces:
            w = was.get(p.id)
            if not w or w[0] != net_side or p.o != net_side:
                continue
            if p.captured:
                lost_all += 1
                if p.t == "s" and w[1]:
                    lost_released += 1

        if st.winner == net_side:
            score += 1.0
        elif st.winner is None:
            score += 0.5
        res = ("vann" if st.winner == net_side
               else "uavgjort" if st.winner is None else "tapte")
        print(f"[{g + 1}/{games}] nett som {'Menn' if net_side == 0 else 'Koner'}: "
              f"{res} etter {st.moves} trekk ({time.time() - t0:.0f}s)", flush=True)

    n = games
    print(f"\nstraffar mot {net_path}: {score / n * 100:.1f} % for nettet ({score:.0f}/{n})")
    print(f"nettet miste {lost_released / n:.1f} utløyste soldatar "
          f"og {lost_all / n:.1f} brikker per parti")
    print(f"snitt {plies_total / n:.0f} halvtrekk per parti")
    return score / n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="checkpoint eller eksportert .pt")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=45)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    probe(a.net, a.games, a.sims, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
