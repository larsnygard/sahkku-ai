"""Kor mykje gir sims — og kva kostar dei?

For eit sett stillingar frå eit verkeleg parti: køyr MCTS med fleire budsjett
og mål (a) tid per avgjerd, (b) kor skarp policyen (pi) er, og (c) kor ofte
valet er det same som eit 400-sim-søk ville teke. (c) er det viktigaste: det
seier kor mykje søket legg til oppå nettet sin eigen prior, altså kor mykje
læringssignal målet ber.

Køyr: .venv/bin/python tools/sims_probe.py [checkpoint] [fil.spg]
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import math
import random
import sys
import time

import numpy as np
import torch

from sahkku_ai import engine as E
from sahkku_ai import mcts as MCTS
from sahkku_ai.fastplay import Predictor
from sahkku_ai.model import Net
from sahkku_ai.spg import game_from_spg_any

BUDGETS = [5, 10, 20, 45, 90, 180]
REF = 400
N_POS = 20


def entropy(pi: dict) -> float:
    return -sum(p * math.log(max(p, 1e-12)) for p in pi.values() if p > 0)


def positions_from_spg(path: str, n: int):
    txt = open(path, encoding="utf-8", errors="replace").read()
    _final, samples = game_from_spg_any(txt, strict=True)
    states = [st for st, _ in samples if len(E.legal_moves(st, st.turn)) >= 2]
    if len(states) <= n:
        return states
    step = len(states) / n
    return [states[int(i * step)] for i in range(n)]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/nais2/c4.pt"
    spg = sys.argv[2] if len(sys.argv) > 2 else \
        "tests/fixtures/human/menneske-2026-10-29T1226Z.spg"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net.load(ckpt, map_location="cpu").to(dev).eval()
    pred = Predictor(net, dev, 8)

    states = positions_from_spg(spg, N_POS)
    print(f"nett: {ckpt}  stillingar: {len(states)}  device: {dev}\n")

    base = {"c_puct": 1.6, "leaf_batch": 8, "temp": 1.0,
            "noise_alpha": 0.0, "noise_eps": 0.0}

    def run(st, sims):
        cfg = dict(base, sims=sims)
        t0 = time.time()
        pi, _z = MCTS.search(pred, st, cfg, random.Random(7), root_noise=False)
        return pi, (time.time() - t0) * 1000.0

    # referanse: 400 sim
    ref = []
    t0 = time.time()
    for st in states:
        pi, _ = run(st, REF)
        ref.append(max(pi.items(), key=lambda kv: kv[1])[0] if pi else None)
    print(f"referanse ({REF} sim) tok {time.time() - t0:.0f}s\n")

    print(f"{'sims':>5} {'ms/avgjerd':>11} {'entropi':>8} {'topp-masse':>11} {'= 400-sim':>10}")
    for s in BUDGETS:
        ms, ent, top, agree = [], [], [], 0
        for st, r in zip(states, ref):
            pi, dt = run(st, s)
            if not pi:
                continue
            ms.append(dt)
            ent.append(entropy(pi))
            top.append(max(pi.values()))
            best = max(pi.items(), key=lambda kv: kv[1])[0]
            agree += int(best == r)
        n = len(ms)
        print(f"{s:5d} {np.mean(ms):11.0f} {np.mean(ent):8.2f} {np.mean(top):11.2f} "
              f"{100.0 * agree / max(1, n):9.0f} %")


if __name__ == "__main__":
    main()
