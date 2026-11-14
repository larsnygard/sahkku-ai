#!/usr/bin/env python3
"""Verdihovudet sitt helsetal: kor metta er det?

Eit nett som er trena på få prøver med z=±1 kan mette tanh-hovudet: det
svarar ±1.000 i nesten alle stillingar, og då er 1-ply lookahead (LA) og
CAP_MARGIN i JS-motoren berre myntkast — sjå
analysis/blanka3_overnight/FINDINGS.md.

Dette verktøyet svarar på eit tal: kor stor del av stillingane i eit
SPG-parti har |v| over terskelen. Bra: <30 % på 0.9. BLANKA.3: 55 %.

Køyr:
  .venv/bin/python tools/value_health.py <parti.spg> <motor.pt>
      [--steps 7] [--thresh 0.9] [--tag blanka4]
Flagg:
  --side-split   vis metninga òg per side-i-trekket (avdekker side-skjevheit)
  --json         skriv éi JSON-linje (for treningsloggen)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from sahkku_ai import engine as E
from sahkku_ai import features as FT
from sahkku_ai.model import Net
from sahkku_ai.spg import game_from_spg_any
from tools.analysis_stats import split_games


def scan(spg_paths, net_path, every=7, thresh=0.9):
    net = Net.load(net_path, map_location="cpu").eval()
    vals = []
    sides = ([], [])
    for src in spg_paths:
        text = open(src, encoding="utf-8", errors="replace").read()
        for gt in split_games(text):
            pairs = game_from_spg_any(gt, strict=False)
            samples = pairs[1] if isinstance(pairs, tuple) else pairs
            for st, _mv in samples[::max(1, every)]:
                if st.winner is not None:
                    continue
                pl, gl = FT.encode_state(st)
                with torch.no_grad():
                    _, v = net(torch.tensor(pl.reshape(-1)).unsqueeze(0).float(),
                               torch.tensor(gl).unsqueeze(0).float())
                vals.append(float(v[0]))
                sides[st.turn].append(float(v[0]))
    return np.asarray(vals), sides


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spg", nargs="+")
    ap.add_argument("net")
    ap.add_argument("--steps", type=int, default=7,
                    help="sample every Nth ply (default 7)")
    ap.add_argument("--thresh", type=float, default=0.9)
    ap.add_argument("--tag", default="")
    ap.add_argument("--side-split", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    vals, sides = scan(a.spg, a.net, a.steps, a.thresh)
    if len(vals) == 0:
        print("no positions found", file=sys.stderr)
        sys.exit(1)
    sat = float((np.abs(vals) > a.thresh).mean() * 100)
    sat85 = float((np.abs(vals) >= 0.85).mean() * 100)
    row = {"tag": a.tag, "n": int(len(vals)), "mean": round(float(vals.mean()), 3),
           "sat_gt": round(sat, 1), "sat_085": round(sat85, 1),
           "std": round(float(vals.std()), 3), "thresh": a.thresh}
    if a.side_split:
        for s, nm in enumerate(("menn", "giera")):
            if sides[s]:
                arr = np.asarray(sides[s])
                row[f"sat_{nm}"] = round(float((np.abs(arr) > a.thresh).mean() * 100), 1)
    if a.json:
        print(json.dumps(row))
    else:
        print(f"{a.tag or 'net'} n={row['n']}  mean={row['mean']:+.3f}  std={row['std']:.3f}\n"
              f"  |v|>{a.thresh}: {sat:.1f} %  (goal <30 %)   |v|>=0.85: {sat85:.1f} %\n"
              f"  |mean|: {abs(row['mean']):.3f} (goal <0.15 — a mirror game must even out)")
        if a.side_split:
            for s in ("menn", "giera"):
                if f"sat_{s}" in row:
                    print(f"  {s} side: {row[f'sat_{s}']} %")
    # Port: metning OG midtpunkt-drift. Eit nett som svarar ±0.9 overalt
    # kan ha «0 % > 0.9» og likevel vere blindt — difor måler vi òg
    # |v|≥0.85 og gjennomsnittet (i spegelparti skal det vera ~0).
    ok = sat < 30.0 and abs(row["mean"]) < 0.15
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
