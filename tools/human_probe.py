"""Måler det treningskøyringa faktisk tok sikte på: ser NAIS.3 farane som NAIS.2 gjekk i?

For kvart menneskeparti: finn stillingane der taparen (NAIS.2, Koner) gjorde
det avgjerande trekket — og sjå kva NAIS.2 og NAIS.3 meiner om dei stillingane:

  p(trekk)  = policy-sannsynet nettet gjev trekket som faktisk vart spelt
  v         = verdien nettet gjev stillinga (frå sida som er i trekket)

Gjekk læringa rett veg, skal NAIS.3 gje det tapande trekket LÅGARE
sannsyn (og/eller lågare verdi) enn NAIS.2, i same stilling.

Køyr: .venv/bin/python tools/human_probe.py <fil.spg> [...]
"""
from __future__ import annotations
import sys

import numpy as np
import torch

from sahkku_ai import engine as E
from sahkku_ai import features as FT
from sahkku_ai.model import Net
from sahkku_ai.spg import parse_spg, game_from_spg_any

CKPTS = [("NAIS.2", "checkpoints/nais2/c4.pt"), ("NAIS.3", "checkpoints/nais3/c2.pt")]


def policy_of(net, st):
    """(sannsyn per lovleg action-id, verdi) for stillinga."""
    planes, g = FT.encode_state(st)
    x = torch.from_numpy(planes.reshape(1, -1)).float()
    gv = torch.from_numpy(np.asarray(g, np.float32).reshape(1, -1)).float()
    with torch.no_grad():
        pol, val = net(x, gv)
    pol = pol[0].numpy()
    legal = [a for a, _ in FT.legal_ids(st)]
    if not legal:
        return {}, float(val.item())
    z = pol[legal]
    z = np.exp(z - z.max())
    z = z / z.sum()
    return {a: float(p) for a, p in zip(legal, z)}, float(val.item())


def critic_positions(path):
    """Stillingane rett før taparen gjorde det som kosta dronninga."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    p = parse_spg(txt)
    p = p[0] if isinstance(p, list) else p
    res = p.get("result")
    if res not in ("1-0", "0-1"):
        return []
    winner = 0 if res == "1-0" else 1
    loser = 1 - winner
    final, samples = game_from_spg_any(txt, strict=True)
    out = []
    # siste stilling der taparen var i trekket, og dei tre siste tapar-trekka
    idx = [i for i, (st, _) in enumerate(samples) if st.turn == loser]
    for i in idx[-4:]:
        st, mv = samples[i]
        aid = FT.move_id(st, mv)
        out.append((i + 1, st, mv, aid))
    return out


def main(files):
    nets = {}
    for name, ck in CKPTS:
        n = Net.load(ck, map_location="cpu")
        n.eval()
        nets[name] = n
        print(f"lasta {name} frå {ck}")

    for path in files:
        print(f"\n=== {path.split('/')[-1]} ===")
        for ply, st, mv, aid in critic_positions(path):
            row = []
            for name, net in nets.items():
                probs, val = policy_of(net, st)
                row.append((name, probs.get(aid, 0.0), val))
            print(f"  tapar-trekk {ply} ({E.sq_name(E.move_dest(st, mv))}):")
            for name, pr, val in row:
                print(f"    {name}: p(trekk) {pr:.3f} · verdi {val:+.3f}")
            if len(row) == 2:
                d_p = row[1][1] - row[0][1]
                d_v = row[1][2] - row[0][2]
                print(f"    skillnad: p {d_p:+.3f} · verdi {d_v:+.3f} "
                      f"({'betre' if d_p < 0 or d_v > 0 else 'uendra/verre'})")


if __name__ == "__main__":
    main(sys.argv[1:])
