#!/usr/bin/env python3
"""Dronning-autopsi: for tapt-parti, sjå på augneblikket dronninga vart slått.

For kvart dronningeslag: Finn taparen sin siste handling FØR slaget, og skriv
policy-toppene (probs + decoded) + verdien nettet gav den stillinga. Dermed
ser ein om nettet VALDE det dødelege trekket (policy-vekt høgst) eller om
trekket var trygt og dronninga vart tekne seinare i same kasting.

Køyr: .venv/bin/python tools/queen_autopsy.py <fil.spg> <motor.pt> [max slag]
"""
from __future__ import annotations
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


def main():
    src, netpath = sys.argv[1], sys.argv[2]
    maxk = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    net = Net.load(netpath, map_location="cpu").eval()
    text = open(src, encoding="utf-8", errors="replace").read()
    shown = 0
    for gi, gt in enumerate(split_games(text)):
        pairs = game_from_spg_any(gt, strict=False)
        samples = pairs[1] if isinstance(pairs, tuple) else pairs
        final = pairs[0] if isinstance(pairs, tuple) else None
        n = len(samples)
        for i in range(n):
            st, mv = samples[i]
            if st.winner is not None or not mv or mv.get("pid") is None:
                continue
            after = samples[i + 1][0] if i + 1 < n else final
            side = st.turn
            foe = 1 - side
            land = E.move_dest(st, mv)
            vic = [q for q in E.occ(st, land) if not q.captured and q.o is not None and q.o != side]
            if not any(q.t == "q" for q in vic):
                continue
            # taparen = foe. Finn foe sin siste handling før ply i
            autopsi = None
            for j in range(i - 1, -1, -1):
                stj, mvj = samples[j]
                if stj.turn == foe and mvj and mvj.get("pid") is not None:
                    autopsi = (j, stj, mvj)
                    break
            if autopsi is None:
                continue
            j, stj, mvj = autopsi
            pl, gl = FT.encode_state(stj)
            ids = [aid for aid, _ in FT.legal_ids(stj)]
            mask = np.zeros(FT.N_ACTIONS, np.float32)
            for a in ids:
                mask[a] = 1.0
            x = torch.tensor(pl.reshape(-1), dtype=torch.float32).unsqueeze(0)
            g = torch.tensor(gl, dtype=torch.float32).unsqueeze(0)
            mk = torch.tensor(mask).unsqueeze(0)
            with torch.no_grad():
                praw, val = net(x, g)
                probs = torch.softmax(praw[0], dim=0).numpy()
            legal = []
            for aid, mv_ in FT.legal_ids(stj):
                kind, letter, frm, to, sd = FT.decode_id(aid)
                leg = f"{kind}:{letter}{E.sq_name(frm) if frm is not None else '-'}->{E.sq_name(to) if to is not None else '-'}"
                legal.append((float(probs[aid]), leg, aid == FT.move_id(stj, mvj)))
            legal.sort(reverse=True)
            chosen = next((l for l in legal if l[2]), None)
            print(f"\n--- parti {gi} ply {i+1}: {side} tek dronninga ({mv})")
            print(f"    tapar-siste-trekk (ply {j+1}): {mvj}")
            print(f"    nettverdi stilling: {float(val[0]):+.3f}")
            print(f"    TOPP: " + " | ".join(f"{p:.3f} {d}{' <VALT' if c else ''}" for p, d, c in legal[:5]))
            if chosen and chosen not in legal[:5]:
                print(f"    det dødelege trekket: rank {legal.index(chosen)+1}/{len(legal)} prob {chosen[0]:.3f}")
            shown += 1
            if shown >= maxk:
                return


if __name__ == "__main__":
    main()
