#!/usr/bin/env python3
"""Kva gjer policyen? Mål på trekk-kvaliteten i SPG-parti.

For kvart nett-trekk i fila: rekn policy-argmax (lovlig maskert) og samanlikn
med det faktiske trekket (som er motoren sitt val = argmax + lookahead +
capture-margin). Mål:

  * avvik-rate: policy-argmax != motoren sitt val
  * tryggleik: del av policy-argmax-trekk som LANDAR på felt fienden kan slå
    på no (「hengande」) — mot delt av motoren sine faktiske trekk
  * «dødande»-rate: policy-argmax flyttar si eiga DRONNING til slagfelt

Køyr: .venv/bin/python tools/policy_audit.py <fil.spg> <motor.pt>
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
    net = Net.load(netpath, map_location="cpu").eval()
    text = open(src, encoding="utf-8", errors="replace").read()
    n = dev = hang_pol = hang_eng = qpol = nq = 0
    for gi, gt in enumerate(split_games(text)):
        pairs = game_from_spg_any(gt, strict=False)
        samples = pairs[1] if isinstance(pairs, tuple) else pairs
        for st, mv in samples:
            if st.winner is not None or not mv or mv.get("pid") is None:
                continue
            pl, gl = FT.encode_state(st)
            legal = FT.legal_ids(st)
            mask = np.zeros(FT.N_ACTIONS, np.float32)
            for aid, _ in legal:
                mask[aid] = 1.0
            with torch.no_grad():
                praw, _ = net(torch.tensor(pl.reshape(-1)).unsqueeze(0).float(),
                              torch.tensor(gl).unsqueeze(0).float())
            praw = praw[0].numpy()
            praw[mask < 0.5] = -1e9
            aid = int(np.argmax(praw))
            _, amv = next(x for x in legal if x[0] == aid)
            foe = 1 - st.turn
            caps = FT.capture_squares(st, foe)
            a_land = E.move_dest(st, amv)
            e_land = E.move_dest(st, mv)
            n += 1
            if aid != FT.move_id(st, mv):
                dev += 1
            # berre vanlege flyttar tel (løysingar er faste felt, tel ikkje)
            if amv.get("kind") == "mv":
                nq += 1
                if a_land in caps:
                    hang_pol += 1
                ap = E.piece(st, amv["pid"])
                if ap.t == "q" and a_land in caps:
                    qpol += 1
            if mv.get("kind") == "mv" and e_land in caps:
                hang_eng += 1
    print(f"trekk: {n}   policy-argmax != motorval: {dev}/{n} ({100*dev/n:.1f} %)")
    print(f"policy-argmax flyttar ut på slagfelt:  {hang_pol}/{nq} ({100*hang_pol/max(1,nq):.1f} %)")
    print(f"motoren (med LA4+CAP_MARGIN) same tal: {hang_eng}/{nq} — skilnad = lookaheads sin fortjeneste")
    print(f"policy-argmax hengar DRONNINGA si:     {qpol}/{nq} ({100*qpol/max(1,nq):.2f} %)")


if __name__ == "__main__":
    main()
