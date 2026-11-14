#!/usr/bin/env python3
"""Strafferate: kven tek dronninga, og kor raskt?

For kvar side i kvart parti:
  * DRONEFELT: tal på plyar der motstandaren sin dronning stod på eit felt
    SIDA SJØLV kunne slå på (d.e. ho var slagbar for oss no)
  * SLÅTT: tal på dronningeslag sida gjennomførde
  * TAP: om sida tapte med dronninga si slått

Køyr: .venv/bin/python tools/punish_rate.py <fil.spg>
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sahkku_ai import engine as E
from sahkku_ai import features as FT
from sahkku_ai.spg import parse_spg, game_from_spg_any
from tools.analysis_stats import split_games


def main():
    src = sys.argv[1]
    text = open(src, encoding="utf-8", errors="replace").read()
    tot = [0, 0]          # dronningfelt per side (sum over parti)
    caps = [0, 0]         # dronningeslag gjennomførde
    games = [0, 0]        # parti der sida sin dronning vart teken
    nq = 0                # parti totalt
    for gt in split_games(text):
        head = parse_spg(gt)
        head = head[0] if isinstance(head, list) else head
        pairs = game_from_spg_any(gt, strict=False)
        samples = pairs[1] if isinstance(pairs, tuple) else pairs
        final = pairs[0] if isinstance(pairs, tuple) else None
        n = len(samples)
        nq += 1
        for i in range(n):
            st, mv = samples[i]
            if st.winner is not None:
                continue
            side = st.turn
            foe = 1 - side
            qf = E.queen(st, foe)
            if qf is not None and not qf.captured:
                caps_me = FT.capture_squares(st, side)
                if qf.at in caps_me:
                    tot[side] += 1
            if mv and mv.get("pid") is not None:
                land = E.move_dest(st, mv)
                vic = [q for q in E.occ(st, land) if not q.captured and q.o is not None and q.o != side]
                if any(q.t == "q" for q in vic):
                    caps[side] += 1
        for s in (0, 1):
            fin = final if final is not None else (samples[-1][0] if samples else None)
            q = E.queen(fin, s) if fin is not None else None
            if q is None or q.captured:
                games[s] += 1
    print(f"parti: {nq}")
    for s, nm in enumerate(("Menn", "Giera")):
        opp = tot[s] / max(1, nq)
        print(f"{nm}: dronninga til motstandaren slagbar {opp:.1f} ply/parti i snitt | "
              f"dronningeslag utførte {caps[s] / nq:.2f}/parti | eiga dronning teken i {games[s]}/{nq} parti")


if __name__ == "__main__":
    main()
