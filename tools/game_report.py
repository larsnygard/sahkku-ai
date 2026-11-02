"""Kva skjedde i eit menneskeparti? — aktivitet, utsattheit og kor dronninga gjekk.

Skilnaden frå blunder_scan.py: han spør `legal_moves` i ei stilling der sida
ikkje har kasta terning enno, so svaret er alltid «ingen trekk» og dronninga
vert aldri funnen hengd. Her vert spørsmålet stilt med `features.capture_squares`
— same sanning som motoren, uavhengig av kast — so «kan ho takast NO» faktisk
kan svarast for kvar einaste stilling.

Køyr: .venv/bin/python tools/game_report.py <fil.spg>
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys

from sahkku_ai import engine as E
from sahkku_ai import features as FT
from sahkku_ai.spg import game_from_spg_any, parse_spg


def released(st, side):
    return sum(1 for p in E.soldiers(st, side) if p.released and not p.captured)


def exposed(st, side):
    caps = FT.capture_squares(st, 1 - side)
    return sum(1 for p in E.soldiers(st, side)
               if p.released and not p.captured and E.soldier_at(side, p.at) in caps)


def piece_name(p):
    return {"s": "soldier", "q": "queen", "k": "king"}.get(p.t, p.t)


def report(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    p = parse_spg(txt)
    p = p[0] if isinstance(p, list) else p
    final, samples = game_from_spg_any(txt, strict=True)
    result = p.get("result", "")
    winner = 0 if result == "1-0" else (1 if result == "0-1" else None)
    loser = None if winner is None else 1 - winner
    names = {0: p.get("menn") or "Menn", 1: p.get("koner") or "Koner"}

    print(f"{path.split('/')[-1]}  —  {result}, winner: {names.get(winner, '?')}")
    print(f"{len(samples)} moves recorded\n")

    # ---- 1. aktivitet og utsattheit gjennom partiet ---------------------
    print("move   released (M/K)  exposed (M/K)")
    marks = [10, 20, 30, 40, 50, 60, 70, 80]
    for i, (st, _mv) in enumerate(samples):
        if (i + 1) in marks and st.winner is None:
            print(f"{i + 1:4d}   {released(st, 0):>2} / {released(st, 1):<2}"
                  f"        {exposed(st, 0):>2} / {exposed(st, 1):<2}")

    # snitt over heile partiet, etter at opninga er over
    late = [(released(st, 0), released(st, 1), exposed(st, 0), exposed(st, 1))
            for i, (st, _m) in enumerate(samples) if 12 <= i and st.winner is None]
    if late:
        n = len(late)
        print(f"\naverage after move 12: released {sum(x[0] for x in late) / n:.1f} / "
              f"{sum(x[1] for x in late) / n:.1f}"
              f"   exposed {sum(x[2] for x in late) / n:.1f} / {sum(x[3] for x in late) / n:.1f}")

    # ---- 2. kven mista kva ----------------------------------------------
    print("\ncaptures:")
    for i, (st, mv) in enumerate(samples):
        if not mv:
            continue
        land = E.move_dest(st, mv)
        victims = [q for q in E.occ(st, land) if not q.captured and q.o != st.turn]
        for q in victims:
            print(f"  move {i + 1:3d}: {names[st.turn]} took {names[q.o]}'s "
                  f"{piece_name(q)} on {E.sq_name(land)}")

    # ---- 3. når vart taparsida si dronning hengd? ----------------------
    if loser is not None:
        print()
        for i, (st, mv) in enumerate(samples):
            if st.turn != loser or st.winner is not None:
                continue
            after = samples[i + 1][0] if i + 1 < len(samples) else final
            q = E.queen(after, loser)
            if q is None or q.captured:
                continue
            if q.at in FT.capture_squares(after, winner):
                print(f"{names[loser]}'s queen stood OPEN after move {i + 1} "
                      f"({E.sq_name(q.at)}) — could be taken with one die")
                print(f"   move that exposed her: {mv}")
                taken = next((j for j in range(i + 1, len(samples))
                              if E.queen((samples[j + 1][0] if j + 1 < len(samples) else final),
                                         loser) in (None,) or
                              E.queen((samples[j + 1][0] if j + 1 < len(samples) else final),
                                      loser).captured), None)
                print(f"   actually taken {taken - i} half-moves later"
                      if taken is not None else "   (never taken — the game ended first)")
                break
        else:
            print("found no moment where the queen stood open (she was taken in a "
                  "double move, or stayed covered the whole way)")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        report(f)
