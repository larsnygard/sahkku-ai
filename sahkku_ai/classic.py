"""Klassisk CPU (js/cpu.js) — port til Python, so han kan vere MOTSTANDAR i
trening. Nivå 3 = deterministisk grådig (den sterke), nivå 2 = same + støy.

Porten er litterær (òg `mod(..., 75)`-quirket i framdriftstermen), slik at
motstandaren i treninga er den same som spelarane møter i appen.
"""
from __future__ import annotations
import random

from . import engine as E


def _md(st, mv):
    return E.move_dest(st, mv)


def score_move(st, mv) -> int:
    p = E.piece(st, mv["pid"])
    me = st.turn
    s = 0
    copy = st.clone()
    ev = E.apply_move(copy, mv)
    if copy.winner == me:
        return 100000
    for e in ev:
        if e.get("type") == "kill":
            q = E.piece(copy, e["pid"])
            s += 50000 if (q and q.t == "q") else 900
        if e.get("type") == "king":
            s += 1400
    if p.t == "s" and p.o == me:
        step = 1 if mv.get("kind") == "rel" else mv["v"]
        prog = ((p.at + step) % 75) - p.at
        s += prog * 12
        fq = E.queen(st, 1 - me)
        if fq is not None:
            here = E.at_idx(st, p)
            dest = E.soldier_at(me, p.at + step)
            fqi = E.at_idx(st, fq)
            hc, dcol = E.pt(here)[0], E.pt(dest)[0]
            fcol = E.pt(fqi)[0]
            d1 = min(abs(hc - fcol), 15 - abs(hc - fcol))
            d2 = min(abs(dcol - fcol), 15 - abs(dcol - fcol))
            s += (d1 - d2) * 30
        if not p.released:
            s += 260
    if p.t == "k" and p.o == me:
        s += 60
    if p.t == "q" and p.o == me:
        s += 120
        dest = _md(st, mv)
        for f in E.soldiers(st, 1 - me):
            if not f.released or f.captured:
                continue
            for v in (1, 2, 3):
                if E.soldier_at(1 - me, f.at + v) == dest:
                    s -= 800
    myq = E.queen(st, me)
    if myq is not None and E.queen(copy, me) is None:
        s -= 40000
    k = E.king(copy)
    if k is not None and k.o == me:
        kcol = E.pt(E.at_idx(copy, k))[0]
        for f2 in E.soldiers(copy, 1 - me):
            if not f2.released or f2.captured:
                continue
            fcol = E.pt(E.at_idx(copy, f2))[0]
            if abs(kcol - fcol) <= 3:
                s -= 120
    return s


def pick(st, level=3, rnd=None):
    """Nivå 1 = tilfeldig, 2 = grådig + støy, 3 = deterministisk grådig."""
    rnd = rnd or random.random
    lm = E.legal_moves(st, st.turn)
    if not lm:
        return None
    if level <= 1:
        return lm[int(rnd() * len(lm))]
    best, bs = [], None
    for mv in lm:
        s = score_move(st, mv)
        if level == 2:
            s += (rnd() - 0.5) * 250
        if bs is None or s > bs:
            bs, best = s, [mv]
        elif s == bs:
            best.append(mv)
    return best[int(rnd() * len(best))]
