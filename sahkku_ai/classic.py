"""Klassisk CPU (js/cpu.js) — port til Python, so han kan vere MOTSTANDAR i
trening. Porten er litterær: same vekter, same rekkjefølgje, same quirkar,
slik at motstandaren i treninga er den same som spelarane møter i appen.

Nivåa (som i appen):
  1 = tilfeldig
  2 = gamal greedy-poieng med litt støy  (score_move, uendra)
  3 = FLOKK — dei sju reglane (Lars 2026-11-04), heilt deterministisk:
        1. dronninga vert ikkje aktivert med >3 soldatar i live
        2. X løyser soldatar til 6 er aktive
        3. ingen stabling av soldatar når eit fritt felt finst
        4. flokken held saman; den bakarste går fyrst
        5. soldaten slår ALLTID når han kan (hard regel)
        6. kongen attende til heimrekkja
        7. kongen slår berre når han kan attende til trygg grunn same tur
"""
from __future__ import annotations
import random

from . import engine as E

HARD = 200000
DIE_VALUE = {"X": 1, "II": 2, "III": 3, "_": 0}


def _md(st, mv):
    return E.move_dest(st, mv)


# ---------- hjelparar ----------
def _active(st, o):
    return [p for p in E.soldiers(st, o) if p.released and not p.captured]


def _dist(a: int, b: int) -> int:
    ca, ra = E.pt(a)
    cb, rb = E.pt(b)
    dc = abs(ca - cb)
    if dc > 7:
        dc = 15 - dc
    return dc + abs(ra - rb)


def _home_row(o: int) -> int:
    return 0 if o == 0 else 2


def _cycle_dist(a: int, b: int) -> int:
    d = abs(a % 60 - b % 60)
    return min(d, 60 - d)


def _die_value(face: str) -> int:
    return DIE_VALUE.get(face, 0)


def _has_unused_die(st, v: int) -> bool:
    n = min(3, len(st.dice), len(st.used))
    for i in range(n):
        if not st.used[i] and _die_value(st.dice[i]) == v:
            return True
    return False


def _retreat_dir(me: int) -> str:
    return "B" if _home_row(me) == 0 else "F"


def _path_along(sq: int, d: str, n: int) -> int:
    cur = sq
    for _ in range(n):
        c, r = E.pt(cur)
        if d == "F":
            if r == 0:
                cur = E.idx_of(c, 1)
            elif r == 1:
                cur = E.idx_of(c, 2)
            else:
                return -1
        else:
            if r == 2:
                cur = E.idx_of(c, 1)
            elif r == 1:
                cur = E.idx_of(c, 0)
            else:
                return -1
    return cur


def _threat_zone(st, me):
    """Kva felt kan fienden lande på neste tur? (spegl av JS threatZone)"""
    out = set()
    foe = 1 - me
    for p in st.pieces:
        if p.captured or p.o == me:
            continue
        if p.t == "s":
            if not p.released:
                continue
            for k in (1, 2, 3):
                out.add(E.soldier_at(p.o, p.at + k))
        elif p.t == "q" and p.released:
            tq = E.cycle_index(p.o, p.at)
            if tq >= 0:
                for k in (1, 2, 3):
                    out.add(E.soldier_at(p.o, tq + k))
                    out.add(E.soldier_at(p.o, tq - k))
            for k in (1, 2, 3):
                f = _path_along(p.at, "F", k)
                if f >= 0:
                    out.add(f)
                b = _path_along(p.at, "B", k)
                if b >= 0:
                    out.add(b)
        elif p.t == "k" and p.o == foe:
            tk = E.cycle_index(foe, p.at)
            if tk >= 0:
                for k in (1, 2, 3):
                    out.add(E.soldier_at(foe, tk + k))
                    out.add(E.soldier_at(foe, tk - k))
            for k in (1, 2, 3):
                f = _path_along(p.at, "F", k)
                if f >= 0:
                    out.add(f)
                b = _path_along(p.at, "B", k)
                if b >= 0:
                    out.add(b)
    return out


def _king_safe(st, sq: int, me: int) -> bool:
    foe = 1 - me
    for f in st.pieces:
        if f.captured or f.o == me or f.t != "s":
            continue
        if _dist(sq, E.at_idx(st, f)) < 5:
            return False
    fq = E.queen(st, foe)
    if fq is not None and fq.released:
        tq = E.cycle_index(fq.o, fq.at)
        if tq >= 0:
            for k in (1, 2, 3):
                if E.soldier_at(fq.o, tq + k) == sq or E.soldier_at(fq.o, tq - k) == sq:
                    return False
        for k in (1, 2, 3):
            if _path_along(fq.at, "F", k) == sq or _path_along(fq.at, "B", k) == sq:
                return False
    return True


def _can_retreat_safe(st, me: int, sq: int) -> bool:
    d = _retreat_dir(me)
    for v in (1, 2, 3):
        if not _has_unused_die(st, v):
            continue
        dest = _path_along(sq, d, v)
        if dest >= 0 and _king_safe(st, dest, me):
            return True
    return False


# ---------- FLOKK-poieng (nivå 3) ----------
def flokk_score(st, mv) -> int:
    me = st.turn
    foe = 1 - me
    p = E.piece(st, mv["pid"])
    copy = st.clone()
    ev = E.apply_move(copy, mv)
    s = 0

    if copy.winner == me:
        return 1000000

    act_before = _active(st, me)
    my_act_now = len(act_before)
    my_sold_now = len([x for x in E.soldiers(st, me) if not x.captured])
    thr = _threat_zone(st, me)
    land = E.move_dest(st, mv)

    # 1) dronninga ikkje aktivert med >3 soldatar i live
    if p.t == "q" and mv.get("kind") == "rel":
        s += -HARD if my_sold_now > 3 else 120

    # 5) soldaten slår alltid når han kan
    for e in ev:
        if e.get("type") == "kill":
            q = E.piece(copy, e["pid"])
            is_q = q is not None and q.t == "q"
            if p.t == "s":
                s += HARD + (1200 if is_q else 900)
            else:
                s += 1200 if is_q else 950
        if e.get("type") == "king":
            s += HARD + 1400

    # 6 + 7) kongen
    if p.t == "k" and p.o == me:
        kk = E.king(copy)
        land = E.at_idx(copy, kk)
        struck = any(e.get("type") == "kill" and e.get("by") == me for e in ev)
        if struck:
            if not _king_safe(copy, land, me) or not _can_retreat_safe(copy, me, land):
                s -= HARD
        elif mv.get("kind") == "mv" and mv.get("dir") in ("F", "B") and not _king_safe(st, land, me):
            s -= HARD
        r_before = E.pt(E.at_idx(st, p))[1]
        r_after = E.pt(land)[1]
        hr = _home_row(me)
        if r_after == hr:
            s += 1000
        if abs(r_after - hr) < abs(r_before - hr):
            s += 800
        if r_after != hr and r_before == hr:
            s -= 800

    # 2) X løyser soldatar til 6 er aktive
    if mv.get("kind") == "rel" and p.t == "s":
        s += 3000 if my_act_now < 6 else -100

    # 3) ikkje stabling
    if p.t == "s" and mv.get("kind") == "mv":
        stacked = sum(1 for z in E.occ(copy, land)
                      if z.id != p.id and z.t == "s" and z.o == me)
        if stacked:
            s -= 400

    # 4) flokken held saman; den bakarste går fyrst
    if p.t == "s":
        s += (1 if mv.get("kind") == "rel" else mv["v"]) * 15
        here = E.at_idx(st, p)
        if p.released and here in thr and land not in thr:
            s += 200
        if p.released and land in thr:
            s -= 250
        moved = E.piece(copy, mv["pid"])
        if moved is not None and not moved.captured:
            pals = [x for x in _active(copy, me) if x.id != moved.id]
            if pals:
                s -= min(min(_cycle_dist(moved.at, x.at) for x in pals), 10) * 25
        if p.released and act_before:
            min_at = min(x.at for x in act_before)
            if p.at == min_at:
                s += 80

    # eiga dronning: i spel, men ikkje åleine i faret
    my_q = E.queen(st, me)
    if my_q is not None:
        if E.queen(copy, me) is None:
            s -= 40000
        if p.t == "q" and p.released:
            if land in thr and len(_active(copy, me)) < 4:
                s -= 350
            s += 30

    k_after = E.king(copy)
    if k_after is not None and k_after.o == me and E.at_idx(copy, k_after) in thr and p.t != "k":
        s -= 350
    if k_after is not None and k_after.o == foe:
        s += 20
    return s


def flokk_pick(st):
    lm = E.legal_moves(st, st.turn)
    if not lm:
        return None
    best, bs = lm[0], None
    for mv in lm:
        sc = flokk_score(st, mv)
        if bs is None or sc > bs:
            bs, best = sc, mv
    return best


# ---------- gamal greedy (nivå 2) ----------
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
            hc = E.pt(here)[0]
            dcol = E.pt(dest)[0]
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
    """Nivå 1 = tilfeldig, 2 = grådig + støy, 3 = FLOKK (deterministisk)."""
    rnd = rnd or random.random
    lm = E.legal_moves(st, st.turn)
    if not lm:
        return None
    if level <= 1:
        return lm[int(rnd() * len(lm))]
    if level >= 3:
        return flokk_pick(st)
    best, bs = [], None
    for mv in lm:
        s = score_move(st, mv) + (rnd() - 0.5) * 250
        if bs is None or s > bs:
            bs, best = s, [mv]
        elif s == bs:
            best.append(mv)
    return best[int(rnd() * len(best))]
