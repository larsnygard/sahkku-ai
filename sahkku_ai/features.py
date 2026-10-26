"""Feature encoder + canonical action space.

Board features: 3x15 grid planes, absolute orientation (sáhkku cycles are
side-asymmetric, so no flipping augmentation; side-to-move is a plane).

Action space (fixed, dense):
  RELEASE:  side(2) x class(soldier|queen) x target square(45)   = 180
  MOVE:     class(M|K|Q|k) x from square(45) x to square(45)     = 8100
  SPECIAL:  pass, reroll                                         = 2
Total = 8282. A move is the canonical identity (letter, from, to) shared
with the parity probe; legality masking happens at search/decision time.
"""
from __future__ import annotations
import numpy as np
from . import engine as E

# ---- geometry ----
B_H, B_W = 3, 15
SQQ = 45  # squares
REL_BASE = 0
MOVE_BASE = 180
SPEC_BASE = 180 + 8100
N_ACTIONS = 8100 + 180 + 2

CLASSES = ["M", "K", "Q", "k"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

N_PLANES = 16   # see encode_state
N_GLOBAL = 10   # global scalar vector


def sq_to_hw(i: int):
    c, r = E.pt(i)
    return r, c     # row 0 = rank1 (menn home) .. row 2 = rank3


def encode_state(st: E.State):
    """Return (planes[N_PLANES,3,15], globals[N_GLOBAL]) float32, side POV."""
    pl = np.zeros((N_PLANES, B_H, B_W), np.float32)
    def S(i, r, c):
        pl[i, r, c] = 1.0
    def add_count(i, r, c, v):
        pl[i, r, c] += v
    for p in st.pieces:
        if p.captured:
            continue
        sq = E.soldier_at(p.o, p.at) if p.t == "s" else p.at
        r, c = sq_to_hw(sq)
        if p.t == "s":
            if not p.released:
                S(2 if p.o == 0 else 3, r, c)              # home rank
            else:
                S(0 if p.o == 0 else 1, r, c)              # released
                add_count(12 + p.o, r, c, 1.0 / 15.0)      # stack count
                # second-pass marker (rank 2 squares visited twice)
                if r == 1:
                    first = _first_pass(p.o, sq)
                    if first is not None and p.at % 60 != first:
                        S(14 if p.o == 0 else 15, r, c)
        elif p.t == "q":
            if p.released:
                S(4 if p.o == 0 else 5, r, c)
            else:
                S(6 if p.o == 0 else 7, r, c)
        elif p.t == "k":
            S(8 if p.o == 0 else (9 if p.o == 1 else 10), r, c)
    # active dice face planes (rank-order active slot)
    rk = E.active_rank(st)
    v = E.active_value(st)
    if st.phase == "move" and rk is not None:
        if rk == E.RANK["X"]:
            S(11, 0, 0)
        elif rk == E.RANK["III"]:
            S(11, 0, 1)
        else:
            S(11, 0, 2)
    g = np.zeros(N_GLOBAL, np.float32)
    g[0] = 1.0 if st.turn == 0 else -1.0
    g[1] = st.kills[0] / 15.0
    g[2] = st.kills[1] / 15.0
    g[3] = len(E.alive_soldiers(st, 0)) / 15.0
    g[4] = len(E.alive_soldiers(st, 1)) / 15.0
    g[5] = min(st.moves, 200) / 200.0
    g[6] = 1.0 if E.queen(st, 0) else 0.0
    g[7] = 1.0 if E.queen(st, 1) else 0.0
    g[8] = (st.dice.count("X") if st.dice else 0) / 3.0
    g[9] = (1.0 if st.winner is None else
            (1.0 if st.winner == st.turn else -1.0))
    return pl, g


_PASS1 = [{}, {}]


def _first_pass(pl: int, sq: int):
    if not _PASS1[pl]:
        m = _PASS1[pl]
        for t in range(60):
            s = E.soldier_at(pl, t)
            if s not in m:
                m[s] = t
    return _PASS1[pl].get(sq)


def move_id(st: E.State, mv: dict) -> int:
    """Canonical action id for a legal move object (see header)."""
    p = E.piece(st, mv["pid"])
    if mv.get("kind") == "rel":
        cls = 0 if p.t == "s" else 1
        if p.t == "s":
            target = E.soldier_at(p.o, p.at + 1)
        else:
            ok, land, _ = E.path_for(p.at, mv["dir"], 1)
            target = land
        side = 0 if p.o == 0 else 1
        return REL_BASE + side * 90 + cls * 45 + target
    letter = "Q" if p.t == "q" else ("k" if p.t == "k" else ("M" if p.o == 0 else "K"))
    if p.t == "s":
        frm = E.soldier_at(p.o, p.at)
        to = E.soldier_at(p.o, p.at + mv["v"])
    elif mv.get("to") is not None:      # konge/dronning: tallsyklusen
        frm, to = p.at, mv["to"]
    else:                               # dronninga si X-løysing
        ok, land, _ = E.path_for(p.at, mv["dir"], mv["v"])
        frm, to = p.at, land
    return MOVE_BASE + CLASS_IDX[letter] * 2025 + frm * 45 + to


def special_id(kind: str) -> int:
    return SPEC_BASE + (0 if kind == "pass" else 1)


def decode_id(aid: int):
    """(kind, letter, from_sq, to_sq, side) — inverse of move_id (display)."""
    if aid >= SPEC_BASE:
        return ("pass" if aid == SPEC_BASE else "reroll", None, None, None, None)
    if aid >= MOVE_BASE:
        j = aid - MOVE_BASE
        letter = CLASSES[j // 2025]
        frm = (j % 2025) // 45
        to = j % 45
        return ("move", letter, frm, to, 0 if letter == "M" else (1 if letter == "K" else None))
    j = aid - REL_BASE
    side = j // 90
    cls = (j % 90) // 45
    target = j % 45
    return ("rel", "s" if cls == 0 else "q", None, target, side)


def legal_ids(st: E.State):
    """[(action id, move object)] for the current position (roll phase → specials).

    Deduplicated by action id: two soldiers of one side stacked on one square
    (FEN-anonymous) moving to one square share a canonical id and produce
    identical successor FENs — they are ONE move of the FEN state.
    """
    if st.winner is not None:
        return []
    out = []
    if st.phase == "roll":
        out.append((special_id("pass"), {"kind": "pass"}))
        return out
    seen = set()
    for mv in E.legal_moves(st, st.turn):
        aid = move_id(st, mv)
        if aid not in seen:
            seen.add(aid)
            out.append((aid, mv))
    if E.can_reroll_x(st):
        out.append((special_id("reroll"), {"kind": "reroll"}))
    if not seen:
        out.append((special_id("pass"), {"kind": "pass"}))
    return out


def encode_pair(st: E.State):
    pl, g = encode_state(st)
    return np.concatenate([pl.reshape(-1), g]).astype(np.float32)
