"""Lágesvuotna sáhkku — pure-python rules engine.

Exhaustive literal port of SnartSahkku js/engine.js (+ js/fen.js). The JS
engine is the reference implementation (spec: SnartSahkku docs/SAHKKU-UCI.md
and docs/SPG.md); this module must behave identically, verified move-for-move
by tools/parity_check.js against the JS engine (see tests/test_parity.py).

Board: 3 rows x 15 cols, flat index: row0 (menn home) 0..14,
row1 (middle) 29..15 (mirrored), row2 (giera home) 30..44.
Soldiers ride a 60-point number cycle per side; king/queen move v steps in a
single straight direction (R/L along rows, F/B columns; corners keep going).
"""
from __future__ import annotations
import json
import random

N = 45
W = 15
DICE = ["X", "II", "III", "_"]
RANK = {"X": 0, "III": 1, "II": 2, "_": 3}
CASTLE = 22
DIRS = ["R", "L", "F", "B"]
DSEQ = ["R", "L", "F", "B"]  # canonical direction order


def pt(i: int):
    if i < 15:
        return (i, 0)
    if i < 30:
        return (29 - i, 1)
    return (i - 30, 2)


def idx_of(col: int, row: int) -> int:
    return col if row == 0 else (29 - col if row == 1 else 30 + col)


def name_of(i: int) -> str:
    c, r = pt(i)
    return chr(65 + c) + str(r + 1)


def sq_name(i: int, flip: bool = False) -> str:
    """Square name A1..O3 (row 2 mirrored), as in SPG/UCI notation."""
    if i < 15:
        c, r = i, 1
    elif i < 30:
        c, r = 29 - i, 2
    else:
        c, r = i - 30, 3
    if flip:
        c = 14 - c
    return chr(65 + c) + str(r)


def _mod(i: int, m: int) -> int:
    i = i % m
    return i + m if i < 0 else i


def _build_tour1():
    t = []
    for i in range(15):
        t.append(idx_of(i, 0))       # row0 A->O
    for i in range(15):
        t.append(idx_of(14 - i, 1))  # row1 O->A
    for i in range(15):
        t.append(idx_of(i, 2))       # row2 A->O
    for i in range(15):
        t.append(idx_of(14 - i, 1))  # row1 O->A
    return t                        # length 60


TOUR1 = _build_tour1()


def _tour2_of(t: int) -> int:
    c, r = pt(TOUR1[t])
    return idx_of(14 - c, 2 - r)


TOUR2 = [_tour2_of(t) for t in range(60)]


def soldier_at(pl: int, t: int) -> int:
    return TOUR1[_mod(t, 60)] if pl == 0 else TOUR2[_mod(t, 60)]


def start_t(pl: int, col: int) -> int:
    return col if pl == 0 else 14 - col


def step_dir(i: int, d: str):
    c, r = pt(i)
    if d == "R":
        if r == 0:
            return idx_of(c + 1, 0) if c < 14 else idx_of(14, 1)
        if r == 1:
            return idx_of(c + 1, 1) if c < 14 else idx_of(14, 2)
        return idx_of(c + 1, 2) if c < 14 else None
    if d == "L":
        if r == 1:
            return idx_of(c - 1, 1) if c > 0 else idx_of(0, 2)
        return idx_of(c - 1, r) if c > 0 else None
    if d == "F":
        if r == 0:
            return idx_of(c, 1)
        if r == 1:
            return idx_of(c, 2)
        return None
    if d == "B":
        if r == 2:
            return idx_of(c, 1)
        if r == 1:
            return idx_of(c, 0)
        return None
    return None


def path_for(p_at: int, d: str, v: int):
    """[legal, landing, path] — v steps in direction d from p.at."""
    cur = p_at
    path = []
    for _ in range(v):
        q = step_dir(cur, d)
        if q is None:
            return [False, None, []]
        path.append(q)
        cur = q
    return [True, path[-1], path]


def free_dirs_at(i: int, v: int):
    out = []
    for d in DIRS:
        cur = i
        ok = True
        for _ in range(v):
            q = step_dir(cur, d)
            if q is None:
                ok = False
                break
            cur = q
        if ok:
            out.append(d)
    return out


class Piece:
    __slots__ = ("id", "t", "o", "seq", "at", "t0", "released", "captured")

    def __init__(self, pid, t, o, seq, at, released=False, captured=False):
        self.id = pid
        self.t = t              # 's' soldier · 'q' queen · 'k' king
        self.o = o              # 0 menn · 1 giera · None neutral king
        self.seq = seq
        self.at = at            # cycle pos for soldiers, board idx for q/k
        self.t0 = at
        self.released = released
        self.captured = captured


class State:
    """Full game state; field-for-field mirror of the JS state object."""

    def __init__(self, first=0, queen_mode=True):
        pieces = []
        pid = 0
        for pl in (0, 1):
            for col in range(15):
                pieces.append(Piece(pid, "s", pl, col, start_t(pl, col)))
                pid += 1
            pieces.append(Piece(pid, "q", pl, -1, 18 if pl == 0 else 26))
            pid += 1
        pieces.append(Piece(pid, "k", None, -1, CASTLE, True))
        self.pieces = pieces
        self.turn = first
        self.phase = "roll"
        self.dice = []
        self.used = []
        self.dice_by = None
        self.winner = None
        self.win_reason = ""
        self.last_move = None
        self.kings = 0
        self.kills = [0, 0]
        self.moves = 0
        self.queen_mode = queen_mode
        self.pass_streak = 0

    def clone(self) -> "State":
        return deserialize(serialize(self))


def occ(st: State, i: int):
    out = []
    for p in st.pieces:
        if p.captured:
            continue
        a = soldier_at(p.o, p.at) if p.t == "s" else p.at
        if a == i:
            out.append(p)
    return out


def piece(st: State, pid: int):
    for p in st.pieces:
        if p.id == pid:
            return p
    return None


def soldiers(st: State, o: int):
    return [p for p in st.pieces if p.t == "s" and p.o == o]


def queen(st: State, o: int):
    for p in st.pieces:
        if p.t == "q" and p.o == o and not p.captured:
            return p
    return None


def king(st: State):
    for p in st.pieces:
        if p.t == "k":
            return p
    return None


def is_home(p: Piece) -> bool:
    return (p.t == "s" or p.t == "q") and not p.released and not p.captured


def at_idx(st: State, p: Piece) -> int:
    return soldier_at(p.o, p.at) if p.t == "s" else p.at


def alive_soldiers(st: State, o: int):
    return [s for s in soldiers(st, o) if not s.captured]


def can_land(st: State, p: Piece, land: int) -> bool:
    """Landing regulation (js canLand): passing is free, landing is not.

    - soldier: never own king/queen; takes enemy king (seize) / enemy queen /
      enemy soldier; may stack on own released soldiers and own home rank;
      never enemy home rank.
    - king: NEVER lands on a square with pieces (never captures; the king is
      captured only by an enemy soldier/queen landing on him).
    - queen: never shares with own pieces; takes enemy pieces; seizes king;
      never enemy home rank.
    """
    for q in occ(st, land):
        if q is p:
            continue
        if p.t == "s":
            if q.t == "k":
                if q.o == p.o:
                    return False
                continue
            if q.t == "q":
                if q.o != p.o:
                    continue
                return False
            if is_home(q):
                if q.o == p.o:
                    return True
                return False
            return True
        if p.t == "k":
            return False
        if p.t == "q":
            if q.t == "s" and q.o == p.o:
                return False
            if q.t == "q" and q.o == p.o:
                return False
            if q.t == "k":
                continue
            if is_home(q):
                return False
            return True
    return True


def soldier_landing_ok(st: State, p: Piece, land: int) -> bool:
    """Soldiers can never land on a line with UNRELEASED pieces (any side)."""
    for q in occ(st, land):
        if is_home(q):
            return False
    return True


def roll_dice(rnd=None):
    rnd = rnd or random.random
    return [DICE[int(rnd() * 4)] for _ in range(3)]


def roll(st: State, rnd=None):
    if st.phase != "roll":
        return None
    st.dice = roll_dice(rnd)
    st.used = [d == "_" for d in st.dice]
    st.dice_by = st.turn
    return st.dice[:]


def opening_roll(st: State, rnd=None):
    """Race for the opening: X-counts per roll, ties re-rolled."""
    for _ in range(10000):
        a = roll_dice(rnd)
        b = roll_dice(rnd)
        xa = a.count("X")
        xb = b.count("X")
        if xa != xb:
            first = 0 if xa > xb else 1
            st.turn = first
            return {"first": first, "p0": a, "p1": b}
    raise RuntimeError("opening roll did not terminate")


def active_rank(st: State):
    """Lowest rank number of the unspent slots (= the die to play).

    Defensive exactly like the JS referee: a state with fewer than three
    dice (e.g. a fresh `phase == "roll"` position, before the GUI rolls)
    simply has no active die -> None. JS reads `undefined` in the same case
    and every comparison is false, so this keeps Python and JS identical
    on malformed/not-yet-rolled input too.
    """
    best = 4
    for i in range(min(3, len(st.dice))):
        used = st.used[i] if i < len(st.used) else False
        if used or st.dice[i] == "_":
            continue        # blank = dødt fjes, aldri aktiv (jf. dice_left)
        if RANK[st.dice[i]] < best:
            best = RANK[st.dice[i]]
    return None if best == 4 else best


def active_value(st: State) -> int:
    rk = active_rank(st)
    if rk is None:
        return 0
    return 1 if rk == RANK["X"] else (3 if rk == RANK["III"] else 2)


def dice_left(st: State) -> int:
    return sum(1 for i in range(3) if not st.used[i] and st.dice[i] != "_")


def can_reroll_x(st: State) -> bool:
    """sa hkku.no rule: reroll Xs only when X (and the rest of the roll)
    cannot be used at all."""
    if st.winner is not None or st.phase != "move":
        return False
    if active_rank(st) != RANK["X"]:
        return False
    return len(legal_moves(st, st.turn)) == 0


def reroll_x(st: State, rnd=None):
    if not can_reroll_x(st):
        return None
    for i in range(3):
        if st.dice[i] == "X":
            nf = roll_dice(rnd)[0]
            st.dice[i] = nf
            st.used[i] = (nf == "_")    # nytt blankt fjes er brukt med ein gong
    return st.dice[:]


def gen_kq_moves(st: State, p: Piece, v: int):
    res = []
    for d in free_dirs_at(p.at, v):
        ok, land, _ = path_for(p.at, d, v)
        if not ok:
            continue
        if not can_land(st, p, land):
            continue
        res.append({"pid": p.id, "dir": d, "v": v, "kind": "mv"})
    return res


def legal_moves(st: State, me: int):
    """Legal moves for the ACTIVE dice face: [{pid, dir, v, kind}].

    X face: release frontmost unreleased soldier (highest cycle pos) and/or
    queen release in 4 straight directions. II/III: soldiers move v steps
    FORWARD along their number cycle; queen (if released, queenMode) and
    owned king move v steps in one straight direction.
    """
    if st.phase != "move" or st.winner is not None:
        return []
    out = []
    rk = active_rank(st)
    if rk is None:
        return out
    v = active_value(st)
    if rk == RANK["X"]:
        home = [s for s in soldiers(st, me) if not s.released and not s.captured]
        if home:
            front = None
            for s in home:
                if front is None or s.at > front.at:
                    front = s
            land = soldier_at(me, front.at + 1)
            if soldier_landing_ok(st, front, land) and can_land(st, front, land):
                out.append({"pid": front.id, "dir": 1, "v": 1, "kind": "rel"})
        q = queen(st, me)
        if q and not q.released and st.queen_mode:
            for d in DIRS:
                ok, land, _ = path_for(q.at, d, 1)
                if ok and soldier_landing_ok(st, q, land) and can_land(st, q, land):
                    out.append({"pid": q.id, "dir": d, "v": 1, "kind": "rel"})
    # X (sáhkku) VEKKJER berre: ho løyser ei inaktiv brikke (soldat eller
    # dronning) og flyttar henne eitt felt som del av vekkinga. Ho flyttar
    # ALDRI brikker som alt er utløyste — det gjer II og III.
    if rk == RANK["X"]:
        return out
    for p in st.pieces:
        if p.captured:
            continue
        if p.t == "s":
            if p.o != me or not p.released:
                continue
            dest = soldier_at(me, p.at + v)
            if not soldier_landing_ok(st, p, dest) or not can_land(st, p, dest):
                continue
            out.append({"pid": p.id, "dir": 1, "v": v, "kind": "mv"})
        elif p.t == "q":
            if not st.queen_mode or p.o != me or not p.released:
                continue
            out.extend(gen_kq_moves(st, p, v))
        elif p.t == "k":
            if p.o != me:
                continue
            out.extend(gen_kq_moves(st, p, v))
    return out


def use_die(st: State, v: int) -> int:
    face = "X" if v == 1 else ("II" if v == 2 else "III")
    for i in range(3):
        if not st.used[i] and st.dice[i] == face:
            st.used[i] = True
            return i
    return -1


def apply_move(st: State, mv: dict):
    """Execute a legal move; returns event list (js-compatible shape)."""
    ev = []
    p = piece(st, mv["pid"])
    me = st.turn
    use_die(st, mv["v"])
    if p.t == "s":
        from_idx = soldier_at(p.o, p.at)
        land = soldier_at(p.o, p.at + (1 if mv["kind"] == "rel" else mv["v"]))
        p.at = p.at + (1 if mv["kind"] == "rel" else mv["v"])
    else:
        from_idx = p.at
        ok, land, _ = path_for(p.at, mv["dir"], mv["v"])
        p.at = land
    if mv["kind"] == "rel":
        p.released = True
    ev.append({"type": "mv", "pid": p.id, "from": from_idx, "to": land,
               "by": me, "v": mv["v"], "dir": mv["dir"],
               "rel": mv["kind"] == "rel"})
    lg = {"pid": p.id, "t": p.t, "o": p.o, "from": from_idx, "to": land,
          "v": mv["v"], "dir": mv["dir"], "rel": mv["kind"] == "rel",
          "k": 0, "kg": False}
    st.last_move = {"pid": p.id, "from": from_idx, "to": land, "by": me}
    st.moves += 1
    st.pass_streak = 0
    # landing effects
    for q in occ(st, land):
        if q is p:
            continue
        if q.t == "k":
            q.o = me
            st.kings += 1
            lg["kg"] = True
            ev.append({"type": "king", "by": me, "at": land})
            continue
        if q.o == me:
            continue  # own pieces: stack
        if q.t == "s":
            q.captured = True
            q.released = True
            st.kills[me] += 1
            lg["k"] += 1
            lg["cap"] = (lg.get("cap") or []) + [sq_name(land) + ("M" if q.o == 0 else "K")]
            ev.append({"type": "kill", "pid": q.id, "at": land, "by": me})
        elif q.t == "q":
            q.captured = True
            st.kills[me] += 1
            lg["k"] += 1
            lg["cap"] = (lg.get("cap") or []) + ["Q"]
            ev.append({"type": "kill", "pid": q.id, "at": land, "by": me})
            st.winner = me
            st.win_reason = "dronning"
    # win: annihilation (Lágesvuotna: only queen-take and annihilation)
    foe = 1 - me
    if st.winner is None:
        if not alive_soldiers(st, foe):
            st.winner = me
            st.win_reason = "utryddelse"
    if st.winner is not None:
        st.phase = "over"
        return ev
    if dice_left(st) == 0:
        end_turn(st, ev)
    return ev


def end_turn(st: State, ev: list):
    st.dice = []
    st.used = []
    st.dice_by = None
    st.turn = 1 - st.turn
    st.phase = "roll"
    st.pass_streak += 1
    ev.append({"type": "turn", "next": st.turn})


def pass_remaining(st: State):
    ev = []
    end_turn(st, ev)
    return ev


def settle(st: State):
    """Post-move drive: {over} | {turnEnded} | {reroll} | {moves}."""
    if st.winner is not None:
        return {"over": True}
    if st.phase != "move":
        return {"turnEnded": True, "events": []}
    rk = active_rank(st)
    if rk is None:
        return {"turnEnded": True, "events": pass_remaining(st)}
    lm = legal_moves(st, st.turn)
    if not lm:
        if rk == RANK["X"]:
            return {"reroll": True, "moves": []}
        return {"turnEnded": True, "events": pass_remaining(st)}
    return {"moves": lm}


def serialize(st: State) -> str:
    return json.dumps({
        "turn": st.turn, "phase": st.phase, "dice": st.dice, "used": st.used,
        "diceBy": st.dice_by, "winner": st.winner, "winReason": st.win_reason,
        "queenMode": st.queen_mode, "kings": st.kings, "kills": st.kills,
        "moves": st.moves,
        "pieces": [{"id": p.id, "t": p.t, "o": p.o, "seq": p.seq, "at": p.at,
                    "released": bool(p.released), "captured": bool(p.captured)}
                   for p in st.pieces],
    })


def deserialize(s) -> State:
    o = json.loads(s) if isinstance(s, str) else s
    st = State.__new__(State)
    st.pieces = [Piece(p["id"], p["t"], p["o"], p["seq"], p["at"],
                       p["released"], p["captured"]) for p in o["pieces"]]
    for p in st.pieces:
        p.t0 = p.at
    st.turn = o["turn"]
    st.phase = o["phase"]
    st.dice = o["dice"]
    st.used = o["used"]
    st.dice_by = o["diceBy"]
    st.winner = o["winner"]
    st.win_reason = o["winReason"]
    st.queen_mode = o["queenMode"]
    st.last_move = None
    st.kings = o["kings"]
    st.kills = o["kills"]
    st.moves = o["moves"]
    st.pass_streak = 0
    return st
