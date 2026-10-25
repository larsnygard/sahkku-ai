"""S-UCI FEN — literal port of SnartSahkku js/fen.js (spec docs/SAHKKU-UCI.md §6)."""
from __future__ import annotations
from .engine import (N, State, Piece, soldier_at, start_t, pt, idx_of,
                     name_of, sq_name, king, is_home, at_idx)

ENC = {"X": "X", "II": "2", "III": "3", "_": "_"}
DEC = {"X": "X", "2": "II", "3": "III", "_": "_"}

# square -> cycle position per side (row-2 squares sit on TWO passes:
# plain = first pass, prime = second), and home square -> column seq
_SQ2T, _SQ2T2, _HOME2SEQ = [], [], []


def _init_maps():
    for pl in (0, 1):
        m, m2, h = {}, {}, {}
        for t in range(60):
            sq = soldier_at(pl, t)
            if sq not in m:
                m[sq] = t
            elif sq not in m2:
                m2[sq] = t
        for c in range(15):
            h[soldier_at(pl, start_t(pl, c))] = c
        _SQ2T.append(m)
        _SQ2T2.append(m2)
        _HOME2SEQ.append(h)


_init_maps()


def sq_of_name(n: str) -> int:
    n = str(n).upper()
    for i in range(N):
        if name_of(i) == n:
            return i
    return -1


def token_for(p: Piece) -> str:
    if p.t == "s":
        if not p.released:
            return "m" if p.o == 0 else "g"
        sq = soldier_at(p.o, p.at)
        prime = _SQ2T2[p.o].get(sq) == p.at % 60
        base = "M" if p.o == 0 else "G"
        return base + ("'" if prime else "")
    if p.t == "q":
        return ("Q" if p.released else "q") if p.o == 0 else ("D" if p.released else "d")
    # king: N neutral · NM seized by menn · NK seized by giera
    return "NM" if p.o == 0 else ("NK" if p.o == 1 else "N")


def to_fen(st: State) -> str:
    rows = [[[] for _ in range(15)] for _ in range(3)]
    for p in st.pieces:
        if p.captured:
            continue
        c, r = pt(at_idx(st, p))
        rows[r][c].append(token_for(p))
    board = []
    for row in range(2, -1, -1):
        cells = []
        for c in range(15):
            # Kanonisk stabelrekkjefylgje: same sett brikker på eit felt må gje
            # SAME FEN-streng i Python og JS (elles driv paritetstesten på
            # G'G vs GG' i stablar med blanda pass). "'" sorterer etter bokstav.
            cells.append("+".join(sorted(rows[row][c], key=lambda t: t.replace("'", "~")))
                         if rows[row][c] else ".")
        board.append("-".join(cells))
    side = "w" if st.turn == 0 else "b"
    dice = "".join(ENC[d] for d in st.dice) if len(st.dice) == 3 else "-"
    used = "".join("1" if u else "0" for u in st.used) if len(st.used) == 3 else "000"
    return (board[0] + "/" + board[1] + "/" + board[2] + " " + side + " " +
            st.phase + " " + dice + " " + used + " " + name_of(king(st).at) +
            " " + str(st.kills[0]) + ":" + str(st.kills[1]) +
            " " + str(st.moves) + " " + ("*" if st.winner is None else ("1-0" if st.winner == 0 else "0-1")))


def _parse_token(tok: str, st: State, sq: int, assigned: set):
    if tok in ("N", "NM", "NK"):
        k = king(st)
        k.at = sq
        k.o = 0 if tok == "NM" else (1 if tok == "NK" else None)
        return
    if tok in ("M", "m", "G", "g", "M'", "G'"):
        prime = len(tok) > 1
        base = tok[0]
        pl = 0 if base in ("M", "m") else 1
        rel = base in ("M", "G") or prime
        if rel:
            t = (_SQ2T2[pl].get(sq) if prime else _SQ2T[pl].get(sq))
            if t is None:
                raise ValueError(f"fen: open soldat «{tok}» på {name_of(sq)} står utanfor syklus")
        else:
            c = _HOME2SEQ[pl].get(sq)
            if c is None:
                raise ValueError(f"fen: heimemann «{tok}» står ikje på heimrekka ({name_of(sq)})")
            t = start_t(pl, c)
        for p in st.pieces:
            if p.t == "s" and p.o == pl and p.id not in assigned:
                p.released = rel
                p.at = t
                p.captured = False
                assigned.add(p.id)
                return
        raise ValueError("fen: for mange soldatar")
    if tok in ("Q", "q", "D", "d"):
        pl = 0 if tok in ("Q", "q") else 1
        rel = tok in ("Q", "D")
        for p in st.pieces:
            if p.t == "q" and p.o == pl:
                p.captured = False
                p.released = rel
                if rel:
                    p.at = sq
                return
        raise ValueError("fen: ukjend dronning-token")
    raise ValueError(f"fen: ukjend brikke «{tok}» på {name_of(sq)}")


def from_fen(fen: str) -> State:
    w = str(fen).strip().split()
    if len(w) < 9:
        raise ValueError(f"fen: for få felt ({len(w)}/9)")
    st = State()
    rows = w[0].split("/")
    if len(rows) != 3:
        raise ValueError("fen: board må ha 3 rekker")
    assigned = set()
    row_names = [2, 1, 0]
    for r in range(3):
        cells = rows[r].split("-")
        if len(cells) != 15:
            raise ValueError(f"fen: rekkje {r} har {len(cells)} felt")
        for c in range(15):
            if cells[c] == ".":
                continue
            for tok in cells[c].split("+"):
                _parse_token(tok, st, idx_of(c, row_names[r]), assigned)
    for p in st.pieces:
        if p.t == "s" and p.id not in assigned:
            p.captured = True
        if p.t == "q":
            seen = any(ch in ("Q", "q") for ch in w[0]) if p.o == 0 else any(ch in ("D", "d") for ch in w[0])
            if not seen:
                p.captured = True
    if w[1] not in ("w", "b"):
        raise ValueError("fen: side")
    st.turn = 0 if w[1] == "w" else 1
    if w[2] not in ("roll", "move", "over"):
        raise ValueError("fen: fase")
    st.phase = w[2]
    st.dice = [] if w[3] == "-" else [DEC[ch] for ch in w[3]]
    if len(st.dice) == 3 and any(d is None for d in st.dice):
        raise ValueError("fen: terning")
    st.used = [ch == "1" for ch in w[4]] if st.dice else []
    ks = sq_of_name(w[5])
    if ks < 0:
        raise ValueError("fen: konge-felt")
    king(st).at = ks
    kl = w[6].split(":")
    st.kills = [int(kl[0]), int(kl[1])]
    st.moves = int(w[7]) if w[7].isdigit() else 0
    if w[8] == "1-0":
        st.winner = 0
        st.phase = "over"
    elif w[8] == "0-1":
        st.winner = 1
        st.phase = "over"
    elif w[8] != "*":
        raise ValueError("fen: resultat")
    return st
