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
N_GLOBAL = 16   # global scalar vector (10 opphavlege + 5 doktrine/terningar + regelmodus)


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
    # Globale tal er POV-relative (mi side fyrst). MERK: g[3]/g[4] tel
    # UTLØYSTE soldatar, ikkje «i live» — ei brikke som står heime er trygg og
    # kan ikkje takast; det er dei utløyste som gjev mobilitet og risiko.
    # (Kor mange som er slagtekne ligg alt i g[1]/g[2], so talet på heimebrikker
    # er 15 − utløyste − slagtekne og er implisitt kjent.)
    me = st.turn
    foe = 1 - me
    g = np.zeros(N_GLOBAL, np.float32)
    g[0] = 1.0 if st.turn == 0 else -1.0
    g[1] = st.kills[me] / 15.0
    g[2] = st.kills[foe] / 15.0
    g[3] = len([p for p in E.soldiers(st, me) if p.released and not p.captured]) / 15.0
    g[4] = len([p for p in E.soldiers(st, foe) if p.released and not p.captured]) / 15.0
    g[5] = min(st.moves, 200) / 200.0
    # g[6]/g[7] = dronninga er UTLØYST (ikkje berre «finst»: ei dronning som
    # er slått ut avsluttar partiet, so før var desse to konstant 1.0 og
    # nettet kunne ikkje lære noko om tidleg vs sein aktivering).
    qm, qf = E.queen(st, me), E.queen(st, foe)
    g[6] = 1.0 if (qm is not None and qm.released) else 0.0
    g[7] = 1.0 if (qf is not None and qf.released) else 0.0
    g[8] = (st.dice.count("X") if st.dice else 0) / 3.0
    g[9] = (1.0 if st.winner is None else
            (1.0 if st.winner == st.turn else -1.0))

    # ---- kongedoktrinen og terningane (Lars 2026-10-27) ------------------
    # g[10..12] svarar på «kongen er bufferen»: han gjev alltid eit trekk, so
    # utan konge kan ein verte sett i tvang. Målt i AIL0.6-parti: 8 tvungne
    # pass med konge mot 46 utan. Ingenting av dette fanst i trekka før.
    k = E.king(st)
    mine = (k.o == me)
    theirs = (k.o == foe)
    g[10] = 1.0 if mine else 0.0
    g[11] = 1.0 if theirs else 0.0
    if mine:
        # avstand til næraste fiende, snudd: 1.0 = langt borte (trygg),
        # 0.0 = rett attmed. Direkte omsetjing av «hald kongen attende».
        foes = [E.soldier_at(foe, s.at) for s in E.soldiers(st, foe) if not s.captured]
        if foes:
            d = min(_cheb(k.at, f) for f in foes)
            g[12] = 1.0 - min(d, 14) / 14.0
    # g[13] = dronninga mi er UTLØYST og kan takast av fienden NO.
    # «Aktivering er risikobusiness»: dette er talet som gjer risikoen synleg.
    if qm is not None and qm.released and not qm.captured:
        # berre dronningfeltet er interessant — ikkje alle slagfelt
        g[13] = 1.0 if _square_attackable(st, qm.at, foe) else 0.0

    # g[14] = kor sannsynleg er det at MOTSTANDAREN får eit trekk dei kan
    # nytte i neste kast? Fyrst kva terningar dei kan nytte i det heile
    # (noko dei ikkje kan bruke, kan dei heller ikkje spelt), so ein
    # 8-rads tabell over terningkasta (sjå DICE_USE_PROB).
    mask = _usable_mask(st, foe)
    g[14] = DICE_USE_PROB[mask]

    # g[15] = regelmodus: 1.0 når Nesseby er på. Utan dette trekket kan eitt
    # nett ikkje vite kva reglar det spelar etter, og då måtte vi hatt eitt
    # nett per regelvariant i staden for eitt som kan veljast i menyen.
    g[15] = 1.0 if getattr(st, "nesseby", False) else 0.0
    return pl, g


def _cheb(a, b):
    ca, ra = E.pt(a)
    cb, rb = E.pt(b)
    return max(abs(ca - cb), abs(ra - rb))


# Sannsynet for at eit kast (tre terningar, fire fjes) inneheld minst éin
# brukbar terning, gjeve kva fjes sida kan nytte i det heile. Bit 0 = X(1),
# bit 1 = II(2), bit 2 = III(3). Rekna ut over alle 64 ordna kasta; tabellen
# er liten nok til å leggjast inn som konstant.
#   0 fjes brukbare -> 0.0 (kastet er daudt uansett)
#   1 fjes           -> 1 - (3/4)^3 = 0.5781
#   2 fjes           -> 1 - (2/4)^3 = 0.8750
#   3 fjes           -> 1 - (1/4)^3 = 0.9844
DICE_USE_PROB = (0.0,      # 0 fjes
                 0.5781, 0.5781, 0.8750,     # 1 fjes (X / II / X+II)
                 0.5781,                    # III åleine
                 0.8750, 0.8750,             # X+III / II+III
                 0.9844)                    # alle tre


def _square_attackable(st, sq, side):
    """Kan `side` slå ein brikke som står på `sq`, med terning 1-3?

    Banelogikk i staden for tre fulle trekk-mengder: konge/dronning går rette
    liner 1-3 steg, soldatar berre framover i sin eigen syklus.
    """
    for p in st.pieces:
        if p.captured or p.o != side:
            continue
        if p.t == "s":
            if not p.released:
                continue
            for v in (1, 2, 3):
                if E.soldier_at(side, p.at + v) == sq and E.can_land(st, p, sq):
                    return True
            continue
        for v in (1, 2, 3):
            for d in ("R", "L", "F", "B"):
                ok, land, _ = E.path_for(p.at, d, v)
                if ok and land == sq and E.can_land(st, p, sq):
                    return True
    return False


def _usable_mask(st, side):
    """Kva terningverdiar kan `side` nytte i denne stillinga? (bitmask)

    Rekna med banelogikk (kvar kan kvar brikke lande) i staden for å byggje
    tre fulle trekk-mengder. Det same svaret — test_usable_mask samanliknar
    med den dyre varianten — men ~0,02 ms i staden for ~0,26 ms, og denne
    funksjonen vert kalla éin gong per MCTS-node.
    """
    mask = 0
    home = [s for s in E.soldiers(st, side) if not s.captured and not s.released]
    for p in st.pieces:
        if p.captured or p.o != side:
            continue
        if p.t == "s":
            if not p.released:
                continue                      # heimesoldatar vert handsama under
            for v in (1, 2, 3):
                if mask & (1 << (v - 1)):
                    continue
                land = E.soldier_at(side, p.at + v)
                if E.soldier_landing_ok(st, p, land) and E.can_land(st, p, land):
                    mask |= 1 << (v - 1)
            continue
        # dronninga må vere UTLØYST (og spelet må ha dronning med) — ei
        # heimeståande dronning kan berre vekkjast med X, ikkje flyttast.
        # Utan denne sjekken overvurderte vi kva ho kan nytte.
        if p.t == "q" and (not st.queen_mode or not p.released):
            continue
        # konge/dronning går BÅDE langs tallsyklusen (åttetalet) og rett
        # opp/ned/høgre/venstre — begge må med.
        t0 = E.cycle_index(p.o, p.at)
        for v in (1, 2, 3):
            if mask & (1 << (v - 1)):
                continue
            found = False
            if t0 >= 0:
                for sgn in (-1, 1):
                    land = E.soldier_at(p.o, t0 + sgn * v)
                    if land != p.at and E.can_land(st, p, land):
                        found = True
                        break
            if not found:
                for d in ("R", "L", "F", "B"):
                    ok, land, _ = E.path_for(p.at, d, v)
                    if ok and E.can_land(st, p, land):
                        found = True
                        break
            if found:
                mask |= 1 << (v - 1)
    # vekking med X: berre den FREMSTE heimesoldaten kan vekkjast, og ei
    # heimeståande dronning kan løysast eitt steg i fire retningar.
    if not (mask & 1):
        if home:
            front = max(home, key=lambda s: s.at)
            land = E.soldier_at(side, front.at + 1)
            if E.soldier_landing_ok(st, front, land) and E.can_land(st, front, land):
                mask |= 1
        if not (mask & 1):
            q = E.queen(st, side)
            if q is not None and st.queen_mode and not q.released and not q.captured:
                for d in ("R", "L", "F", "B"):
                    ok, land, _ = E.path_for(q.at, d, 1)
                    if ok and E.soldier_landing_ok(st, q, land) and E.can_land(st, q, land):
                        mask |= 1
                        break
    return mask


def _capture_squares(st, side, max_v=3):
    """Kva felt kan `side` slå på no, med terningverdi 1..max_v? {felt: [v..]}"""
    out = {}
    save = (list(st.dice), list(st.used), st.phase, st.turn)
    try:
        st.turn = side
        st.phase = "move"
        for v, face in ((1, "X"), (2, "II"), (3, "III")):
            if v > max_v:
                continue
            st.dice = [face, "_", "_"]
            st.used = [False, True, True]
            for mv in E.legal_moves(st, side):
                land = E.move_dest(st, mv)
                if any(q.o is not None and q.o != side and q.id != mv["pid"]
                       for q in E.occ(st, land)):
                    out.setdefault(land, []).append(v)
    finally:
        st.dice, st.used, st.phase, st.turn = save
    return out


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
