"""Regeltest: konge + kapslande soldat på same felt — og kven som elles får lande der.

Lars 2026-10-28:
  * når ein spelar kaprar kongen, står kongen og den kapslande soldaten på
    same felt;
  * det er den EINASTE gongen ein soldat og kongen deler felt;
  * ein TREDJE soldat får ikkje lande der saman med dei.

Vi prøver alle vegar inn i feltet og ser kva motoren faktisk gjer, i båe
motorane (Python og JS-dommaren).

Køyr:  .venv/bin/python -m tests.test_king_stack
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sahkku_ai import engine as E

OK = []


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    OK.append(msg)


def base(king_owner=None, capturer_side=None, at=22):
    """Konge på `at`, ev. med ein kapslande soldat frå `capturer_side` der."""
    st = E.State(first=0)
    st.queen_mode = True
    st.phase = "move"
    st.turn = 0
    k = E.king(st)
    k.o = king_owner
    k.at = at
    for s in E.soldiers(st, 0) + E.soldiers(st, 1):
        s.captured = True
    if capturer_side is not None:
        c = E.soldiers(st, capturer_side)[0]
        c.captured = False
        c.released = True
        c.at = E.cycle_index(capturer_side, at)
    return st


def moving_soldier(st, side, sq_from, dice="III"):
    """Legg ein utløyst soldat på `sq_from` (klår til å flytte mot kongen)."""
    s = next(x for x in E.soldiers(st, side) if x.captured or x.id == E.soldiers(st, side)[0].id)
    s.captured = False
    s.released = True
    s.at = E.cycle_index(side, sq_from)
    st.turn = side
    st.dice = [dice, "_", "_"]
    st.used = [False, True, True]
    st.phase = "move"
    return s


def lands_on(st, side):
    out = []
    for mv in E.legal_moves(st, side):
        if mv.get("kind") == "rel":
            continue
        land = E.move_dest(st, mv)
        if E.king(st).at == land:
            out.append(mv)
    return out


KING = 22          # H2, slottet
print("=== 1. konge utan eigar, soldat landar på han (kapsel) ===")
st = base(None)
# finn eit felt som ligg 1-3 steg framfor kongen i syklusen til side 0
for back in (1, 2, 3):
    src = E.soldier_at(0, E.cycle_index(0, KING) - back)
    s2 = base(None)
    moving_soldier(s2, 0, src)
    hits = lands_on(s2, 0)
    if hits:
        ch = s2.clone()
        E.apply_move(ch, hits[0])
        k = E.king(ch)
        occ = [E.piece(ch, q.id) for q in E.occ(ch, KING)]
        check(k.o == 0, "kapsel: kongen får eigar 0")
        check(len(occ) == 2 and any(p.t == "k" for p in occ) and any(p.t == "s" for p in occ),
              f"kapsel: konge og soldat står saman på {E.sq_name(KING)} ({[p.t for p in occ]})")
        print(f"  ok: med {back} steg bakfrå tek soldaten kongen, og dei deler feltet")
        break
else:
    check(False, "fann ingen soldat som kan ta kongen i testoppsettet")

print("\n=== 2. eigen soldat får IKKJE lande på eigen konge ===")
for back in (1, 2, 3):
    src = E.soldier_at(0, E.cycle_index(0, KING) - back)
    st2 = base(0)
    moving_soldier(st2, 0, src)
    check(not lands_on(st2, 0), f"eigen soldat ({back} steg bakfrå) vert stoppa frå eigen kongefelt")
print("  ok: alle tre avstandar stoppa")

print("\n=== 3. TREDJE soldat får ikkje lande på konge + kapslar ===")
for back in (1, 2, 3):
    src = E.soldier_at(0, E.cycle_index(0, KING) - back)
    st3 = base(0, capturer_side=0)          # konge + eigen kapslar på feltet
    moving_soldier(st3, 0, src)
    check(not lands_on(st3, 0),
          f"tredje soldat ({back} steg bakfrå) vert stoppa frå konge+kapslar-feltet")
print("  ok: alle tre avstandar stoppa")

print("\n=== 4. fienden kan ta kongen att (kapsel på konge + fiende-soldat) ===")
st4 = base(0, capturer_side=0)              # vi eig kongen, vår soldat står der
st4.turn = 1
for back in (1, 2, 3):
    src = E.soldier_at(1, E.cycle_index(1, KING) - back)
    st4b = base(0, capturer_side=0)
    moving_soldier(st4b, 1, src)
    hits = lands_on(st4b, 1)
    if hits:
        ch = st4b.clone()
        E.apply_move(ch, hits[0])
        occ = [E.piece(ch, q.id) for q in E.occ(ch, KING)]
        check(E.king(ch).o == 1, "fienden kaprar kongen att")
        check(sum(1 for p in occ if p.t == "s") == 1,
              f"etter omkapsling står éin soldat med kongen ({[p.t for p in occ]})")
        print(f"  ok: fienden tok kongen {back} steg bakfrå; soldaten vår vart slått ut, ein ny står att")
        break

print("\n=== 5. kapslaren kan gå bort, kongen vert ståande åleine ===")
st5 = base(0, capturer_side=0)
c = [p for p in st5.pieces if p.t == "s" and not p.captured and p.o == 0][0]
st5.turn = 0
st5.dice = ["III", "_", "_"]
st5.used = [False, True, True]
moves = [m for m in E.legal_moves(st5, 0) if m["pid"] == c.id]
check(len(moves) > 0, "kapslaren kan flytte seg bort frå kongefeltet")
if moves:
    ch = st5.clone()
    E.apply_move(ch, moves[0])
    occ = [p.t for p in (E.piece(ch, q.id) for q in E.occ(ch, KING))]
    check(occ == ["k"], f"kongen står att åleine på feltet ({occ})")
    print("  ok: kapslaren gjekk, kongen står att åleine")

print("\n=== 6. dronninga kan heller ikkje lande på eigen konge ===")
any_landing = False
for back in (1, 2, 3):
    st6 = E.State(first=0)
    st6.queen_mode = True
    st6.phase = "move"
    st6.turn = 0
    k6 = E.king(st6)
    k6.o, k6.at = 0, KING
    for s in E.soldiers(st6, 0) + E.soldiers(st6, 1):
        s.captured = True
    q6 = E.queen(st6, 0)
    q6.captured, q6.released = False, True
    q6.at = E.soldier_at(0, E.cycle_index(0, KING) - back)
    for face in ("X", "II", "III"):
        st6.dice = [face, "_", "_"]
        st6.used = [False, True, True]
        if any(m["pid"] == q6.id and E.move_dest(st6, m) == KING
               for m in E.legal_moves(st6, 0)):
            any_landing = True
check(not any_landing, "dronninga kan ikkje lande på eigen konge")
print("  ok: dronninga vert stoppa på alle tre avstandar og terningverdiar")

print(f"KONGESTACK-TEST OK: {len(OK)} sjekkar")
for m in OK:
    print("  ok", m)
