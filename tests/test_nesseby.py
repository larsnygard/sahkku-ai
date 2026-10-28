"""Nesseby-regelen: kongen kan ikkje kaprast ved landing, berre ved heimrad.

Regelen (valfri): kongen er immun til ein soldat når MOTSTANDARENS heimrad;
den fyrste soldaten som når dit kaprar kongen for si side.

Testen pinnar fire ting i Python-porten, og at JS-dommaren gjer det SAME
(reglane må vere identiske, elles trenar vi på noko anna enn appen spelar):

  1. konge utan eigar: soldat kan IKKJE lande på han når Nesseby er på
  2. same stilling utan Nesseby: landing kaprar (som før)
  3. soldat på motstandarens heimrad kaprar kongen for si side
  4. dronninga kan heller ikkje lande på ein uaktivert konge

Køyr:  .venv/bin/python -m tests.test_nesseby
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sahkku_ai import engine as E

OK = []
ROOT = os.path.join(os.path.dirname(__file__), "..")
VENDOR = os.path.join(ROOT, "vendor", "engine.js")


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    OK.append(msg)


def js(expr):
    """Køyr eit JS-uttrykk mot vendor/engine.js og les JSON-svaret."""
    code = ("global.SS=require(%r);" % VENDOR) + expr
    out = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise SystemExit("JS feila: " + out.stderr[:300])
    return json.loads(out.stdout.strip() or "null")


# --- 1+2. konge utan eigar, soldat eitt steg unna -----------------------
def rig(nesseby, queen_instead=False):
    st = E.State(first=0, nesseby=nesseby)
    st.phase = "move"
    st.turn = 0
    for s in E.soldiers(st, 0) + E.soldiers(st, 1):
        s.captured = True
    k = E.king(st)
    k.o, k.at = None, 22                       # H2, utan eigar
    if queen_instead:
        q = E.queen(st, 0)
        q.captured, q.released = False, True
        # dronninga eitt steg frå kongen langs syklusen
        q.at = E.soldier_at(0, E.cycle_index(0, 22) - 1)
        st.dice = ["X", "_", "_"]
        st.used = [False, True, True]
        return st, q
    s = E.soldiers(st, 0)[0]
    s.captured, s.released = False, True
    s.at = E.cycle_index(0, 22) - 1             # eitt steg før kongefeltet
    st.dice = ["X", "_", "_"]
    st.used = [False, True, True]
    return st, s


def reaches_king(st, piece):
    for mv in E.legal_moves(st, 0):
        if mv["pid"] == piece.id and E.move_dest(st, mv) == 22:
            return True
    return False


st_on, s = rig(True)
check(not reaches_king(st_on, s), "Nesseby på: soldat kan ikkje lande på uaktivert konge")

st_off, s2 = rig(False)
check(reaches_king(st_off, s2), "Nesseby av: soldat kan lande på og kapre kongen")

# --- 3. soldat på motstandarens heimrad kaprar kongen -------------------
st3 = E.State(first=0, nesseby=True)
st3.phase = "move"
st3.turn = 0
for q in E.soldiers(st3, 0) + E.soldiers(st3, 1):
    q.captured = True
k3 = E.king(st3)
k3.o, k3.at = None, 22
hero = E.soldiers(st3, 0)[0]
hero.captured, hero.released = False, True
# finn ein syklusposisjon der neste steg landar på rad 3 (konenes heimrad)
target = None
for t in range(60):
    if E.pt(E.soldier_at(0, t + 1))[1] == 2:      # rad 3
        target = t
        break
check(target is not None, "fann eit steg inn på konenes heimrad")
hero.at = target
st3.dice = ["X", "_", "_"]
st3.used = [False, True, True]
mv = next(m for m in E.legal_moves(st3, 0) if m["pid"] == hero.id)
ev = E.apply_move(st3, mv)
check(E.king(st3).o == 0, "soldat på heimrada kaprar kongen for si side")
check(any(e.get("nesseby") for e in ev), "hendinga er merkt som Nesseby-kapring")

# --- 4. dronninga kan ikkje lande på uaktivert konge --------------------
st4, q4 = rig(True, queen_instead=True)
check(not reaches_king(st4, q4), "Nesseby på: dronninga kan ikkje lande på uaktivert konge")

# --- 5. JS-dommaren gjer det same ---------------------------------------
res = js("""
  function rig(nb) {
    const st = SS.newGame({ first: 0, nesseby: nb });
    st.phase='move'; st.turn=0;
    SS.soldiers(st,0).forEach(s=>{s.captured=true});
    SS.soldiers(st,1).forEach(s=>{s.captured=true});
    const k=SS.king(st); k.o=null; k.at=22;
    let cyc=-1; for (let t=0;t<60;t++) if (SS.soldierAt(0,t)===22) { cyc=t; }
    // syklusposisjonen EITT steg før 22:
    let before=-1; for (let t=0;t<60;t++) if (SS.soldierAt(0,t+1)===22) { before=t; }
    const s=SS.soldiers(st,0)[0]; s.captured=false; s.released=true; s.at=before;
    st.dice=['X','_','_']; st.used=[false,true,true];
    return { before: before, reaches: SS.legalMoves(st,0).some(m=>m.pid===s.id && SS.moveDest(st,m)===22) };
  }
  console.log(JSON.stringify({ on: rig(true), off: rig(false) }));
""")
check(res["on"]["reaches"] is False, f"JS: Nesseby på nektar landing på uaktivert konge ({res['on']})")
check(res["off"]["reaches"] is True, f"JS: Nesseby av tillet kapsel ({res['off']})")

print(f"NESSEBY-TEST OK: {len(OK)} sjekkar (Python og JS same svar)")
for m in OK:
    print("  ok", m)
