"""Bonderisiko-trekka (g[16]/g[17]): rett meining, og likt i Python og JS.

Trekka finst fordi menneske slår NAIS på nøyaktig dette: nettet held få
brikker ute og set dei framme, og då vert den eine aktive brikka plukka.
Nettet kunne sjå KOR MANGE som var ute (g[3]/g[4]) men ikkje kor UTSETTE dei
stod. Denne testen pinnar:

  1. meininga: ei utløyst brikke på eit felt fienden kan slå på tel, og ho
     sluttar å telje når ho står trygt;
  2. at Python og JS reknar same vektor (alle 18 globale tal), ikkje berre
     dei nye — dei to implementasjonane må gje identisk inn-data til nettet;
  3. at trekka faktisk er i bruk (ikkje alltid 0) i stillingar frå ekte parti.

Køyr: .venv/bin/python -m tests.test_exposure
"""
from __future__ import annotations
import json
import os
import copy
import random
import subprocess
import sys
import tempfile

from sahkku_ai import engine as E
from sahkku_ai import features as FT

PASS = FAIL = 0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, "engines", "tmp-g18.js")      # utvida nais5, 18 globale


def ok(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" :: {extra}" if extra else ""))


def positions(n=12, seed=7):
    """Stillingar frå tilfeldige parti (same veg som fuzz_parity brukar).

    Nesseby er AV her: FEN ber ikkje regelflaget, so JS-sida (som byggjer
    stillinga frå FEN) ville alltid fått g[15]=0 medan Python fekk 1. Det er ei
    kjend avgrensing i FEN, ikkje ein feil i trekka.
    """
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        st = E.State(first=rnd.randint(0, 1), nesseby=False)
        E.opening_roll(st, rnd.random)
        for _ in range(80):
            if st.winner is not None:
                break
            if st.phase == "roll":
                E.roll(st, rnd.random)
                st.phase = "move"
            lm = E.legal_moves(st, st.turn)
            if not lm:
                if E.can_reroll_x(st):
                    E.reroll_x(st, rnd.random)
                    continue
                E.pass_remaining(st)
                continue
            E.apply_move(st, lm[rnd.randrange(len(lm))])
            E.settle(st)
            if 12 <= st.moves <= 60 and st.winner is None:
                snap = copy.deepcopy(st)      # stilling UNDERVEGS, ikkje berre slutt
                out.append(snap)
                break
    return out


def main():
    # ---- 1. meininga: konstruert stilling --------------------------------
    st = E.State(first=0, nesseby=True)
    rnd = random.Random(3)
    E.opening_roll(st, rnd.random)
    st.turn = 0
    E.roll(st, rnd.random)
    st.phase = "move"
    # løys éin soldat og set han rett framfor ein uutløyst fiendebrikke
    # (det er stillinga mennesket utnyttar), og mål risikoen
    s0 = E.soldiers(st, 0)[0]
    s0.released = True
    foe = E.soldiers(st, 1)[0]
    foe.released = True
    # fienden går FRAMOVER i sin eigen syklus: legg brikka mi to steg framfor
    s0.at = E.cycle_index(0, E.soldier_at(1, foe.at + 2))
    g0 = FT.encode_state(st)[1]
    ok("utsett brikke tel i g[16]", g0[16] > 0.0, f"g16={g0[16]:.3f}")
    # flytt henne til eit felt fienden ikkje kan nå, og mål att
    # Definisjonen: g[16] skal vere NETTOPP delen utsette soldatar — rekna
    # om att i testen, so talet ikkje kan stemme ved eit uhell.
    caps = FT.capture_squares(st, 1)
    manual = sum(1 for p in E.soldiers(st, 0)
                 if p.released and not p.captured and E.soldier_at(0, p.at) in caps)
    ok("g[16] er lik delen utsette soldatar", abs(float(g0[16]) - manual / 15.0) < 1e-6,
       f"g16={g0[16]:.4f} mot {manual}/15")
    # Og i ei opning utan utløyste fiendar er risikoen 0.
    st2 = E.State(first=0, nesseby=False)
    E.opening_roll(st2, random.Random(1).random)
    ok("opning utan utløyste fiendar gjev 0", float(FT.encode_state(st2)[1][16]) == 0.0,
       float(FT.encode_state(st2)[1][16]))
    ok("fienden sitt tal er spegla (g[17] finst)", g0.shape[0] == 18, g0.shape)

    # ---- 2. Python vs JS: alle 18 globale tal ---------------------------
    pos = positions(n=20, seed=7) + positions(n=20, seed=99)
    ok("fekk stillingar å jamføre", len(pos) >= 12, len(pos))
    if not os.path.exists(ENGINE):
        # Bygg han: nais5 breidd ut til 18 globale (nullkolonner) og eksportert.
        # Trengst berre éin gong; ligg utanfor git.
        base = os.path.join(ROOT, "engines", "nais5-nn.pt")
        if os.path.exists(base):
            subprocess.run([sys.executable, "tools/expand_net.py", base,
                            os.path.join(ROOT, "engines", "tmp-g18.pt")],
                           cwd=ROOT, capture_output=True, timeout=300)
            subprocess.run([sys.executable, "-m", "sahkku_ai.export",
                            "--net", os.path.join(ROOT, "engines", "tmp-g18.pt"),
                            "--out", os.path.join(ROOT, "engines", "tmp-g18"),
                            "--name", "g18"], cwd=ROOT, capture_output=True, timeout=300)
        ok("fekk bygd ein 18-global motor", os.path.exists(ENGINE), ENGINE)

    if os.path.exists(ENGINE):
        fens = [E.fen_of(st) if hasattr(E, "fen_of") else None for st in pos]
        if any(f is None for f in fens):
            from sahkku_ai.fen import to_fen
            fens = [to_fen(st) for st in pos]
        payload = json.dumps([{"fen": f} for f in fens])
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write(payload)
            path = fh.name
        try:
            r = subprocess.run(["node", ENGINE, "features-test"],
                               stdin=open(path), capture_output=True, text=True, timeout=180)
        finally:
            os.unlink(path)
        ok("node-sida svara", r.returncode == 0 and r.stdout.strip().startswith("["),
           (r.stdout + r.stderr)[-200:])
        if r.returncode == 0 and r.stdout.strip().startswith("["):
            js = json.loads(r.stdout)
            worst, wix = 0.0, -1
            nz16 = nz17 = 0
            for st_, j in zip(pos, js):
                g = FT.encode_state(st_)[1]
                for i in range(18):
                    d = abs(float(g[i]) - float(j["glob"][i]))
                    if d > worst:
                        worst, wix = d, i
                nz16 += int(g[16] > 0)
                nz17 += int(g[17] > 0)
            ok("alle 18 globale tal er like i Python og JS", worst < 2e-5,
               f"største avvik {worst:.2e} ved g[{wix}]")
            ok("g[16] er i bruk i ekte stillingar", nz16 >= 2, f"{nz16}/{len(pos)} stillingar >0")
            ok("g[17] er i bruk i ekte stillingar", nz17 >= 2, f"{nz17}/{len(pos)} stillingar >0")

    print(f"test_exposure: {PASS} pass, {FAIL} fail")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
