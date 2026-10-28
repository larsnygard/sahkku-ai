"""Materialsklausa: eit gratis slag skal ikkje veljast bort av verdihovudet.

Bakgrunn (Lars 2026-10-28, parti mot AIL0.7): på halvtrekk 114 kunne Koner ta
ein soldat på E2 — eit fullstendig gratis slag (Menn kunne ikkje ta dronninga
att med X, II eller III). Motoren spela i staden E3->D3.

Kvifor: policyen VIL ta slaget (Lookahead=0 gjer det), men verdihovudet
vurderte stillinga etter slaget 0,08 dårlegare — og overstyrte difor, på alle
djupner (LA 4/8/12). Verditapet var innbilt: i 27 trekk vidare spel frå
stillinga etter slaget overlevde dronninga og Koner tok endå ein soldat.

Klausa: er eit trygt slag (ingen motslag) innanfor CAP_MARGIN av beste trekk,
vel vi slaget — materiale er konkret, verdiestimatet er ikkje.

Køyr:  .venv/bin/python -m tests.test_material_clause
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sahkku_ai import engine as E
from sahkku_ai import fen as FEN

OK = []
ROOT = os.path.join(os.path.dirname(__file__), "..")
ENGINE = os.path.join(ROOT, "engines", "sahkku-nn.js")
# Stillinga frå partiet, rett før halvtrekk 114 (Koner/Giera i tur, terningar II X II)
FEN_114 = (".-.-.-.-D-.-.-.-g-g-g-g-g-g-g/.-.-.-.-M-.-.-.-M-.-.-q-.-.-."
           "/m-m-m-m-m-m-m-m-.-.-NM-M-.-.-. b move 2X2 000 K1 8:4 113 *")


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    OK.append(msg)


def bestmove(cap_margin=None, lookahead=None):
    cmds = ["uci"]
    if cap_margin is not None:
        cmds.append(f"setoption name CapMargin value {cap_margin}")
    if lookahead is not None:
        cmds.append(f"setoption name Lookahead value {lookahead}")
    cmds += ["position fen " + FEN_114, "go", "quit"]
    out = subprocess.run(["node", ENGINE], input="\n".join(cmds) + "\n",
                         capture_output=True, text=True, timeout=120).stdout
    for ln in out.splitlines():
        if ln.startswith("bestmove"):
            return ln.split()[1]
    raise SystemExit("motoren svara ikkje")


def main():
    st = FEN.from_fen(FEN_114)
    # 1. slaget finst, og det er gratis
    caps = []
    for mv in E.legal_moves(st, st.turn):
        land = E.move_dest(st, mv)
        vic = [q for q in E.occ(st, land) if q.o is not None and q.o != st.turn
               and q.id != mv["pid"]]
        if vic:
            caps.append((mv, land, vic))
    check(len(caps) == 1, f"éin lovleg kapsel i stillinga ({len(caps)})")
    mv, land, vic = caps[0]
    check(E.piece(st, vic[0].id).t == "s", "kapselen tek ein soldat")
    check(E.sq_name(land) == "E2", f"kapselen landar på E2 ({E.sq_name(land)})")

    after = st.clone()
    E.apply_move(after, mv)
    E.settle(after)
    q = E.queen(after, st.turn)
    taken_back = False
    for v, face in ((1, "X"), (2, "II"), (3, "III")):
        probe = after.clone()
        probe.phase = "move"
        probe.turn = 1 - st.turn
        probe.dice = [face, "_", "_"]
        probe.used = [False, True, True]
        for m2 in E.legal_moves(probe, probe.turn):
            if E.move_dest(probe, m2) == q.at:
                taken_back = True
    check(not taken_back, "dronninga kan ikkje takast att med X, II eller III (gratis slag)")

    # 2. motoren: utan klausa spelar han roleg, med klausa tek han slaget
    utan = bestmove(cap_margin=0)
    med = bestmove(cap_margin=0.12)
    check(utan == "E3-D3", f"utan klausa: roleg trekk ({utan})")
    check(med == "E3-E2", f"med klausa: tek det gratis slaget ({med})")
    check(bestmove() == "E3-E2", "klausa er PÅ som standard")

    # 3. policyen åleine vil ta slaget — det er verdihovudet som avviste det
    pol = bestmove(lookahead=0)
    check(pol == "E3-E2", f"rein policy tek slaget ({pol}) — difor er det verdiestimatet som sviktar")

    print(f"MATERIALKLAUSE-TEST OK: {len(OK)} sjekkar")
    for m in OK:
        print("  ok", m)


if __name__ == "__main__":
    main()
