"""«Sein dronning»-handikapet for AIL0.6 (Lars 2026-10-27).

Eksperimentet: sjølvspel der éi side aldri får vekkje dronninga før ho har
berre N soldatar att. Desse testane pinnar sjølve filteret — at det tek
bort NETTOPP dronningvekkinga (ikkje soldatvekking, ikkje vanlege trekk),
at det slepp henne gjennom når soldatane er få nok, og at det aldri tømmer
trekkmengda heilt (då ville partiet stått fast).

Køyr:  .venv/bin/python -m tests.test_queen_delay
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sahkku_ai import engine as E
from sahkku_ai import features as FT
from sahkku_ai.selfplay import filter_qd, queen_activation_blocked, qd_side_for_game

OK = []


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    OK.append(msg)


def rig(soldiers_alive=15, queen_released=False):
    """Stilling der side 0 har X og kan vekkje dronninga si."""
    st = E.State(first=0)
    st.queen_mode = True
    st.phase = "move"
    st.turn = 0
    q = E.queen(st, 0)
    q.captured = False
    q.released = queen_released
    q.at = E.idx_of(7, 1)                       # H2, slottet
    sold = E.soldiers(st, 0)
    for i, s in enumerate(sold):
        s.captured = i >= soldiers_alive        # berre dei fyrste er att
        s.released = False
    st.dice = ["X", "_", "_"]
    st.used = [False, True, True]
    return st, q


def piece_of(st, mv):
    """Brikka eit trekk gjeld, eller None (pass/omkast har ingen pid)."""
    pid = mv.get("pid")
    return E.piece(st, pid) if pid is not None else None


def is_q(st, mv):
    p = piece_of(st, mv)
    return p is not None and p.t == "q"


def is_s(st, mv):
    p = piece_of(st, mv)
    return p is not None and p.t == "s"


def queen_rels(ids, st):
    return [mv for _, mv in ids if mv.get("kind") == "rel" and is_q(st, mv)]


# 1. standardtilfellet: 15 soldatar -> dronningvekkinga bort
st, q = rig(15)
ids = FT.legal_ids(st)
check(len(queen_rels(ids, st)) > 0, "oppsett: dronningvekking finst utan handikap")
fil = filter_qd(st, ids, 0, 2)
check(len(queen_rels(fil, st)) == 0, "handikap: dronningvekkinga er borte (15 soldatar att)")
check(len(fil) > 0, "handikap: andre trekk står att (soldatvekking)")

# 2. 2 soldatar att -> ho får vekkje
st2, q2 = rig(2)
ids2 = FT.legal_ids(st2)
fil2 = filter_qd(st2, ids2, 0, 2)
check(len(queen_rels(fil2, st2)) == len(queen_rels(ids2, st2)) > 0,
      "grensa: med 2 soldatar att er dronningvekkinga lovleg att")

# 3. 3 soldatar att -> framleis forbode (grensa er «berre 2»)
st3, _ = rig(3)
check(len(queen_rels(filter_qd(st3, FT.legal_ids(st3), 0, 2), st3)) == 0,
      "grensa: 3 soldatar att er framleis forbode")

# 4. andre sida vert ikkje råka
st4, _ = rig(15)
fil4 = filter_qd(st4, FT.legal_ids(st4), 1, 2)     # handikap på side 1, side 0 i tur
check(len(queen_rels(fil4, st4)) == len(queen_rels(FT.legal_ids(st4), st4)) > 0,
      "handikapet rører ikkje motstandaren sine trekk")

# 5. vanlege dronningtrekk (ikkje vekking) er aldri blokkerte
st5, q5 = rig(15, queen_released=True)
ids5 = FT.legal_ids(st5)
moves5 = [mv for _, mv in ids5 if is_q(st5, mv) and mv.get("kind") != "rel"]
check(len(moves5) > 0, "oppsett: utløyst dronning har trekk")
check(len(filter_qd(st5, ids5, 0, 2)) == len(ids5),
      "handikapet blokkerer ikkje vanlege dronningtrekk, berre vekkinga")

# 6. aldri tøm heile mengda: eit kunstig tilfelle der alt er dronningvekking
st6, _ = rig(15)
only_q = [(a, mv) for a, mv in FT.legal_ids(st6) if mv.get("kind") == "rel"
          and is_q(st6, mv)]
check(len(only_q) > 0, "oppsett: berre dronningvekking i mengda")
check(len(filter_qd(st6, only_q, 0, 2)) == len(only_q),
      "fallback: heller vekkje dronninga enn å stå fast")

# 7. sidevalet per parti
rng = random.Random(7)
check(qd_side_for_game({"queen_delay": -1}, rng) == -1, "av: ingen handikapside")
check(qd_side_for_game({"queen_delay": 1}, rng) == 1, "fast side 1")
sider = {qd_side_for_game({"queen_delay": 2}, rng) for _ in range(40)}
check(sider == {0, 1}, "tilfeldig side: båe sider vert handikappa over mange parti")

# 8. soldatvekking er ikkje råka av filteret
st8, _ = rig(15)
rels_s = [mv for _, mv in filter_qd(st8, FT.legal_ids(st8), 0, 2)
          if mv.get("kind") == "rel" and is_s(st8, mv)]
check(len(rels_s) > 0, "handikapet let soldatvekking stå urørt")

# 9. sjølve predikatet
st9, q9 = rig(15)
mv_q = queen_rels(FT.legal_ids(st9), st9)[0]
check(queen_activation_blocked(st9, mv_q, 0, 2) is True, "predikat: dronningvekking blokkert")
check(queen_activation_blocked(st9, mv_q, 1, 2) is False, "predikat: berre for handikapsida")
check(queen_activation_blocked(st9, mv_q, 0, 20) is False, "predikat: 15 soldatar ≤ 20 -> lovleg")

print(f"QUEEN-DELAY TEST OK: {len(OK)} sjekkar")
for m in OK:
    print("  ok", m)
