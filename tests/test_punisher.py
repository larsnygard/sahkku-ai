"""Straffar-motstandaren: lovleg spel, og den åtferda han finst for.

Testen pinnar fire ting:
  1. alle handlingar han vel er LOVLEGE (motoren sine eigne trekk), elles
     forgiftar han treningsdataa;
  2. han tek eit gratis slag når det finst;
  3. han går ikkje frivillig inn i fiendens rekkjevidd når han har eit trygt
     alternativ — dette er heile poenget, det NAIS gjer feil;
  4. han aktiverer fleire soldatar før han byrjar å manøvrere.

Køyr: .venv/bin/python -m tests.test_punisher
"""
from __future__ import annotations

import random
import sys

from sahkku_ai import engine as E
from sahkku_ai import features as FT
from sahkku_ai import punisher as PU

PASS = FAIL = 0


def ok(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" :: {extra}" if extra else ""))


def fresh(seed=0):
    st = E.State(first=0, nesseby=False)
    E.opening_roll(st, random.Random(seed).random)
    st.phase = "move"
    st.dice = ["X", "III", "II"]
    st.used = [False, False, False]
    st.turn = 0
    return st


def legal_ids(st):
    return {a for a, _ in FT.legal_ids(st)}


def test_capture_taken():
    """Eigenskap: finst eit TRYGT slag, tek han det (i staden for å nøle).

    Konstruerte stillingar er skjøre — kvar stilling frå ekte spel vert
    undersøkt, og dei utan trygt slag tel ikkje.
    """
    tested = 0
    rng = random.Random(17)
    for _ in range(60):
        st = E.State(first=rng.randint(0, 1), nesseby=False)
        E.opening_roll(st, rng.random)
        for _ply in range(rng.randint(14, 80)):
            if st.winner is not None:
                break
            if st.phase == "roll":
                E.roll(st, rng.random)
                st.phase = "move"
            lm = E.legal_moves(st, st.turn)
            if not lm:
                E.pass_remaining(st)
                continue
            E.apply_move(st, lm[rng.randrange(len(lm))])
            E.settle(st)
        if st.winner is not None or st.moves < 10:
            continue
        caps = FT.capture_squares(st, 1 - st.turn)
        moves = [(a, m) for a, m in FT.legal_ids(st) if m.get("pid") is not None]
        trygt = [(a, m) for a, m in moves
                 if PU._value_of(st, m) > 0 and E.move_dest(st, m) not in caps]
        if not trygt:
            continue
        tested += 1
        act = PU.pick_action(st)
        land = E.move_dest(st, act) if act and act.get("pid") is not None else None
        ok(f"tek eit trygt slag (stilling {tested})",
           land is not None and PU._value_of(st, act) > 0,
           f"valde {act}")
        if tested >= 5:
            break
    ok("fekk stillingar med trygt slag å prøve", tested > 0, tested)


def test_no_gift_when_risky():
    """Med eit trygt alternativ skal han ikkje gå inn i rekkjevidd."""
    st = fresh(2)
    for q in E.soldiers(st, 0)[:5]:
        q.released = True
    for q in E.soldiers(st, 1)[:5]:
        q.released = True
    st.dice = ["III", "II", "X"]
    st.used = [False, True, True]          # berre III
    caps = FT.capture_squares(st, 1)
    act = PU.pick_action(st)
    ok("handlar når han har val", act is not None and act.get("kind") != "pass", act)
    if act is not None and act.get("pid") is not None:
        land = E.move_dest(st, act)
        ok("held seg utanfor fiendens rekkjevidd", land not in caps,
           f"landa på {E.sq_name(land)} som er i {sorted(E.sq_name(x) for x in caps)}")


def test_activates_many():
    st = fresh(3)
    st.dice = ["X", "II", "III"]
    st.used = [False, True, True]          # berre X -> må løysast
    act = PU.pick_action(st)
    ok("løyser soldat når han har for få ute",
       act is not None and act.get("kind") == "rel", act)


def test_full_games_legal():
    """Han og spelet sjølv: alle trekk lovlege, partia må ta slutt."""
    for seed in range(6):
        rng = random.Random(100 + seed)
        st = E.State(first=rng.randint(0, 1), nesseby=True)
        E.opening_roll(st, rng.random)
        plies = 0
        bad = None
        for _ in range(2500):
            if st.winner is not None:
                break
            if st.phase == "roll":
                E.roll(st, rng.random)
                st.phase = "move"
            legal = {(a) for a, _ in FT.legal_ids(st)}
            act = PU.pick_action(st)
            if act is None or act.get("kind") in ("pass", "reroll"):
                if E.can_reroll_x(st):
                    E.reroll_x(st, rng.random)
                    continue
                E.pass_remaining(st)
                continue
            a = FT.action_id(st, act) if hasattr(FT, "action_id") else None
            if a is not None and a not in legal:
                bad = (seed, plies, act)
                break
            E.apply_move(st, act)
            E.settle(st)
            plies += 1
        ok(f"parti {seed}: alle trekk lovlege og partiet vart ferdig",
           bad is None and st.winner is not None,
           f"brot {bad} etter {plies} trekk, vinnar={st.winner}")


def main():
    test_capture_taken()
    test_no_gift_when_risky()
    test_activates_many()
    test_full_games_legal()
    print(f"test_punisher: {PASS} pass, {FAIL} fail")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
