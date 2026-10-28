"""Den billige terning-maska må gje NØYAKTIG same svar som trekk-generatoren.

`_usable_mask` seier kva terningverdiar (X/II/III) ei side kan nytte i
stillinga, og vert kalla éin gong per MCTS-node. Den raske utgåva reknar med
banelogikk i staden for å byggje tre fulle trekk-mengder — men då må ho vere
eins tydande med generatoren, elles lærer nettet feil ting om kva som er lov.

To feil vart fanga av nettopp denne testen då ho vart skriven:
  * konge/dronning går òg langs TALLSYKLUSEN, ikkje berre rette liner
    (190 avvik utan den biten);
  * ei heimeståande dronning kan ikkje flyttast, berre vekkjast
    (254 avvik då ho vart tald som rørleg).

Køyr:  .venv/bin/python -m tests.test_usable_mask [parti]
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sahkku_ai import engine as E
from sahkku_ai.features import _usable_mask

OK = []


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    OK.append(msg)


def generator_mask(st, side):
    """Fasit: kva verdiar generatoren faktisk gjev trekk for."""
    mask = 0
    save = (list(st.dice), list(st.used), st.phase, st.turn)
    try:
        st.turn = side
        st.phase = "move"
        for v, face in ((1, "X"), (2, "II"), (3, "III")):
            st.dice = [face, "_", "_"]
            st.used = [False, True, True]
            for mv in E.legal_moves(st, side):
                if mv.get("v") == v:
                    mask |= 1 << (v - 1)
                    break
    finally:
        st.dice, st.used, st.phase, st.turn = save
    return mask


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rnd = random.Random(9)
    n = bad = 0
    first = None
    for g in range(games):
        st = E.State(first=g % 2)
        E.opening_roll(st, rnd.random)
        for _ in range(50):
            if st.winner is not None:
                break
            if st.phase == "roll":
                E.roll(st, rnd.random)
                st.phase = "move"
                continue
            lm = E.legal_moves(st, st.turn)
            if not lm:
                E.pass_remaining(st)
                continue
            for side in (0, 1):
                n += 1
                a, b = generator_mask(st, side), _usable_mask(st, side)
                if a != b:
                    bad += 1
                    if first is None:
                        first = (side, a, b)
            E.apply_move(st, lm[rnd.randrange(len(lm))])
            E.settle(st)
    check(n > 0, f"prøvde {n} stillingar")
    check(bad == 0, f"maskene er like i alle {n} stillingane ({bad} avvik, fyrste {first})")
    print(f"TERNINGMASKE-TEST OK: {len(OK)} sjekkar — {n} stillingar, 0 avvik")
    for m in OK:
        print("  ok", m)


if __name__ == "__main__":
    main()
