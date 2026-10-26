"""Sáhkku dice rules as an executable spec (per Lars, 2026-10-25).

The four clauses, straight from the rule clarification:

  1. X (sáhkku) wakes ONE inactive piece — soldier or queen — and the move is
     part of the wake: release = exactly one step/setting.
  2. X may be re-rolled when it CANNOT be used ("kan også kastes på nytt hvis
     du ikke vil ha den"), but NOT as an escape hatch when a wake exists
     ("du kan imidlertid ikke kaste om igjen X for å unngå å flytte brikkene
     dine"). The old rule allowed the latter; it is gone.
  3. Dice are used in the order X > III > II.
  4. If the active die cannot move, the remaining dice are FORFEITED — "får
     du ikke flytta 3, får du ikke flytta 2 heller" — the turn ends.

Each test below fails loudly if the engine (Python port or the JS referee it
mirrors) drifts from these. Run:  .venv/bin/python -m tests.test_rules
"""
import random
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sahkku_ai import engine as E
from sahkku_ai.fen import sq_of_name

OK = []


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        raise SystemExit(1)
    OK.append(msg)


def start_state(dice, used, turn=0):
    st = E.State(first=turn)
    st.dice = list(dice)
    st.used = list(used)
    st.dice_by = turn
    st.phase = "move"
    return st


# ---------------------------------------------------------------- clause 1
def test_x_wakes_one_piece():
    """X only ever WAKES (release, one step); never moves a released piece."""
    st = start_state(["X", "_", "_"], [False, True, True])
    lm = E.legal_moves(st, st.turn)
    check(lm, "startstilling med X: minst éi vekking")
    for mv in lm:
        p = E.piece(st, mv["pid"])
        check(mv["kind"] == "rel", "X gjev berre vekking (kind='rel')")
        check(not p.released, "vekkinga gjeld ei INAKTIV brikke")
        check(mv["v"] == 1, "vekking flyttar nøyaktig eitt felt")
        check(p.t in ("s", "q"), "både soldat og dronning kan vekkast")

    kinds = {E.piece(st, mv["pid"]).t for mv in lm}
    check(kinds == {"s", "q"}, "startstillinga gjev vekking for både soldat og dronning")

    # applied: the piece is released and stands exactly one cycle step further
    mv = [m for m in lm if E.piece(st, m["pid"]).t == "s"][0]
    p = E.piece(st, mv["pid"])
    before = p.at
    E.apply_move(st, mv)
    check(p.released and p.at == before + 1, "vekking = inaktiv -> utløyst, +1 i tallsyklusen")


# ---------------------------------------------------------------- clause 2
def test_reroll_only_when_x_unusable():
    # (a) a wake exists -> X may NOT be re-rolled ("ikkje for å unngå å flytte")
    st = start_state(["X", "III", "II"], [False, False, False])
    check(E.legal_moves(st, st.turn), "startstilling: X har lovleg vekking")
    check(not E.can_reroll_x(st),
          "X med lovleg vekking kan IKKJE kastast om (gammal regel, no forbode)")

    # (b) X has nothing to wake -> re-roll allowed
    st2 = start_state(["X", "III", "II"], [False, False, False])
    for p in E.soldiers(st2, st2.turn):
        p.released = True
    q = E.queen(st2, st2.turn)
    q.released = True
    q.at = 18
    check(E.legal_moves(st2, st2.turn) == [], "alle brikker utløyste: X har inga vekking")
    check(E.can_reroll_x(st2), "X utan bruk -> omkast er lovleg")

    # (c) re-roll replaces only the X dice, and the turn does not end
    before_turn, before_dice = st2.turn, len(st2.dice)
    rnd = random.Random(4)
    E.reroll_x(st2, rnd.random)
    check(st2.turn == before_turn, "omkast av X avsluttar IKKJE turen")
    check(len(st2.dice) == before_dice, "omkast held tre terningar")
    check(not E.can_reroll_x(st2) or E.RANK[st2.dice[0]] == E.RANK["X"],
          "etter omkast gjeld vanlege reglar for dei nye fjesa")


# ---------------------------------------------------------------- clause 3
def test_dice_order():
    st = start_state(["II", "III", "X"], [False, False, False])
    seq, vals = [], []
    for _ in range(3):
        rk = E.active_rank(st)
        seq.append(rk)
        vals.append(E.active_value(st))
        for i in range(3):
            if not st.used[i] and E.RANK[st.dice[i]] == rk:
                st.used[i] = True
                break
    check(seq == [E.RANK["X"], E.RANK["III"], E.RANK["II"]],
          f"rekkjefylgja er X > III > II (fekk {seq})")
    check(vals == [1, 3, 2], f"X=1, III=3, II=2 steg (fekk {vals})")
    check(E.active_rank(st) is None and E.active_value(st) == 0,
          "alle fjes brukte -> ingen aktiv terning")
    check(E.active_rank(start_state(["X", "_", "_"], [False, True, True])) == E.RANK["X"],
          "blanke fjes er alt brukte; X er aktiv")


# ---------------------------------------------------------------- clause 4
def test_forfeits_rest():
    """Kan du ikkje flytta III, mistar du II òg — turen sluttar."""
    # Startstilling: ingen utløyste soldatar, kongen er nøytral -> III/II har
    # ingen trekk; X er alt brukt.
    st = start_state(["III", "II", "X"], [False, False, True], turn=0)
    check(E.legal_moves(st, st.turn) == [], "startstilling: III har ingen trekk")
    check(not E.can_reroll_x(st), "omkast gjeld berre når X er aktiv")
    res = E.settle(st)
    check(res.get("turnEnded") is True, "settle seier turen er over (ikkje 'moves')")
    check(st.turn == 1, "turen gjekk til motstandaren")
    check(st.dice == [] and st.used == [], "resten av terningane (II) er FORTAPTE")
    check(st.phase == "roll", "neste side må kaste på nytt")

    # same in the real loop: III unusable -> II is never played
    st2 = start_state(["III", "II", "_"], [False, False, True], turn=0)
    E.settle(st2)
    check(st2.turn == 1, "III utan trekk -> II vert aldri spelt")


# ---------------------------------------------------------------- clause 5
def test_king_and_queen_follow_the_cycle():
    """Konge og dronning går langs TALLSYKLUSEN (åttetalet), begge vegar.

    Lars 2026-10-26: «The King can jump pieces ... from M2 to J1 and the other
    direction to O1». O1 er berre nåbar om brikka følgjer syklusen (M2→N2→O2→O1)
    — ei rett line kjem aldri dit. Ho hoppar over brikker: berre landingsfeltet
    er regulert.
    """
    st = E.State(first=1)
    for p in st.pieces:
        if p.t in ("s", "q"):
            p.captured = True
    k = E.king(st)
    k.o = 1                                  # kapt av koner
    k.at = sq_of_name("M2")
    st.dice = ["III", "_", "_"]
    st.used = [False, True, True]
    st.phase = "move"
    st.turn = 1

    def king_dest():
        return sorted(E.sq_name(mv["to"]) for mv in E.legal_moves(st, 1)
                      if E.piece(st, mv["pid"]).t == "k")

    check(king_dest() == ["J2", "O1"], f"M2 med III: J2 bakover og O1 framover (fekk {king_dest()})")

    # ... og ho hoppar over ei brikke som står mellom (N2)
    blocker = [p for p in st.pieces if p.t == "s" and p.o == 1][0]
    blocker.captured = False
    blocker.released = True
    blocker.at = 28                            # P2 t28 = N2
    check(E.soldier_at(1, blocker.at) == sq_of_name("N2"), "blokkeringa står på N2")
    check(king_dest() == ["J2", "O1"], "kongen hoppar over brikka på N2 og når O1")
    # ... men ho kan ikkje lande på henne
    blocker.at = 24                            # P2 t24 -> J2
    check(E.soldier_at(1, blocker.at) == sq_of_name("J2"), "blokkeringa står no på J2")
    check(king_dest() == ["O1"], f"kongen landar ikkje på brikke (fekk {king_dest()})")

    # dronninga følgjer same syklus: P1-dronning L2 med II -> J2 og N2
    st2 = E.State(first=0)
    q = E.queen(st2, 0)
    for p in st2.pieces:
        if p.id != q.id:
            p.captured = True
    q.released = True
    q.at = sq_of_name("L2")
    st2.dice = ["II", "_", "_"]
    st2.used = [False, True, True]
    st2.phase = "move"
    st2.turn = 0
    qd = sorted(E.sq_name(mv["to"]) for mv in E.legal_moves(st2, 0))
    check(qd == ["J2", "N2"], f"dronninga L2 med II: J2 og N2 (fekk {qd})")


def main():
    test_x_wakes_one_piece()
    test_king_and_queen_follow_the_cycle()
    test_reroll_only_when_x_unusable()
    test_dice_order()
    test_forfeits_rest()
    for m in OK:
        print("  ok:", m)
    print(f"REGLETEST OK: {len(OK)} sjekkar — X vekkjer, omkast berre når X er ubrukeleg,"
          " X>III>II, resten av terningane er tapt")


if __name__ == "__main__":
    main()
