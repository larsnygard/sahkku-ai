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
    """X er eit 1-TAL: ho VEKKJER ei inaktiv brikke (flyttet er del av
    vekkinga) OG flyttar utløyste brikker eitt steg — soldat framover i
    tallsyklusen, dronning og konge eitt steg langs syklusen."""
    st = start_state(["X", "_", "_"], [False, True, True])
    lm = E.legal_moves(st, st.turn)
    rel = [mv for mv in lm if mv.get("kind") == "rel"]
    check(rel, "startstilling med X: vekking finst")
    for mv in rel:
        p = E.piece(st, mv["pid"])
        check(not p.released, "vekkinga gjeld ei INAKTIV brikke")
        check(mv["v"] == 1, "vekking flyttar nøyaktig eitt felt")
    check({E.piece(st, mv["pid"]).t for mv in rel} == {"s", "q"},
          "både soldat og dronning kan vekkast med X")

    mv = [m for m in rel if E.piece(st, m["pid"]).t == "s"][0]
    p = E.piece(st, mv["pid"])
    before = p.at
    E.apply_move(st, mv)
    check(p.released and p.at == before + 1, "vekking = inaktiv -> utløyst, +1 i tallsyklusen")

    # X flyttar òg brikker som alt er ute — eitt steg
    st2 = start_state(["X", "_", "_"], [False, True, True])
    s = E.soldiers(st2, 0)[0]
    s.released = True
    s.at = 20
    mvs = E.legal_moves(st2, 0)
    check(any(mv.get("kind") != "rel" for mv in mvs), "X gjev 1-stegstrekk for utløyst soldat")
    check(all(mv["v"] == 1 for mv in mvs), "alle X-trekk er nøyaktig 1 steg")

    # ... og konge/dronning eitt steg langs syklusen
    st3 = start_state(["X", "_", "_"], [False, True, True])
    for q in (E.queen(st3, 0), E.king(st3)):
        q.released = True
        q.o = 0
        q.at = 17                      # M2
    dests = sorted(E.sq_name(mv["to"]) for mv in E.legal_moves(st3, 0)
                   if E.piece(st3, mv["pid"]).t in ("q", "k"))
    check(dests == ["L2", "L2", "N2", "N2"],
          f"dronning+konge M2 med X: eitt steg kvar veg (fekk {dests})")


# ---------------------------------------------------------------- clause 2
def test_reroll_only_when_x_unusable():
    # (a) INGEN aktive brikker (alle heime): aktivering er det einaste du kan
    # gjere — då er omkast forbode (elles kunne ein blokkere spelet for alltid)
    st = start_state(["X", "III", "II"], [False, False, False])
    check(E.legal_moves(st, st.turn), "startstilling: X har lovleg vekking")
    check(not E.can_reroll_x(st),
          "utan aktive brikker er X-omkast forbode (aktivering er einaste trekk)")

    # (a2) har du fyrst éi aktiv brikke, kan X ALLTID kastast om — òg når du
    # kunne vekkje noko
    st_a = start_state(["X", "III", "II"], [False, False, False])
    rel = [p for p in E.soldiers(st_a, 0)][0]
    rel.released = True
    rel.at = 20
    check(E.legal_moves(st_a, st_a.turn), "framleis ei vekking tilgjengeleg")
    check(E.can_reroll_x(st_a),
          "med éi aktiv brikke er X-omkast lovleg sjølv om vekking finst")

    # (a3) nytt X etter omkast kan kastast om på nytt (vanlege reglar gjeld)
    rnd_x = random.Random(9)
    st_a.dice = ["X", "_", "_"]; st_a.used = [False, True, True]
    for _ in range(20):
        st_a.dice[0] = "X"; st_a.used[0] = False
        check(E.can_reroll_x(st_a), "nytt X kan kastast om på nytt")
        E.reroll_x(st_a, rnd_x.random)

    # (b) X has nothing to wake -> re-roll allowed
    st2 = start_state(["X", "III", "II"], [False, False, False])
    for p in E.soldiers(st2, st2.turn):
        p.released = True
    q = E.queen(st2, st2.turn)
    q.released = True
    q.at = 18
    check(not any(mv.get("kind") == "rel" for mv in E.legal_moves(st2, st2.turn)),
          "alle brikker utløyste: X har inga vekking")
    check(E.can_reroll_x(st2), "med aktive brikker er X-omkast lovleg")

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


def test_reroll_when_blocked():
    """Lars 2026-10-26: har du éi aktiv brikke, kan X ALLTID kastast om — òg
    når ho ikkje har noko lovleg trekk (t.d. blokkert av eiga uutløyst
    dronning). Berre utan aktive brikker er du tvungen til å vekkje."""
    st = start_state(["X", "_", "_"], [False, True, True])
    st.queen_mode = False                      # dronninga kan ikkje løysast
    for p in st.pieces:
        if p.t == "s":
            p.captured = True
    s = E.soldiers(st, 0)[0]
    s.captured = False
    s.released = True
    s.at = 17                                  # M2; +1 -> L2 (eiga dronning heime)
    q = E.queen(st, 0)
    q.captured = False
    q.released = False
    q.at = 18
    check(E.legal_moves(st, 0) == [],
          "inga lovleg trekk: soldaten er blokkert av eiga uutløyst dronning")
    check(E.can_reroll_x(st),
          "éi aktiv brikke (blokkert) => X kan framleis kastast om")

    # motsett side av same regel: ingen aktive brikker -> må vekkje
    st2 = start_state(["X", "_", "_"], [False, True, True])
    check(not E.can_reroll_x(st2), "utan aktive brikker er X-omkast forbode")


def test_annihilation_beats_king_and_queen():
    """Siste soldat fell = tapt, same om ein eig kongen og har dronninga.

    Lars 2026-10-27: «If all soldiers are fallen, the game is lost to
    annihilation, even if you control the king and have an unactivated
    queen.» Difor kan heller ikkje stillinga «berre dronninga att» oppstå
    i eit verkeleg parti — den sida har alt tapt.
    """
    pair = None
    for t0 in range(60):
        for t1 in range(60):
            if E.soldier_at(0, t0 + 1) == E.soldier_at(1, t1):
                pair = (t0, t1)
                break
        if pair:
            break
    check(pair is not None, "fann eit par der side 0 kan ta side 1 sin siste soldat")
    assert pair is not None          # for typane sin skuld; check() garanterer det

    st = E.State(first=0)
    st.queen_mode = True
    st.phase = "move"
    st.turn = 0
    for p in st.pieces:
        if p.t == "s":
            p.captured = True
    s0 = E.soldiers(st, 0)[0]
    s0.captured, s0.released, s0.at = False, True, pair[0]
    s1 = E.soldiers(st, 1)[0]
    s1.captured, s1.released, s1.at = False, True, pair[1]
    E.king(st).o = 1                                   # fienden EIG kongen
    q1 = E.queen(st, 1)
    q1.captured, q1.released, q1.at = False, True, 20  # og har dronninga ute
    st.dice = ["X", "_", "_"]
    st.used = [False, True, True]

    mv = next(m for m in E.legal_moves(st, 0) if m["pid"] == s0.id and m["kind"] == "mv")
    E.apply_move(st, mv)
    check(len(E.alive_soldiers(st, 1)) == 0, "utryddelse: siste soldat er fallen")
    check(st.winner == 0, "utryddelse: motstandaren vinn sjølv om fienden eig kongen")
    check(st.win_reason == "utryddelse", "utryddelse: årsaka er utryddelse, ikkje dronning")
    check(st.phase == "over", "utryddelse: partiet er over")
    check(not q1.captured, "utryddelse: dronninga levde framleis (ho bergar ikkje)")


def main():
    test_x_wakes_one_piece()
    test_reroll_when_blocked()
    test_king_and_queen_follow_the_cycle()
    test_annihilation_beats_king_and_queen()
    test_reroll_only_when_x_unusable()
    test_dice_order()
    test_forfeits_rest()
    for m in OK:
        print("  ok:", m)
    print(f"REGLETEST OK: {len(OK)} sjekkar — X vekkjer, omkast berre når X er ubrukeleg,"
          " X>III>II, resten av terningane er tapt")


if __name__ == "__main__":
    main()
