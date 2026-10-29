"""Kva gjorde taparsida gale i menneskepartia?

Spel partiet opp att med den ekte motoren og finn den fyrste stillinga der
taparen sin dronning stod fritt for motstandaren ETTER taparen sitt eige
trekk — augneblinken dronninga vart hengd — og kor mange trekk seinare ho
faktisk vart teken. Det er den konkrete lærdomen dei to menneskepartia ber
med seg, og det treningskøyringa NAIS.3 blandar inn.

Køyr: .venv/bin/python tools/blunder_scan.py <fil.spg> [...]
"""
from __future__ import annotations
import sys

from sahkku_ai import engine as E
from sahkku_ai.spg import parse_spg, game_from_spg_any


def queen_sq(st, side):
    q = E.queen(st, side)
    if q is None or q.captured:
        return None
    return q.at


def can_take_queen(st, by_side):
    """Kan `by_side` slå motstandarens dronning i denne stillinga?"""
    qsq = queen_sq(st, 1 - by_side)
    if qsq is None:
        return False
    for mv in E.legal_moves(st, by_side):
        if E.move_dest(st, mv) == qsq:
            return True
    return False


def scan(path, quiet=False):
    txt = open(path, encoding="utf-8", errors="replace").read()
    p = parse_spg(txt)
    p = p[0] if isinstance(p, list) else p
    result = p.get("result")
    if result not in ("1-0", "0-1"):
        print(f"{path}: uavgjort — hoppar over")
        return None
    winner = 0 if result == "1-0" else 1
    loser = 1 - winner
    _final, samples = game_from_spg_any(txt, strict=True)

    # stillinga etter kvart trekk: samples[i][0] er FØR trekk i, so
    # tilstanden etter trekk i er samples[i+1][0] (og den siste etter siste trekk
    # er sluttstillinga).
    hung_at = None
    for i, (st, mv) in enumerate(samples):
        if st.turn != loser:
            continue
        after = samples[i + 1][0] if i + 1 < len(samples) else _final
        if can_take_queen(after, winner):
            hung_at = (i, mv, after, queen_sq(after, loser))
            break

    # når vart ho faktisk teken?
    taken_at = None
    for i, (st, mv) in enumerate(samples):
        if st.turn != winner:
            continue
        after = samples[i + 1][0] if i + 1 < len(samples) else _final
        if queen_sq(after, loser) is None:
            taken_at = i
            break

    name = path.split("/")[-1]
    if hung_at is None:
        if not quiet:
            print(f"{name}: fann ingen augneblink der dronninga stod fritt etter taparen sitt trekk")
        return {"file": name, "hung": None, "taken": taken_at}
    i, mv, after, qsq = hung_at
    gap = (taken_at - i) if taken_at is not None else None
    print(f"{name}: dronninga (kone) vart hengd etter trekk {i + 1} av taparen — ho stod på "
          f"{E.sq_name(qsq)} og kunne takast med éin manøver")
    print(f"   trekk som hengde ho: {mv}")
    print(f"   teken {gap} halvtrekk seinare" if gap is not None else "   (vart ikkje teken)")
    return {"file": name, "hung": i + 1, "square": E.sq_name(qsq), "taken": taken_at, "gap": gap}


if __name__ == "__main__":
    for f in sys.argv[1:]:
        try:
            scan(f)
        except Exception as e:  # noqa: BLE001
            print(f"{f}: feil — {type(e).__name__}: {e}")
