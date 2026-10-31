"""Analyser eit parti: kva lovlege slag og loddrette trekk vart IKKJE spela?

Spel partiet opp att trekk for trekk og, for kvar avgjerd, samanlikn
- kor mange slag (landing på fiendeleg brikke) som var lovlege, og om
  trekket som vart spela var eitt av dei
- kor mange LODDRETTE trekk (dir F/B utan `to` — same kolonne opp/ned) som
  var lovlege, og om det spela trekket var eitt av dei

Køyr:  .venv/bin/python tools/analyse_game.py <fil.spg>
"""
import sys
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from sahkku_ai import engine as E
from sahkku_ai import spg as SPG


def victims_of(st, mv):
    """Kva fiendelege brikker trekket landar på?"""
    land = E.move_dest(st, mv)
    me = st.turn
    return [q for q in E.occ(st, land) if q.o is not None and q.o != me]


def is_vertical(st, mv):
    """Loddrett = konge/dronning utan `to` (då er banen same kolonne opp/ned)."""
    p = E.piece(st, mv["pid"])
    return p.t in ("q", "k") and mv.get("to") is None


def main():
    path = sys.argv[1]
    text = open(path).read()
    p = SPG.parse_spg(text)
    st = E.State()
    stats = {
        "avgjerder": 0,
        "slag_lovleg": 0, "slag_spela": 0,
        "slag_konge": 0, "slag_konge_spela": 0,
        "slag_dronning": 0, "slag_dronning_spela": 0,
        "lodd_lovleg": 0, "lodd_spela": 0,
        "avgjerder_med_slag": 0, "avgjerder_der_slag_vart_teke": 0,
        "døme_miss": [],
    }
    for ev in p["events"]:
        if ev[0] == "open":
            st = E.State(first=ev[1])
            st.queen_mode = True
            st.phase = "roll"
            continue
        if ev[0] in ("roll", "reroll"):
            st.turn = ev[2]
            st.dice = list(ev[1])
            st.used = [d == "_" for d in ev[1]]
            st.phase = "move"
            continue
        pool = E.legal_moves(st, st.turn)
        caps = [(m, victims_of(st, m)) for m in pool]
        caps = [(m, v) for m, v in caps if v]
        verts = [m for m in pool if is_vertical(st, m)]
        stats["average"] = None
        if pool:
            stats["avgjerder"] += 1
            stats["slag_lovleg"] += len(caps)
            stats["lodd_lovleg"] += len(verts)
            if caps:
                stats["avgjerder_med_slag"] += 1
            for m, v in caps:
                pt = E.piece(st, m["pid"]).t
                if pt == "k":
                    stats["slag_konge"] += 1
                elif pt == "q":
                    stats["slag_dronning"] += 1

        # finn trekket som vart spela (same metode som parsaren)
        played = None
        if ev[0] == "rel":
            for m in pool:
                if m.get("kind") == "rel" and E.move_dest(st, m) == _sq(ev[1]):
                    played = m
                    break
        else:
            for m in pool:
                pp = E.piece(st, m["pid"])
                frm = E.soldier_at(pp.o, pp.at) if pp.t == "s" else pp.at
                if E.sq_name(frm) == ev[2] and E.sq_name(E.move_dest(st, m)) == ev[3]:
                    played = m
                    break
        if played is not None and victims_of(st, played):
            stats["slag_spela"] += 1
            stats["avgjerder_der_slag_vart_teke"] += 1
            pt = E.piece(st, played["pid"]).t
            if pt == "k":
                stats["slag_konge_spela"] += 1
            elif pt == "q":
                stats["slag_dronning_spela"] += 1
        elif played is not None and is_vertical(st, played):
            stats["lodd_spela"] += 1
        elif caps and len(stats["døme_miss"]) < 5:
            stats["døme_miss"].append({
                "ply": stats["avgjerder"], "side": st.turn, "dice": st.dice,
                "slag": [(E.piece(st, m["pid"]).t, E.sq_name(E.move_dest(st, m)),
                          [E.sq_name(q.at) if q.t != "s" else E.sq_name(E.soldier_at(q.o, q.at))
                           for q in v]) for m, v in caps[:4]],
                "spela": (E.piece(st, played["pid"]).t, E.sq_name(E.move_dest(st, played)))
                         if played is not None else None,
            })
        if played is not None:
            E.apply_move(st, played)
        if st.phase == "over":
            break

    stats.pop("average", None)
    import json
    print(json.dumps(stats, ensure_ascii=False, indent=1))


def _sq(name):
    from sahkku_ai.fen import sq_of_name
    return sq_of_name(name)


if __name__ == "__main__":
    main()
