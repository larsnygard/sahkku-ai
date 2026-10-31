"""Mål: kor ofte tek motoren eit lovleg slag i stillingar frå det same partiet?

Plukkar stillingar frå eit SPG der eit slag var lovleg, køyrer den EKSPORTERte
JS-motoren (S-UCI) på kvar av dei, og tel kor mange av bestmove-a som faktisk
slår ein fiendeleg brikke. Fyr/etter samanlikning av runner-endringa
(=2026-10-27) er dette talet som gjeld.

Køyr:  .venv/bin/python tools/measure_captures.py <spg> <engine.js> [n]
"""
import json
import subprocess
import sys
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from sahkku_ai import engine as E
from sahkku_ai import fen as FEN
from sahkku_ai import spg as SPG


def positions_with_captures(path):
    p = SPG.parse_spg(open(path).read())
    st = E.State()
    out = []
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
        caps = []
        for mv in pool:
            land = E.move_dest(st, mv)
            vic = [q for q in E.occ(st, land) if q.o is not None and q.o != st.turn]
            if vic:
                caps.append((mv, [E.piece(st, v.id).t for v in vic]))
        if caps:
            out.append({"fen": FEN.to_fen(st), "caps": [(mv["pid"], E.move_dest(st, mv)) for mv, _ in caps]})
        played = None
        if ev[0] == "rel":
            for mv in pool:
                if mv.get("kind") == "rel" and E.sq_name(E.move_dest(st, mv)) == ev[1]:
                    played = mv
                    break
        else:
            for mv in pool:
                pp = E.piece(st, mv["pid"])
                frm = E.soldier_at(pp.o, pp.at) if pp.t == "s" else pp.at
                if E.sq_name(frm) == ev[2] and E.sq_name(E.move_dest(st, mv)) == ev[3]:
                    played = mv
                    break
        if played is not None:
            E.apply_move(st, played)
        if st.phase == "over":
            break
    return out


def main():
    spg, engine = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    pos = positions_with_captures(spg)
    step = max(1, len(pos) // n)
    sample = pos[::step][:n]
    cmds = ["uci"]
    for q in sample:
        cmds += ["position fen " + q["fen"], "go"]
    cmds.append("quit")
    out = subprocess.run(["node", engine], input="\n".join(cmds) + "\n",
                         capture_output=True, text=True, timeout=1200).stdout
    best = [ln.split()[1] for ln in out.splitlines() if ln.startswith("bestmove")]
    took = 0
    details = []
    for q, bm in zip(sample, best):
        # bestmove er ein aid? S-UCI svarar med trekk-token; match mot kapsel-felta
        hit = None
        for pid, land in q["caps"]:
            if E.sq_name(land) in bm:
                hit = (pid, E.sq_name(land))
        took += 1 if hit else 0
        if len(details) < 6:
            details.append({"bestmove": bm, "slag_mogleg": [E.sq_name(l) for _, l in q["caps"][:3]],
                            "tok_slag": bool(hit)})
    print(json.dumps({"stillingar": len(sample), "tok_slag": took,
                      "prosent": round(100.0 * took / max(1, len(sample)), 1),
                      "examples": details}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
