"""Gjer eit oppteke sjølvspelparti om til eit ARKIV-BUNDLE appen kan importere.

Appen sin «Importer»-knapp les berre JSON (js/archive.js sin importBundle),
ikkje SPG — han *eksporterer* SPG, men har ingen SPG-innlesing. Difor lagar
vi den forma appen faktisk tek: eit bundle med `steps` (éin per tur, med kast
og trekk) og `fens` (FEN for kvar stilling, so avspelinga vert eksakt).

Køyr:  .venv/bin/python tools/write_archive.py <opptak.json> <ut.json>
"""
import json
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, "/home/lars/src/sahkku-ai")
from sahkku_ai import engine as E
from sahkku_ai import fen as FEN


def main():
    src, dst = sys.argv[1], sys.argv[2]
    spg_path = sys.argv[3] if len(sys.argv) > 3 else None   # valfri: legg SPG-teksten i posten
    spg_text = open(spg_path).read() if spg_path else str()
    g = json.load(open(src))
    rec = g["record"]

    st = E.State(first=rec["first"])
    st.queen_mode = True
    st.phase = "roll"
    st.turn = rec["first"]
    fens = [FEN.to_fen(st)]

    first = rec["first"]
    steps = []
    applied = 0
    for i, ply in enumerate(rec["plies"]):
        side = first if i % 2 == 0 else 1 - first
        st.turn = side
        st.dice = list(ply["dice"][-1])
        st.used = [d == "_" for d in st.dice]
        st.phase = "move"
        moves = []
        for mv in ply["moves"]:
            p = E.piece(st, mv["pid"])
            frm = E.soldier_at(p.o, p.at) if p.t == "s" else p.at
            to = E.move_dest(st, mv)
            moves.append([frm, to, mv["v"], 1 if mv.get("kind") == "rel" else 0])
            E.apply_move(st, mv)
            applied += 1
            fens.append(FEN.to_fen(st))
            if st.phase == "over":
                break
        if st.phase == "move":
            E.pass_remaining(st)
        steps.append({"t": side, "rolls": [list(d) for d in ply["dice"]], "moves": moves})
        if st.phase == "over":
            break

    bundle = {
        "v": 1,
        "app": "SnartSahkku",
        "exported": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "games": [{
            "id": "ail06-undecided-" + uuid.uuid4().hex[:8],
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": "cpu",
            "engine": "AIL0.5",
            "level": 2,
            "nesseby": False,
            "first": rec["first"],
            "result": "1-0" if st.winner == 0 else ("0-1" if st.winner == 1 else "*"),
            "reason": st.win_reason or "",
            "plies": applied,
            "steps": steps,
            "fens": fens,
            "spg": spg_text,
            "note": ("AIL0.6-eksperiment: éi side (denne: %s) fekk ikkje vekkje "
                     "dronninga før ho hadde 2 soldatar att. Stoppa på plytaket "
                     "utan vinnar." % ("Menn" if rec.get("qd_side") == 0 else "Koner")),
        }],
    }
    json.dump(bundle, open(dst, "w"), ensure_ascii=False)
    print(json.dumps({"fil": dst, "trekk": applied, "steg": len(steps), "fens": len(fens),
                      "slutt_fen": fens[-1], "vinnar": st.winner}, ensure_ascii=False))


if __name__ == "__main__":
    main()
