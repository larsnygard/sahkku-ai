"""Skriv eit oppteke parti (selfplay-opptak) som SPG.

Formatet er det same som js/spg.js lagar (SPG-V1): tags, opningslinje, så éi
linje per kast med trekka under. Kapteiningar («x [..]», «§») er utelatne —
dei er valfrie i notasjonen, og parsaaren i sahkku_ai/spg.py godtek linjer
utan dei. Det som må stemme er (brikkebokstav, frå, til) + terningverdi.

Køyr:  .venv/bin/python tools/write_spg.py <opptak.json> <ut.spg>
"""
import json
import sys

sys.path.insert(0, "/home/lars/src/sahkku-ai")
from sahkku_ai import engine as E

DICE = {"X": "X", "II": "II", "III": "III", "_": "_"}


def letter(p):
    if p.t == "q":
        return "Q"
    if p.t == "k":
        return "k"
    return "M" if p.o == 0 else "K"


def sq(i):
    return E.sq_name(i)


def move_line(st, mv):
    """Ei SPG-trekklinje for eitt trekk (same reglar som js/spg.js)."""
    p = E.piece(st, mv["pid"])
    pre = "II: " if mv["v"] == 2 else ("III: " if mv["v"] == 3 else "")
    to = E.move_dest(st, mv)
    if mv.get("kind") == "rel":
        return "   X->" + sq(to)          # vekking: frå-felt finst ikkje
    frm = E.soldier_at(p.o, p.at) if p.t == "s" else p.at
    return "   " + pre + letter(p) + sq(frm) + "->" + sq(to)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    g = json.load(open(src))
    rec = g["record"]

    # spel partiet opp att i Python for å ha ein state å lese brikker/felt frå
    st = E.State(first=rec["first"])
    st.queen_mode = True
    st.phase = "roll"
    st.turn = rec["first"]

    out = []
    out.append('[Event "AIL0.6 — langt uavgjort parti (sein dronning, side %s)"]'
               % ("Menn" if rec.get("qd_side") == 0 else "Koner"))
    out.append('[Date "2026-10-27"]')
    out.append('[Time "12:00:00"]')
    out.append('[UTC "2026-10-27T12:00:00Z"]')
    out.append('[Menn "%s"]' % ("AIL0.5 (sein dronning)" if rec.get("qd_side") == 0 else "AIL0.5"))
    out.append('[Koner "%s"]' % ("AIL0.5 (sein dronning)" if rec.get("qd_side") == 1 else "AIL0.5"))
    out.append('[Result "%s"]' % ("1-0" if st.winner == 0 else ("0-1" if st.winner == 1 else "*")))
    out.append("SPG-V1")
    out.append("")

    o = rec["opening"][0]
    x = lambda d: sum(1 for v in d if v == "X")
    out.append("1. open: M(%s) K(%s) -> %s" % (
        " ".join(o["p0"]), " ".join(o["p1"]),
        "M" if x(o["p0"]) > x(o["p1"]) else "K"))

    # Sida i kvar ply er gjeven av vekslinga: kvart kast startar ein ny tur,
    # so ply 0 er opningssida, ply 1 den andre, ply 2 den fyrste att, osb.
    # (Verifisert mot opptaket: kvar ply sine trekk høyrer til nøyaktig éi side.)
    first = rec["first"]
    ply_no = 0
    applied = 0
    for i, ply in enumerate(rec["plies"]):
        side_i = first if i % 2 == 0 else 1 - first
        side = "M" if side_i == 0 else "K"
        for ix, dice in enumerate(ply["dice"]):
            ply_no += 1
            out.append("%d. %s%s: %s" % (ply_no, side, "" if ix == 0 else " (omkast #%d)" % ix,
                                         " ".join(dice)))
            if ix > 0:
                out.append("   X!")
        # set terningane slik at trekka kan lesast ut
        st.turn = side_i
        st.dice = list(ply["dice"][-1])
        st.used = [d == "_" for d in st.dice]
        st.phase = "move"
        for mv in ply["moves"]:
            out.append(move_line(st, mv))
            E.apply_move(st, mv)
            applied += 1
        # turen er avslutta av trekka sjølv (terningane brukte) — elles: pass.
        # Berre ÉIN gong: før kalla vi pass_remaining på toppen av ein tur som
        # alt var avslutta, og då vart turen snudd to gonger (og sidene bytte).
        if st.phase == "move":
            E.pass_remaining(st)
        if st.phase == "over":
            break

    out.append("")
    text = "\n".join(out)
    open(dst, "w").write(text)
    print(json.dumps({"fil": dst, "ply_linjer": ply_no, "trekk": applied,
                      "slutt_fen": __import__("sahkku_ai.fen", fromlist=["x"]).to_fen(st),
                      "vinnar": st.winner}, ensure_ascii=False))


if __name__ == "__main__":
    main()
