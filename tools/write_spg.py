"""Skriv eit oppteke parti (frå recorder) som SPG — og lat andre skript bruke han.

Formatet er det same som js/spg.js lagar (SPG-V1): tags, opningslinje, så éi
linje per kast med trekka under. Kapteiningar («x [..]», «§») er utelatne —
dei er valfrie i notasjonen, og parsaren i sahkku_ai/spg.py godtek linjer utan
dei. Det som må stemme er (brikkebokstav, frå, til) + terningverdi.

Bruk som modul:
    from tools.write_spg import write_spg
    info = write_spg(rec, "parti.spg", {"event": "…", "menn": "…", "koner": "…"})

Frå kommandolinja:
    .venv/bin/python tools/write_spg.py <opptak.json> <ut.spg> [--event «…»]
"""
import json
import sys
from datetime import datetime, timezone

sys.path.insert(0, "/home/lars/src/sahkku-ai")
from sahkku_ai import engine as E
from sahkku_ai import fen as FEN


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


def write_spg(rec, dst, meta=None):
    """Skriv opptaket `rec` som SPG til `dst`. Returnerer eit samandrag.

    Partiet vert spela opp att undervegs (trekka må tolkast mot stillinga),
    so Resultatet kan først skrivast ETTER oppspelinga — den gamle utgåva
    skreiv «*» i alle filer fordi ho las vinnaren før spelet var spela.
    """
    meta = meta or {}
    now = datetime.now(timezone.utc)
    first = rec["first"]

    st = E.State(first=first)
    st.queen_mode = True
    st.phase = "roll"
    st.turn = first

    body, ply_no, applied = [], 0, 0
    o = rec["opening"][0]
    x = lambda d: sum(1 for v in d if v == "X")
    body.append("1. open: M(%s) K(%s) -> %s" % (
        " ".join(o["p0"]), " ".join(o["p1"]), "M" if x(o["p0"]) > x(o["p1"]) else "K"))

    for i, ply in enumerate(rec["plies"]):
        side_i = first if i % 2 == 0 else 1 - first
        side = "M" if side_i == 0 else "K"
        for ix, dice in enumerate(ply["dice"]):
            ply_no += 1
            body.append("%d. %s%s: %s" % (ply_no, side,
                                          "" if ix == 0 else " (omkast #%d)" % ix,
                                          " ".join(dice)))
            if ix > 0:
                body.append("   X!")
        st.turn = side_i
        st.dice = list(ply["dice"][-1])
        st.used = [d == "_" for d in st.dice]
        st.phase = "move"
        for mv in ply["moves"]:
            body.append(move_line(st, mv))
            E.apply_move(st, mv)
            applied += 1
        # Turen er avslutta av trekka sjølv (terningane brukte) — elles pass.
        # Berre ÉIN gong: å leggje pass på toppen av ein avslutta tur snur
        # turen to gonger og byter sidene.
        if st.phase == "move":
            E.pass_remaining(st)
        if st.phase == "over":
            break

    result = "1-0" if st.winner == 0 else ("0-1" if st.winner == 1 else "*")
    head = [
        '[Event "%s"]' % meta.get("event", "Sáhkku-parti"),
        '[Date "%s"]' % meta.get("date", now.strftime("%Y-%m-%d")),
        '[Time "%s"]' % meta.get("time", now.strftime("%H:%M:%S")),
        '[UTC "%s"]' % meta.get("utc", now.strftime("%Y-%m-%dT%H:%M:%SZ")),
        '[Menn "%s"]' % meta.get("menn", "Olbmát"),
        '[Koner "%s"]' % meta.get("koner", "Gálgut"),
        '[Result "%s"]' % meta.get("result", result),
    ]
    if st.win_reason:
        head.append('[Termination "%s"]' % st.win_reason)
    head.append("SPG-V1")
    head.append("")
    if meta.get("comment"):
        head.append("{%s}" % meta["comment"])
        head.append("")

    # Sluttlinje på same form som js/spg.js: «0-1 {dronning}» — ho er det
    # parsaren les resultatet og årsaka frå, so ho må med for at fila skal
    # vere lesbar for båe skrivarane.
    if st.winner is not None:
        body.append("%s {%s}" % (result, st.win_reason or ""))

    open(dst, "w").write("\n".join(head + body) + "\n")
    return {"fil": dst, "ply_linjer": ply_no, "trekk": applied, "resultat": result,
            "slutt_fen": FEN.to_fen(st), "vinnar": st.winner}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        raise SystemExit("bruk: write_spg.py <opptak.json> <ut.spg> [--event «…»]")
    src, dst = args[0], args[1]
    meta = {}
    for flag, key in (("--event", "event"), ("--menn", "menn"), ("--koner", "koner")):
        if flag in sys.argv and sys.argv.index(flag) + 1 < len(sys.argv):
            meta[key] = sys.argv[sys.argv.index(flag) + 1]
    rec = json.load(open(src))["record"]
    print(json.dumps(write_spg(rec, dst, meta), ensure_ascii=False))


if __name__ == "__main__":
    main()
