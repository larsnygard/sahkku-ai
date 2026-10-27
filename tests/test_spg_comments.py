"""SPG 1.2: tidsstempel i hovudet og PGN-kommentarar.

Testar at
  * [Time]/[UTC] vert lesne som tags (og at Date/Time/UTC er *starttida*),
  * {…} og ; kommentarar kan stå på eiga linje, etter eit trekk, etter ei
    kastelinje og over fleire linjer,
  * kommentarane ikkje rører spelet: same hendingar og same slutt-FEN med og
    utan kommentarar,
  * «1-0 {dronning}» framleis gjev årsak = dronning (det er ikkje ein
    kommentar, men resultat-merket).

Køyr:  .venv/bin/python -m tests.test_spg_comments [fil.spg]
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sahkku_ai import fen as FEN
from sahkku_ai import spg as SPG

OK = []
DEFAULT_FIXTURE = os.path.join(os.path.dirname(__file__), "..", "analysis",
                              "ail06-undecided-768ply.spg")


def check(cond, msg):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    OK.append(msg)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FIXTURE
    base = open(path).read()
    check("[Date" in base and "SPG-V1" in base, "grunnfila er gyldig SPG")

    # ---- 1. grunnfila: replay og slutt-FEN -----------------------------
    p0 = SPG.parse_spg(base)
    st0, s0 = SPG.game_from_spg(base, strict=True)
    fen0 = FEN.to_fen(st0)
    check(len(s0) > 0, "grunnfila spelar opp att")

    # ---- 2. same fil med tidsstempel og kommentarar --------------------
    c = base
    # Set inn berre om fila ikkje alt har dei (nyare eksportar har dei).
    if "[Time " not in c:
        c = c.replace('[Date "', '[Time "19:42:07"]\n[UTC "2026-10-27T17:42:07Z"]\n[Date "', 1)
    c = c.replace("SPG-V1\n", "SPG-V1\n\n{Kommentar på eiga linje, før spelet.}\n", 1)
    lines = c.splitlines()
    # (a) kommentar etter ei kastelinje, (b) ; på ei anna kastelinje,
    # (c) kommentar over to linjer etter eit trekk
    did = {"roll": False, "semi": False, "multi": False}
    for i, ln in enumerate(lines):
        if not did["roll"] and re.match(r"^\d+\.\s+[MK]:\s", ln):
            lines[i] = ln + "   {kommentar etter kastelinja}"
            did["roll"] = True
            continue
        if not did["semi"] and re.match(r"^\d+\.\s+[MK]:\s", ln):
            lines[i] = ln + "   ; kommentar til linjeslutt, skal ignorerast"
            did["semi"] = True
            continue
        if not did["multi"] and ln.startswith("   ") and "->" in ln:
            lines[i] = ln + "\n   {fyrste linje av ein kommentar\n   som held fram på neste linje.}"
            did["multi"] = True
    c = "\n".join(lines)
    c = c.rstrip("\n") + "\n"      # resultatlinja «*» = uavgjort/ikkje ferdig

    p1 = SPG.parse_spg(c)
    # ---- 3. tags -------------------------------------------------------
    check(bool(re.match(r"^\d\d:\d\d:\d\d$", str(p1["tags"].get("Time", "")))),
          f"[Time] lesen som tag ({p1['tags'].get('Time')})")
    check(bool(re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$",
                        str(p1["tags"].get("UTC", "")))),
          f"[UTC] lesen som tag ({p1['tags'].get('UTC')})")
    check("Date" in p1["tags"], "[Date] framleis der")
    check(len(p1.get("comments", [])) >= 4,
          f"kommentarane vart fanga ({len(p1.get('comments', []))} stk)")
    joined = " | ".join(p1.get("comments", []))
    check("eiga linje" in joined, "{…} på eiga linje vart lesen")
    check("etter kastelinja" in joined, "{…} etter kastelinje vart lesen")
    check("linjeslutt" in joined, "; kommentar vart lesen")
    check("held fram" in joined, "fleirlinjes {…} vart lesen")

    # ---- 4. kommentarane rører ikkje spelet ----------------------------
    check(p1["events"] == p0["events"],
          f"same hendingar med og utan kommentarar ({len(p1['events'])} mot {len(p0['events'])})")
    st1, s1 = SPG.game_from_spg(c, strict=True)
    check(len(s1) == len(s0), f"same tal trekk ({len(s1)} mot {len(s0)})")
    check(FEN.to_fen(st1) == fen0, "same slutt-FEN med kommentarar")

    # ---- 5. resultatlinja sitt {…} er ÅRSAK, ikkje kommentar -----------
    tiny = ('[Event "x"]\n[Date "2026-10-27"]\n[Menn "a"]\n[Koner "b"]\n'
            '[Result "1-0"]\nSPG-V1\n\n'
            '1. open: M(X _ _) K(_ _ _) -> M\n'
            '1. M: X _ _\n'
            '   X->B1\n'
            '1-0 {dronning}\n')
    p2 = SPG.parse_spg(tiny)
    check(p2["result"] == "1-0", "resultatlinja vert lesen")
    check(p2["reason"] == "dronning", "«{dronning}» på resultatlinja er årsak, ikkje kommentar")
    check("dronning" not in " ".join(p2.get("comments", [])),
          "årsaka hamna ikkje i kommentarlista")

    print(f"SPG-KOMMENTARTEST OK: {len(OK)} sjekkar")
    for m in OK:
        print("  ok", m)


if __name__ == "__main__":
    main()
