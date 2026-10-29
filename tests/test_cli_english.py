"""CLI-tekst i Python-verktøya skal vere ENGELSK.

Kvifor: koden er norsk i kommentarar og docstrings (det er fint), men alt som
kjem ut i terminalen — print(), argparse-help, SystemExit, raise — skal vere
engelsk, so han kan lesast av andre og av meg utan å blande språk i loggen.

Testen les kjeldefilene og krev at ingen KØYRANDE tekst har norske markørar.
Kommentarar og docstrings vert ikkje sjekka (dei er ikkje brukargrensesnitt).

Køyr: .venv/bin/python -m tests.test_cli_english
"""
from __future__ import annotations
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FILES = sorted(list((ROOT / "sahkku_ai").glob("*.py")) + list((ROOT / "tools").glob("*.py")))

# norske ord som ikkje finst i engelsk, eller æøå
NORSKE = re.compile(
    r"[æøåÆØÅ]|\b(ikkje|vert|vart|frå|spel|parti|partia|syklus|trekk|vekt|køyr|fann|"
    r"skriv|skrive|hoppa|startvekta|arbeidar|arbeidarar|sjekk|ingenting|ingen|berre|"
    r"heile|eitt|båe|føre|gjev|gjeve|kjem|kvar|kven|noko|nokre|vere|saman|utan)\b", re.I)

PASS = FAIL = 0


def ok(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" :: {extra}" if extra else ""))


def køyrbar_tekst(fil: pathlib.Path):
    """Alle strengar som faktisk går til brukaren: print(), help=, raise, SystemExit."""
    tre = ast.parse(fil.read_text(encoding="utf-8"))
    ut = []

    def legg(node, kva):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                ut.append((kva, sub.value, sub.lineno))

    for node in ast.walk(tre):
        if isinstance(node, ast.Call):
            navn = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if navn == "print":
                legg(node, "print")
            elif navn in ("add_argument",):
                for kw in node.keywords:
                    if kw.arg in ("help", "metavar", "dest") and isinstance(kw.value, ast.Constant):
                        ut.append(("help", kw.value.value, kw.value.lineno))
            elif navn in ("SystemExit",):
                legg(node, "SystemExit")
            elif navn in ("ArgumentParser",):
                for kw in node.keywords:
                    if kw.arg == "description" and isinstance(kw.value, ast.Constant):
                        ut.append(("description", kw.value.value, kw.value.lineno))
        if isinstance(node, ast.Raise) and node.exc is not None:
            legg(node.exc, "raise")
    return ut


def main():
    tot = 0
    for fil in FILES:
        try:
            tekstar = køyrbar_tekst(fil)
        except SyntaxError as e:
            ok(f"{fil.name}: kan parsast", False, str(e))
            continue
        treff = [(kva, t, ln) for kva, t, ln in tekstar if NORSKE.search(t)]
        # f- Strengar med nøklar/format er greie; vi ser berre på ord
        treff = [x for x in treff if len(x[1].strip()) > 3]
        tot += len(tekstar)
        ok(f"{fil.name}: ingen norsk i CLI-tekst ({len(tekstar)} tekstar)",
           not treff, "; ".join(f"{kva} L{ln}: {t[:48]!r}" for kva, t, ln in treff[:3]))
    print(f"test_cli_english: {PASS} pass, {FAIL} fail ({tot} tekstar sjekka)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
