"""Omstart av ei køyring må halde fram frå siste syklus, ikkje frå startvekta.

Feilen dette pinnar (han kosta 13 syklusar på treningsboksen): valet av
startvekt låg inline i main() med «if args.init:» FYRST, so kvar gong ei
køyring vart starta på nytt — eller delt i blokker — trena han frå --init i
staden for frå siste fullførte syklus. Kvart sjekkpunkt vart då i røynda eitt
syklus oppå startvekta, og loggen sa «[init] continuing from engines/nais7-nn.pt»
der han skulle sagt «... (last completed cycle)».

Testen kallar valfunksjonen direkte, so han treng inga trening og går på
brøkdelen av eit sekund:  .venv/bin/python3 -m tests.test_resume
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

from sahkku_ai.pipeline import choose_start_net

PASS = FAIL = 0


def ok(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" :: {extra}" if extra else ""))


def make_ckdir(cycles):
    d = tempfile.mkdtemp(prefix="ck-")
    pid = os.getpid()
    for c in cycles:
        open(os.path.join(d, f"c{c}.pt"), "wb").write(b"x")
    open(os.path.join(d, f"init-{pid}.pt"), "wb").write(b"x")
    return d


def main():
    # 1. NY køyring (start_cycle 1): --init gjeld, ingen syklusar å halde fram frå
    d = make_ckdir([])
    path, msg = choose_start_net(d, 1, init="engines/nais7-nn.pt")
    ok("ny køyring startar frå --init", path == "engines/nais7-nn.pt", path)
    ok("meldinga seier kvar ho startar", "nais7" in (msg or ""), msg)
    shutil.rmtree(d, ignore_errors=True)

    # 2. OMSTART (start_cycle 5) med c1..c4 ferdige: siste syklus vinn
    d = make_ckdir([1, 2, 3, 4])
    path, msg = choose_start_net(d, 5, init="engines/nais7-nn.pt")
    ok("omstart held fram frå c4, ikkje frå --init",
       path is not None and path.endswith("c4.pt"), path)
    ok("meldinga nemner «last completed cycle»", "last completed cycle" in (msg or ""), msg)

    # 3. Syklusar ETTER startpunktet skal ikkje veljast (t.d. c9 når vi startar på 5)
    open(os.path.join(d, "c9.pt"), "wb").write(b"x")
    path, msg = choose_start_net(d, 5, init="engines/nais7-nn.pt")
    ok("hoppar ikkje på ein seinare syklus", path.endswith("c4.pt"), path)

    # 4. --force-init overstyrer med vilje
    path, msg = choose_start_net(d, 5, init="engines/nais7-nn.pt", force_init=True)
    ok("--force-init gjev --init", path == "engines/nais7-nn.pt", path)
    shutil.rmtree(d, ignore_errors=True)

    # 5. Ny køyring utan --init: lag init.pt om han manglar
    d = tempfile.mkdtemp(prefix="ck-")
    path, msg = choose_start_net(d, 1, init=None)
    ok("utan --init og utan init.pt: be om eit nytt nett", path is None, path)
    open(os.path.join(d, "init.pt"), "wb").write(b"x")
    path, msg = choose_start_net(d, 1, init=None)
    ok("utan --init men med init.pt: bruk han", path is not None and path.endswith("init.pt"), path)
    shutil.rmtree(d, ignore_errors=True)

    print(f"test_resume: {PASS} pass, {FAIL} fail")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
