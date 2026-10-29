"""Tidsbudsjettet skal haldast — ikkje berre «stopp når klokka har gått ut».

Feilen som gjorde at ei økt på éin time varte i to: syklus 1 vart ferdig 57,5
min inn i ein time, so sjekken «er budsjettet brukt opp?» svara nei, og ein ny
syklus på 57 min starta. No vert det òg sjekka om den NESTE syklusen REKK å
bli ferdig, ut frå kor lang den førre vart.

Køyr: .venv/bin/python -m tests.test_budget
"""
from __future__ import annotations
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python3"
PASS = FAIL = 0


def ok(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" :: {extra}" if extra else ""))


def run(name, cycles, max_seconds, timeout=180):
    for d in (ROOT / "runs" / name,):
        shutil.rmtree(d, ignore_errors=True)
    for p in (ROOT / "checkpoints").glob(name + "*"):
        shutil.rmtree(p, ignore_errors=True)
    cmd = [str(PY), "-m", "sahkku_ai.pipeline", "--run", name,
           "--init", "checkpoints/nais2/c4.pt", "--cycles", str(cycles),
           "--games", "2", "--workers", "1", "--sims", "2", "--leaf-batch", "2",
           "--steps", "2", "--batch", "8", "--nesseby", "--max-plies", "200",
           "--max-seconds", str(max_seconds), "--out", f"/tmp/{name}-nn"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    return p.stdout + p.stderr, time.time() - t0


def cycles_done(log: str) -> int:
    return len(re.findall(r"^\[c\d+\] done in", log, re.M))


def main():
    # ---- 1. godt budsjett: alle syklusane går ---------------------------
    out, wall = run("tb_ok", cycles=3, max_seconds=600)
    ok("tre syklusar med godt budsjett", cycles_done(out) == 3, cycles_done(out))
    ok("inga stoppmelding når alt får plass", "[stop]" not in out, out[-200:])

    # ---- 2. trongt budsjett: han skal ikkje starte ein syklus han ikkje rekk
    out, wall = run("tb_tight", cycles=10, max_seconds=6)
    n = cycles_done(out)
    ok("stoppa før alle 10 syklusane", n < 10, n)
    ok("stoppen er grunngjeven med budsjettet",
       "budget" in out or "tidsbudsjett" in out, out[-300:])
    ok("overshoot er under éin syklus",
       wall < 6 + 8, f"{wall:.1f}s for {n} syklusar")

    # ---- 3. den nye regelen: «rekk han ikkje?» vert sagt rett ut --------
    if "remain" in out:
        ok("meldinga seier kor mykje som står att", "needs ~" in out, out[-300:])
        ok("meldinga nemner kor lang den neste syklusen er",
           re.search(r"needs ~\d+s", out) is not None, out[-300:])
    else:
        # på ei svært rask maskin kan han treffe den gamle grensa først
        ok("(rask maskin: budsjettet vart brukt opp i staden)", "used up" in out, out[-300:])

    print(f"test_budget: {PASS} pass, {FAIL} fail")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
