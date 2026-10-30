#!/usr/bin/env python3
"""Read the tactic-duel numbers out of a run log (NAIS.6 styles).

The workers print one summary line per shard with the duel tally and two
behaviour metrics; this adds them up so the gate in docs/NAIS6-PLAN.md can be
checked before spending the night on more cycles.

    .venv/bin/python tools/style_report.py nais6
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINE = re.compile(r"stil: (?P<body>.+?)\)\s*$")
PAIR = re.compile(r"(few|many|kingfwd|kingguard) (\d+)-(\d+) (few|many|kingfwd|kingguard)")
DRAW = re.compile(r"\((\d+) draw\)")
REL = re.compile(r"løyste@40: (\w+) ([\d.]+) vs (\w+) ([\d.]+)")
KRANK = re.compile(r"kongeavstand: (\w+) ([\d.]+) vs (\w+) ([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="nais6")
    ap.add_argument("--log", help="explicit log path")
    a = ap.parse_args()
    log = Path(a.log) if a.log else ROOT / "runs" / f"{a.run}_run.log"
    if not log.exists():
        raise SystemExit(f"no log at {log}")

    duels: dict[tuple[str, str], list[int]] = {}
    draws: dict[tuple[str, str], int] = {}
    rel: dict[str, list[float]] = {}
    krank: dict[str, list[float]] = {}

    for line in log.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        body = m.group("body")
        for name, wa, wb, other in PAIR.findall(body):
            key = tuple(sorted((name, other)))
            d = duels.setdefault(key, [0, 0])
            if name == key[0]:
                d[0] += int(wa); d[1] += int(wb)
            else:
                d[0] += int(wb); d[1] += int(wa)
            dm = DRAW.search(body)
            if dm:
                draws[key] = draws.get(key, 0) + int(dm.group(1))
        rm = REL.search(body)
        if rm:
            rel.setdefault(rm.group(1), []).append(float(rm.group(2)))
            rel.setdefault(rm.group(3), []).append(float(rm.group(4)))
        km = KRANK.search(body)
        if km:
            # loggen skriv «fwd»/«guard»; normaliser til stilnamna
            namn = {"fwd": "kingfwd", "guard": "kingguard"}
            krank.setdefault(namn.get(km.group(1), km.group(1)), []).append(float(km.group(2)))
            krank.setdefault(namn.get(km.group(3), km.group(3)), []).append(float(km.group(4)))

    if not duels and not rel:
        raise SystemExit("no duel lines in the log — was the run started with --styles?")

    print(f"log: {log}\n")
    print("duels (column 1 wins vs column 2):")
    for key, (w0, w1) in sorted(duels.items()):
        n = w0 + w1
        share = 100.0 * w0 / n if n else 0.0
        dr = draws.get(key, 0)
        verdict = ""
        if n >= 100:                       # 2σ for a fair coin at this n
            sigma = (0.25 / n) ** 0.5 * 100
            verdict = " (clear)" if abs(share - 50) > 2 * sigma else " (within noise)"
        print(f"  {key[0]:10} {w0:4d} - {w1:<4d} {key[1]:10} {share:5.1f} %"
              f"  n={n}" + (f"  draws={dr}" if dr else "") + verdict)

    def mean(d, k):
        v = d.get(k) or []
        return f"{statistics.mean(v):.2f} (n={len(v)})" if v else "—"

    print("\nbehaviour:")
    print(f"  activated soldiers at ply 40 — few {mean(rel, 'few')} vs many {mean(rel, 'many')}")
    print(f"  king distance to enemy home — fwd {mean(krank, 'kingfwd')} vs guard {mean(krank, 'kingguard')}")
    print("\ngate (docs/NAIS6-PLAN.md): few must be >=1.0 BELOW many; "
          "fwd <=0.5, guard >=1.0")


if __name__ == "__main__":
    main()
