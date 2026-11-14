#!/usr/bin/env python3
"""
tools/bench_list.py — run a NAMED list of bench matches, print one RESULT
line per match. Shared by tools/train.py (the training box, local bench) and
tools/gate-publish.sh (the Nox gate, the publish bench), so a session's
--bench-list says exactly who the engine plays and how many games — both sides.

  tools/bench_list.py RUN ENGINE_JS "base:30,classic3:40,nais8:20" [SEED] [BASE_ID]

Opponent names:
  base       the current default engine — needs BASE_ID (gate) or the app's
             engines.json; the training box has no app, so 'base' is SKIPPED
             there unless BASE_ID is given
  classic3   the classic CPU at level 3: app = sahkku-ref.js (classic rules),
             box  = vendor/sahkku-ref.js — the same FLOKK, no app checkout
  <id>       any engine id: app engines/<id>.js, else this repo engines/<id>-nn.js
Games: '0' skips the match.
Prints (one line per match):
  RESULT <opp> <games> <pct>          (pct = A's score, 0-100)
Exits 1 only if NO match ran.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = pathlib.Path(os.environ.get("SAHKKU_APP",
                                  ROOT.parent / "SnartSahkku"))


def parse_list(spec: str) -> list:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        opp, _, n = part.partition(":")
        opp = opp.strip()
        try:
            games = int(n.strip())
        except ValueError:
            games = 30
        if opp:
            out.append((opp, games))
    return out


def resolve(opp: str, base_id: str) -> "tuple | None":
    """(cmd, env_extra) for one opponent, or None if unresolvable here."""
    env = {"EVAL_NESSBY": "1"}
    if opp in ("classic3", "classic3c", "classic3n"):
        # the FLOKK CPU at level 3.  vendor/ is the SYNCED COPY of the app's
        # sahkku-ref (scripts/sync_vendor.sh), so the training box benches the
        # same opponent with no app checkout.
        #   classic3 / classic3c — classic (non-nesseby) rules: the original
        #                          gate, kept under both names
        #   classic3n            — the SAME CPU under Nesseby rules.  A net
        #                          trained 50/50 must clear both; a Nesseby-
        #                          only net shows the rule-variant gap here
        #                          (blanka3: 53 % n / 7.5 % c, 2026-11-14).
        for ref in (ROOT / "vendor" / "sahkku-ref.js", APP / "engines" / "sahkku-ref.js"):
            if ref.exists():
                return (["node", str(ref)],
                        {"EVAL_NESSBY": "1" if opp == "classic3n" else "0",
                         "EVAL_SETOPT_B": "Level=3"})
        return None
    if opp == "base":
        if not base_id:
            return None
        for js in (APP / "engines" / f"{base_id}.js",
                   ROOT / "engines" / f"{base_id}.js",
                   ROOT / "engines" / f"{base_id}-nn.js"):
            if js.exists():
                return (["node", str(js)], env)
        return None
    for js in (APP / "engines" / f"{opp}.js",
               ROOT / "engines" / f"{opp}.js",
               ROOT / "engines" / f"{opp}-nn.js"):
        if js.exists():
            return (["node", str(js)], env)
    return None


def play(eng_a: str, cmd_b: list, games: int, seed: int,
         env_extra: dict, outfile: pathlib.Path,
         spgfile: pathlib.Path | None = None) -> float | None:
    env = os.environ.copy()
    env.update(env_extra)
    if spgfile is not None:
        # SPG-opptak av benken: porten (value_health/punish_rate/queen_autopsy)
        # treng partia, ikkje berre poengsummen.
        spgfile.parent.mkdir(parents=True, exist_ok=True)
        env["EVAL_SPG"] = str(spgfile)
    # eval_match spawns each side as a shell command — a bare path would hit
    # the file's exec bit (EACCES), so always run engines through node.
    if not eng_a.startswith("node "):
        eng_a = "node " + eng_a
    p = subprocess.Popen(
        ["node", "tools/eval_match.js", eng_a, " ".join(cmd_b),
         str(games), str(seed)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    lines = []
    assert p.stdout is not None
    for line in p.stdout:                    # STREAM: the caller's progress
        lines.append(line)                   # bar counts games as they land
        if line.startswith("parti "):
            print(line.rstrip(), flush=True)
    p.wait()
    out = "".join(lines)
    pathlib.Path(outfile).write_text(out)
    m = re.search(r"A-poengsum:\s*([\d.]+)", out)
    return float(m.group(1)) if m else None


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    run, eng_a, spec = sys.argv[1], sys.argv[2], sys.argv[3]
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 181
    base_id = sys.argv[5] if len(sys.argv) > 5 else ""
    if not base_id:
        ej = APP / "engines" / "engines.json"
        if ej.exists():
            import json
            base_id = json.loads(ej.read_text()).get("default") or ""
    eng = pathlib.Path(eng_a)
    if not eng.exists():
        sys.exit(f"engine not found: {eng}")
    (ROOT / "runs" / "eval").mkdir(parents=True, exist_ok=True)

    ran = 0
    for i, (opp, games) in enumerate(parse_list(spec)):
        if games <= 0:
            print(f"SKIP {opp}: 0 games")
            continue
        res = resolve(opp, base_id)
        if res is None:
            print(f"SKIP {opp}: not resolvable on this machine")
            continue
        cmd_b, env_extra = res
        out = ROOT / f"runs/eval/{run}_{opp}.txt"
        print(f"BENCH {run} vs {opp} ({games} games, seed {seed0 + i})")
        spg = ROOT / f"runs/eval/{run}_{opp}.spg"
        pct = play(str(eng), cmd_b, games, seed0 + i, env_extra, out, spg)
        if pct is None:
            print(f"RESULT {opp} {games} 0.0  (no score — see {out})")
        else:
            print(f"RESULT {opp} {games} {pct:.1f}")
        ran += 1
    sys.exit(0 if ran else 1)


if __name__ == "__main__":
    main()
