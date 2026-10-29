"""Tests for badass-trainer.sh's fault reporting.

The launcher runs the pipeline and then has to explain what went wrong. That
logic is easy to break silently (it already broke once: `set -e` fired on the
launch line itself, so the diagnosis never ran). These tests slice the launch
block out of the script and run it with an injected command, so each branch is
exercised without needing a real GPU failure.

Run: .venv/bin/python -m tests.test_trainer_faults
"""
from __future__ import annotations
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "badass-trainer.sh"
PASS = FAIL = 0


def ok(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}" + (f" :: {extra}" if extra else ""))


def launch_block() -> str:
    """The part of the script from the log setup to the end (runs the pipeline)."""
    s = SCRIPT.read_text(encoding="utf-8")
    i = s.find("mkdir -p runs")
    assert i > 0, "could not find the launch section"
    return s[i:]


def run_with(command: str, run_name="tf", pre="") -> subprocess.CompletedProcess:
    """Execute the launch block with CMD replaced by `command`."""
    tmp = pathlib.Path(tempfile.mkdtemp())
    preamble = (
        'set -uo pipefail\n'
        'say() { printf "%s\\n" "$*"; }\n'
        f'RUN="{run_name}"\n'
        f'OUT="{tmp}/engines/{run_name}-nn"\n'
        f'PY="{ROOT}/.venv/bin/python3"\n'
        'cd "$1"\n'
        f'CMD=(bash -c {command!r})\n'
    )
    body = preamble + launch_block()
    p = subprocess.run(["bash", "-c", body, "bash", str(ROOT)],
                       capture_output=True, text=True, timeout=120)
    return p


def main():
    ok("script has a launch section", "mkdir -p runs" in SCRIPT.read_text())

    # 1. a GPU OOM is named as the likely cause
    p = run_with('echo "RuntimeError: CUDA out of memory. Tried to allocate"; exit 3')
    out = p.stdout + p.stderr
    ok("non-zero exit is reported as a failure", "RUN FAILED" in out, out[-200:])
    ok("the exit code is printed", "exit code 3" in out, out[-200:])
    ok("a GPU OOM is diagnosed", "GPU out of memory" in out, out[-300:])
    ok("the log tail is shown", "| " in out, out[-200:])
    ok("returns the pipeline's exit code", p.returncode == 3, p.returncode)

    # 2. a rejected flag points at the wrapper/pipeline drift
    p = run_with('echo "error: unrecognized arguments: --nope"; exit 2')
    out = p.stdout + p.stderr
    ok("a bad flag is diagnosed as wrapper drift", "out of step" in out, out[-300:])

    # 3. a checkpoint shape mismatch is diagnosed
    p = run_with('echo "RuntimeError: Error(s) in loading state_dict: size mismatch for val_fc1.weight"; exit 1')
    out = p.stdout + p.stderr
    ok("a shape mismatch is diagnosed", "do not match the current feature vector" in out, out[-300:])

    # 4. exit 0 but no finished cycle (usually the budget) -> warning + exit 1
    p = run_with('echo "[c1] self-play: 10 games"; exit 0')
    out = p.stdout + p.stderr
    ok("no finished cycle is reported", "STOPPED WITHOUT A FINISHED CYCLE" in out, out[-300:])
    ok("no-checkpoint run exits non-zero", p.returncode == 1, p.returncode)

    # 5. exit 0 with a finished cycle but no checkpoint file -> says so
    p = run_with('echo "[c1] done in 5s -> runs/tf/training_log.jsonl"; exit 0')
    out = p.stdout + p.stderr
    ok("a run without a checkpoint warns instead of claiming success",
       "wrote nothing" in out or "no c*.pt" in out, out[-300:])

    # 6. --threads overrides the auto-detected worker count
    def dry(*args):
        return subprocess.run([str(SCRIPT), "--dry", "--run", "tt",
                               "--init", "checkpoints/nais2/c4.pt", *args],
                              capture_output=True, text=True, cwd=str(ROOT), timeout=180)

    p = dry("--threads", "3")
    ok("--threads reaches the pipeline", "--workers 3" in p.stdout, p.stdout[-160:])
    ok("the summary says the count was overridden", "(--threads)" in p.stdout, p.stdout[-200:])
    ok("--threads exits 0 on a dry run", p.returncode == 0, p.returncode)

    p = dry("--threads", "40")
    ok("a count above the VRAM advice still passes", "--workers 40" in p.stdout, p.stdout[-160:])
    # NB: badass-trainer sine eigne meldingar er nynorske (berre Python-CLI-ane
    # skal vere engelske), so åtvaringa er «ÅTVARING».
    ok("...and warns about it",
       "VRAM" in p.stdout and ("ÅTVARING" in p.stdout or "warning" in p.stdout.lower()),
       p.stdout[-200:])

    p = dry("--threads", "abc")
    ok("a non-numeric count is refused", p.returncode != 0 and "whole number" in (p.stdout + p.stderr),
       (p.stdout + p.stderr)[-160:])

    p = dry("--workers", "2")
    ok("--workers still works as an alias", "--workers 2" in p.stdout, p.stdout[-160:])

    # 7. a cycle that cannot fit the time budget must be called out BEFORE the run
    p = dry("--hours", "1", "--games", "400", "--sims", "45", "--threads", "6")
    ok("one cycle longer than the budget is warned about",
       "ÅTVARING" in p.stdout and "--games" in p.stdout, p.stdout[-300:])
    p = dry("--hours", "3", "--games", "400", "--sims", "45", "--threads", "6")
    ok("no warning when the cycle fits", "ÅTVARING" not in p.stdout, p.stdout[-200:])

    print(f"test_trainer_faults: {PASS} pass, {FAIL} fail")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
