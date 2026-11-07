#!/usr/bin/env python3
"""
tools/train.py — one entry point for every sáhkku training session.

  ./tools/train.py                    # curses TUI on the training box
  ./tools/train.py --go --run nais10 --init engines/nais9-nn.pt \
      --games 200 --cycles 40 --push-every 2 \
      --bench-list "base:30,classic3:40,nais8:20"       # headless runner

The runner loop, one cycle at a time (a crash costs one cycle, not the night):
  badass-trainer.sh (one cycle) -> export -> parity gate -> S-UCI gate ->
  local bench (optional) -> cycle log row -> commit + push every N-th cycle

It replaces tools/train-push.sh (export/push per cycle) and
tools/train-forever.sh (endless loop, local bench, GPU-wait):
  * --cycles is the END cycle number (pipeline semantics); the run resumes
    from the last completed cycle in runs/<run>/training_log.jsonl
  * --hours is a wall-clock budget checked BETWEEN cycles
  * the gate on the Nox box (tools/gate-publish.sh) runs the publish bench
    and deploys whenever a push lands on origin/main — --bench here is extra
The TUI collects the parameters, shows a live plan, and hands off to the very
same runner. Run it under:  tmux new -s train './tools/train.py'
"""
from __future__ import annotations

import argparse
import curses
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / ".venv/bin/python3"
PRESETS = ROOT / "tools" / "train_presets.json"

# ── defaults ──────────────────────────────────────────────────────────────
DEFAULTS: dict = {
    "run":          "blanka3",
    "init":         "engines/blanka2-nn.pt",  # '' = newest file (mtime); blanka
    #                                          # is the family trained now
    "gpu":          "auto",        # "auto" | "cpu" | nvidia-smi index "0","1"
    "cycles":       40,          # END cycle number (pipeline semantics)
    "hours":        0.0,         # 0 = no time limit; --cycles rules
    "games":        200,         # self-play games per cycle
    "sims":         45,
    "steps":        450,
    "threads":      0,           # 0 = auto (physical cores, VRAM-capped)
    "seed":         81,
    "opponent":     "punisher",  # punisher | classic3 | "" (pure self-play)
    "hand_style":   "hold",      # hold | few | many | kingfwd | kingguard | ""
    "nesseby":      True,
    "nesseby_frac": 1.0,
    "aggressive":   False,
    "styles":       False,
    "push_every":   1,           # commit+push every N-th cycle (0 = only last)
    # Bench list: "opponent:games" comma-separated, MAX 3 (the picker caps at
    # three), run per cycle (or once at session end — bench_when) AND by the
    # Nox gate (sent to it in runs/<run>_bench.txt).
    #   base      = the current default engine (gate resolves it)
    #   classic3  = the classic FLOKK CPU at level 3 (vendor/sahkku-ref.js)
    #   <id>      = any engine id (nais8, blanka2, kaiser1, …): the app's
    #               engines/<id>.js if the checkout is next door, else this
    #               repo's engines/<id>-nn.js (the box is self-sufficient)
    # The TUI picks from a POPULATED list (opp_choices()); --list-opponents.
    # Empty string / "0" = no local bench.
    "bench_list":   "classic3:40",
    "bench_when":   "cycle",     # "cycle" = bench after each cycle,
    #                            # "end"   = once, after the last cycle
    "bench_seed":   181,
    "gpu_wait_mb":  3000,        # VRAM to free before each cycle starts
    "no_push":      False,
    "dry":          False,
    "resume":       True,
}

# ── machine probes ─────────────────────────────────────────────────────────
def free_vram_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        return max(int(l.split(",")[0]) - int(l.split(",")[1])
                   for l in out.strip().splitlines())
    except Exception:
        return 999_999   # no GPU / NVML broken — never block on it

def gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        return "CPU"

def gpu_list() -> list:
    """Every CUDA card nvidia-smi sees: [{index, name, total, free}].
    Cached 10 s — the TUI probes on every redraw, nvidia-smi is ~0.1 s."""
    now = time.time()
    if _GPU_CACHE["t"] is not None and now - _GPU_CACHE["t"] < 10:
        return _GPU_CACHE["v"]
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        gpus = []
        for line in out.strip().splitlines():
            i, name, total, used = [s.strip() for s in line.split(",")]
            gpus.append({"index": i, "name": name, "total": int(total),
                         "free": int(total) - int(used)})
    except Exception:
        gpus = []
    _GPU_CACHE["v"], _GPU_CACHE["t"] = gpus, now
    return gpus
_GPU_CACHE: dict = {"v": [], "t": None}

def resolve_device(gpu: str) -> tuple:
    """'auto' | 'cpu' | nvidia-smi index → (label, CUDA_VISIBLE_DEVICES|None)."""
    g = (gpu or "auto").strip().lower()
    if g in ("cpu", "none", "off"):
        return "CPU", ""
    if g in ("", "auto"):
        return (gpu_name() if torch_available() else "CPU"), None
    if g.isdigit():
        if not torch_available():
            return f"GPU {g} (not available — CUDA missing)", None
        return f"GPU {g}: {gpu_label(g)}", g
    return gpu_name(), None

def torch_available() -> bool:
    """Cached — importing torch costs seconds and the TUI asks per redraw."""
    if _TORCH_OK["v"] is None:
        try:
            import torch
            _TORCH_OK["v"] = torch.cuda.is_available()
        except Exception:
            _TORCH_OK["v"] = False
    return _TORCH_OK["v"]
_TORCH_OK: dict = {"v": None}

def gpu_label(index: str) -> str:
    for g in gpu_list():
        if g["index"] == index:
            return f"{g['name']} ({g['free']}/{g['total']} MiB free)"
    return f"gpu #{index}"

# ── compatibility: options that fight each other ──────────────────────────
def compat_issues(p: dict) -> list:
    """Human-readable warnings for option combos that are valid but
    pointless, wasteful or self-deceiving. The TUI shows them live and
    refuses START until they are fixed (y=proceed anyway once)."""
    iss = []
    dry, push = p.get("dry"), p.get("no_push")
    benched = bool(parse_bench_spec(p.get("bench_list", "")))
    if dry and not push:
        iss.append("Dry run + push: a dry session commits and pushes nothing "
                   "— set no_push, or turn off dry.")
    if dry and benched:
        iss.append("Dry run + bench list: a dry session exports no engine, "
                   "so the bench cannot run — the list is dead weight.")
    dev, _ = resolve_device(p.get("gpu", "auto"))
    cpu = dev == "CPU"
    if p.get("bench_when") == "cycle" and benched and \
            p.get("cycles", 0) >= 30:
        iss.append("Bench after EVERY cycle over many cycles costs hours of "
                   "playing — bench_when 'end' is the cheap option.")
    if p.get("init") is not None:
        try:
            init = resolve_init(p["init"])
            if not pathlib.Path(init).exists():
                iss.append(f"init weight «{init}» is not a file.")
        except SystemExit:
            pass
    if cpu and (p.get("threads") or 0) > n_cores():
        iss.append(f"{p['threads']} workers on CPU with {n_cores()} cores — "
                   "the run crawls. 0 (auto) is right.")
    if p.get("cycles", 0) and last_cycle(p.get("run", "")) >= p["cycles"]:
        iss.append(f"run «{p['run']}» is already through cycle "
                   f"{last_cycle(p['run'])} — nothing to do; raise 'end "
                   "cycle' (Del: defaults).")
    gval = str(p.get("gpu", "auto")).strip().lower()
    if gval.isdigit() and gval not in [g["index"] for g in gpu_list()]:
        iss.append(f"GPU '{gval}' is not a card on this box "
                   f"({', '.join(g['index'] for g in gpu_list()) or 'none'}).")
    if gval not in ("auto", "", "cpu", "none", "off") and not torch_available():
        iss.append(f"GPU '{gval}' selected, but CUDA is not available here — "
                   "the run falls back to CPU.")
    if gval == "cpu" and torch_available():
        iss.append("GPU set to 'cpu' while CUDA cards exist — training will "
                   "crawl on CPU workers.")
    return iss

def n_cores() -> int:
    try:
        out = subprocess.check_output(["lscpu", "-p=Core,Socket"], text=True)
        cores = {l for l in out.splitlines() if not l.startswith("#")}
        if cores:
            return len(cores)
    except Exception:
        pass
    cpus = os.cpu_count() or 2
    return max(cpus // 2, 1)

def last_cycle(run: str) -> int:
    p = ROOT / f"runs/{run}/training_log.jsonl"
    if not p.exists():
        return 0
    try:
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return max((r.get("cycle", 0) for r in rows), default=0)
    except Exception:
        return 0

def init_choices() -> list:
    """Selectable init weights: engines/*-nn.pt, newest first (tmp-* excluded).
    The newest is the engine being trained right now — blanka2 per 2026-11-07."""
    files = [f for f in (ROOT / "engines").glob("*-nn.pt")
             if not f.name.startswith("tmp-")]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return [str(f.relative_to(ROOT)) for f in files]

def resolve_init(init: str) -> str:
    """'' means 'the newest engine weight' — the one we are training on."""
    if not init:
        ch = init_choices()
        if not ch:
            raise SystemExit("no engine weights found under engines/")
        return ch[0]
    return init

# ── bench opponents: the POPULATED list ───────────────────────────────────
MAX_BENCH = 3          # «up to three benchmarks» — the picker caps here

def opp_choices(run: str = "") -> list:
    """Selectable bench opponents: engines/*-nn.js (exported engines, newest
    first; tmp-* and the engine being trained excluded) + base + classic3."""
    files = [f for f in (ROOT / "engines").glob("*-nn.js")
             if not f.name.startswith(("tmp-", "sahkku-nn"))]
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    ids = [f.name[:-len("-nn.js")] for f in files
           if not (run and f.name.startswith(run + "-nn"))]
    return ["classic3", "base"] + ids

def parse_bench_spec(spec: str) -> list:
    """'opp:games,...' -> [(opp, games)]; '0'/'' -> []."""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or part == "0":
            continue
        opp, _, n = part.partition(":")
        try:
            games = int(n.strip() or 30)
        except ValueError:
            games = 30
        out.append((opp.strip(), games))
    return out

def clamp_bench(spec: str) -> str:
    """Keep at most MAX_BENCH matches — benching more is wasted box-hours."""
    return ",".join(f"{o}:{g}"
                    for o, g in parse_bench_spec(spec)[:MAX_BENCH])

# ── presets + params files ────────────────────────────────────────────────
def load_presets() -> dict:
    try:
        return json.loads(PRESETS.read_text())
    except Exception:
        return {}

def save_preset(name: str, p: dict):
    pres = load_presets()
    pres[name] = {k: p[k] for k in DEFAULTS}
    PRESETS.write_text(json.dumps(pres, indent=2, ensure_ascii=False) + "\n")

def load_preset(name: str, p: dict) -> dict:
    pres = load_presets()
    if name not in pres:
        avail = ", ".join(sorted(pres)) or "(none)"
        raise SystemExit(f"preset '{name}' not found — available: {avail}")
    p.update({k: pres[name][k] for k in DEFAULTS if k in pres[name]})
    return p

def write_params_file(run: str, p: dict):
    """runs/<run>_params.json — the exact parameters of this session, so a
    restart is 'train.py --params runs/<run>_params.json', no typing."""
    (ROOT / "runs").mkdir(exist_ok=True)
    path = ROOT / f"runs/{run}_params.json"
    path.write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n")
    return path

def est_cycle_min(games: int, sims: int, workers: int) -> int:
    rate_pw = 27.0 / max(1, sims)        # games/min/worker (measured, P100)
    return max(1, round(games / (max(1, workers) * rate_pw)))

# ── live progress bar (training cycles + benchmark) ───────────────────────
class Progress:
    """One status line, redrawn every few seconds from the run log: which
    cycle of how many, games done inside the cycle (worker lines the pipeline
    writes), and the bench fraction when a bench is running. train_status.py
    reads the same log; this is the live in-session view."""

    def __init__(self, run: str, cycles: int, log_path: pathlib.Path):
        self.run, self.cycles, self.log_path = run, cycles, log_path
        self.gpc = DEFAULTS["games"]
        self.bench = (0.0, "")          # fraction, label
        self.active = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def set_games_per_cycle(self, g: int):
        self.gpc = max(1, g)

    def set_bench(self, frac: float, label: str = ""):
        self.bench = (frac, label)

    def stop(self):
        self.active = False
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.flush()

    def _parse(self):
        done, cur, now = 0, 1, 0
        try:
            with open(self.log_path, "r", errors="replace") as f:
                for line in f:
                    m = re.match(r"^\[c(\d+)\] self-play: (\d+) parti", line)
                    if m:
                        cur, self.gpc = int(m.group(1)), int(m.group(2))
                        now = 0
                        continue
                    m = re.match(r"^\[c(\d+)\] ferdig", line)
                    if m:
                        done = max(done, int(m.group(1)))
                        now = self.gpc
                        continue
                    m = re.match(r"^\[w\d+\] (\d+)/(\d+) parti", line)
                    if m:
                        now = max(now, int(m.group(1)))
        except OSError:
            pass
        return done, cur, now

    def _loop(self):
        w = 24
        while self.active:
            done, cur, now = self._parse()
            frac = (done * self.gpc + min(now, self.gpc)) / \
                   max(1, self.cycles * self.gpc)
            n = max(0, min(w, int(round(frac * w))))
            if self.bench[0] > 0 and self.bench[0] < 1:
                tail = f"  BENCH {self.bench[1]} {self.bench[0] * 100:3.0f} %"
            elif self.bench[0] >= 1 and self.bench[1]:
                tail = f"  BENCH {self.bench[1]} done"
            else:
                tail = ""
            line = (f"\r{self.run} c{min(cur, self.cycles)}/{self.cycles} "
                    f"[{'#' * n}{'.' * (w - n)}] {frac * 100:5.1f} %{tail}")
            try:
                sys.stdout.write(line.ljust(99))
                sys.stdout.flush()
            except Exception:
                break
            time.sleep(4)

# ── cycle steps (ported from train-push.sh / train-forever.sh) ────────────
def _log(msg: str, log_path: pathlib.Path):
    line = f"{time.strftime('%F %T')}  {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")

def export_and_gate(run: str, k: int, log_path: pathlib.Path):
    say = lambda m: _log(m, log_path)
    ck = ROOT / f"checkpoints/{run}/c{k}.pt"
    if not ck.exists():
        ck = ROOT / f"checkpoints/{run}_c{k}/latest.pt"
    if not ck.exists():
        raise RuntimeError(f"cycle {k}: no checkpoint written")
    say(f"[c{k}] exporting {ck}")
    mat = ["--material"] if run.startswith("kaiser") else []
    subprocess.run(
        [str(PY), "-m", "sahkku_ai.export", "--net", str(ck),
         "--out", f"engines/{run}-nn", "--name", run.upper()] + mat,
        cwd=ROOT, check=True, capture_output=True)
    r = subprocess.run(
        [str(PY), "tests/test_net_parity.py", f"engines/{run}-nn.pt",
         "--js", f"engines/{run}-nn.js"],
        cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    last = tail[-1] if tail else ""
    if "OK" not in last:
        raise RuntimeError(f"[c{k}] net parity FAILED: {last}")
    say(f"[c{k}] net parity: OK")
    p = subprocess.run(["node", f"engines/{run}-nn.js"],
                       input="uci\nposition startpos\ngo depth 1\nquit\n",
                       cwd=ROOT, capture_output=True, text=True)
    if "uciok" not in p.stdout:
        raise RuntimeError(f"[c{k}] S-UCI gate failed (no uciok)")
    say(f"[c{k}] S-UCI: OK")

def bench_cycle(run: str, k: int, spec: str, seed_base: int,
                log_path: pathlib.Path, progress=None) -> list:
    """Run the local bench list; returns [(opp, games, pct)] parsed from
    bench_list.py RESULT lines. bench_list.py STREAMS one 'parti N/...' line
    per finished game, so the progress fraction advances game by game (a
    40-game match is ~30 min — a lumped call would freeze the bar). 'base'
    self-matches on the box: the freshly exported engine plays its own .js —
    same file, ~50 % — so 'base' is only meaningful with an explicit id."""
    say = lambda m: _log(m, log_path)
    eng = ROOT / f"engines/{run}-nn.js"
    if not eng.exists():
        say(f"[c{k}] bench skipped (no {eng.name})")
        return []
    env = os.environ.copy()
    env["SAHKKU_APP"] = str(ROOT.parent / "SnartSahkku")  # if the app is here
    r = subprocess.Popen(
        [str(PY), "tools/bench_list.py", run, f"engines/{run}-nn.js",
         spec, str(seed_base + k)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    results, out_lines = [], []
    total = sum(g for _, g in parse_bench_spec(spec))
    seen = 0                      # games finished, across all matches
    assert r.stdout is not None
    for line in r.stdout:
        out_lines.append(line.rstrip())
        if line.startswith(("BENCH ", "SKIP ", "RESULT ")):
            say(f"[c{k}] {line.rstrip()}")
        m = re.match(r"parti (\d+) ", line)
        if m and progress and total:
            seen += 1
            progress(min(1.0, seen / total))
    r.wait()
    out = "\n".join(out_lines)
    for m in re.finditer(r"^RESULT (\S+) (\d+) ([\d.]+)", out, re.M):
        results.append((m.group(1), int(m.group(2)), float(m.group(3))))
    if progress and total:
        progress(1.0)
    return results

def write_cycle_row(run: str, k: int, init: str, opp: str, hand: str,
                    log_path: pathlib.Path):
    say = lambda m: _log(m, log_path)
    p = ROOT / f"runs/{run}_report.md"
    if not p.exists():
        head = [f"# {run.upper()} — cycle log", "",
                f"- init: `{init}`",
                f"- diet: opponent **{opp}**, hand **{hand}**", "",
                "| cycle | samples (cum) | val loss | checkpoint |",
                "|---|---|---|---|"]
        p.write_text("\n".join(head) + "\n")
    try:
        rows = [json.loads(l) for l in
                (ROOT / f"runs/{run}/training_log.jsonl").read_text()
                .splitlines() if l.strip()]
    except Exception:
        rows = []
    samples = sum(r.get("new_samples", 0) for r in rows)
    best = None
    for cand in (ROOT / f"checkpoints/{run}/c{k}.pt",
                 ROOT / f"checkpoints/{run}_c{k}/latest.pt"):
        if cand.exists():
            try:
                import torch
                best = torch.load(cand, map_location="cpu",
                                  weights_only=False).get("meta", {}).get("best_val")
            except Exception:
                pass
            break
    line = (f"| {k} | {samples} | {best:.3f} | `checkpoints/{run}/c{k}.pt` |"
            if isinstance(best, float)
            else f"| {k} | {samples} | ? | `checkpoints/{run}/c{k}.pt` |")
    p.write_text(p.read_text().rstrip("\n") + "\n" + line + "\n")
    say(f"[c{k}] report row: {line}")

def commit_and_push(run: str, k: int, cycles: int, push_every: int,
                    init: str, opp: str, hand: str, no_push: bool,
                    log_path: pathlib.Path):
    say = lambda m: _log(m, log_path)
    if no_push:
        say(f"[c{k}] --no-push: engine stays in engines/{run}-nn.*")
        return
    if push_every not in (0, 1) and k % push_every != 0 and k != cycles:
        nxt = (k // push_every + 1) * push_every
        say(f"[c{k}] push skipped (every {push_every}-th; next at c{nxt})")
        return
    subprocess.run(["git", "add", f"engines/{run}-nn.pt"], cwd=ROOT,
                   capture_output=True)
    subprocess.run(["git", "add", "-f", f"runs/{run}_report.md"], cwd=ROOT,
                   capture_output=True)
    # hand the bench spec to the Nox gate (it benches with exactly this list)
    spec_file = ROOT / f"runs/{run}_bench.txt"
    if spec_file.exists():
        subprocess.run(["git", "add", "-f", f"runs/{run}_bench.txt"],
                       cwd=ROOT, capture_output=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT,
                      capture_output=True).returncode == 0:
        say(f"[c{k}] no staged changes — commit skipped")
        return
    msg = (f"{run} cycle {k}: exported from checkpoints/{run}/c{k}.pt\n\n"
           f"Diet: --train-opponent {opp} --hand-style {hand} on top of {init}. "
           f"Net parity and S-UCI green locally. The gate on the Nox box runs "
           f"the publish bench, bumps the version and deploys.")
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, check=True)
    say(f"[c{k}] committed")
    for attempt in (1, 2, 3):
        r = subprocess.run(["git", "push", "-q", "origin", "main"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            say(f"[c{k}] pushed — the gate publishes within ~20 min")
            return
        say(f"[c{k}] push failed (try {attempt}): {r.stderr.strip()[:120]}")
        subprocess.run(["git", "pull", "-q", "--rebase", "--autostash",
                        "origin", "main"], cwd=ROOT, capture_output=True)
        time.sleep(5)
    raise RuntimeError(f"[c{k}] push failed 3 times — commit is local; "
                       f"git push origin main before the next cycle")

def cycle_cmd(run: str, k: int, init: str, p: dict, workers: int) -> list:
    cmd = ["./badass-trainer.sh",
           "--run", run, "--init", init,
           "--cycles", str(k), "--hours", "0",
           "--games", str(p["games"]), "--sims", str(p["sims"]),
           "--steps", str(p["steps"]), "--seed", str(p["seed"])]
    if p["nesseby"]:
        cmd += ["--nesseby", "--nesseby-fraction", str(p["nesseby_frac"])]
    if p["opponent"]:
        cmd += ["--train-opponent", p["opponent"]]
    if p["hand_style"]:
        cmd += ["--hand-style", p["hand_style"]]
    if p["aggressive"]:
        cmd += ["--aggressive"]
    if p["styles"]:
        cmd += ["--styles"]
    if workers:
        cmd += ["--threads", str(workers)]
    if p["dry"]:
        cmd += ["--dry"]
    return cmd

def gpu_wait(mb: int, log_path: pathlib.Path):
    say = lambda m: _log(m, log_path)
    tries = 0
    while True:
        f = free_vram_mb()
        if f >= mb:
            if tries:
                say(f"GPU free: {f} MiB")
            return
        tries += 1
        if tries % 12 == 1:
            say(f"waiting for GPU: {f} MiB free, need {mb} "
                f"(strata? stop: systemctl --user stop strata-swift)")
        time.sleep(20)

# ── the runner: one session, cycle by cycle ───────────────────────────────
def run_session(p: dict):
    run, init = p["run"], resolve_init(p["init"])
    p["init"] = init
    os.chdir(ROOT)
    (ROOT / "runs").mkdir(exist_ok=True)
    log_path = ROOT / f"runs/{run}_run.log"
    say = lambda m: _log(m, log_path)

    params_file = write_params_file(run, p)
    say(f"params: {params_file.relative_to(ROOT)} "
        f"(restart: {PY.relative_to(ROOT)} tools/train.py --go "
        f"--params runs/{run}_params.json)")

    say(f"=== session start: run={run} · {p['games']} games/cycle · "
        f"cycles ..{p['cycles']} · push every {p['push_every']} · "
        f"bench {p['bench_list'] or 'none'} ({p.get('bench_when', 'cycle')}) ===")

    # at most three benchmarks — more is wasted box-hours (MAX_BENCH)
    p["bench_list"] = clamp_bench(p["bench_list"])
    bench_spec = p["bench_list"]
    bench_each = bool(bench_spec) and p.get("bench_when", "cycle") == "cycle"

    # the Nox gate reads the bench spec from runs/<run>_bench.txt (committed
    # with each cycle) — so the publish bench plays exactly these opponents
    spec_file = ROOT / f"runs/{run}_bench.txt"
    if bench_spec:
        spec_file.write_text(p["bench_list"].strip() + "\n")
    elif spec_file.exists():
        spec_file.unlink()

    # pre-flight: everything that can break, break NOW (seconds, not hours)
    if not PY.exists():
        raise SystemExit("no .venv — run from the sahkku-ai repo root")
    if not pathlib.Path(init).exists():
        raise SystemExit(f"init weight not found: {init}")
    br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if br != "main":
        raise SystemExit(f"on branch '{br}', not main — the gate reads origin/main")
    if not p["dry"] and not p["no_push"]:
        if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT,
                          capture_output=True).returncode != 0:
            raise SystemExit("worktree dirty — commit/stash first, the session "
                             "would mix your edits into engine commits")
        if subprocess.run(["git", "ls-remote", "--exit-code", "origin"],
                          cwd=ROOT, capture_output=True).returncode != 0:
            raise SystemExit("cannot reach origin — check key/net NOW")
        subprocess.run(["git", "pull", "-q", "--rebase", "--autostash",
                        "origin", "main"], cwd=ROOT, capture_output=True)

    # code guard: the box must run current features/punisher (train-push.sh)
    g = subprocess.run(
        [str(PY), "-c",
         "from sahkku_ai import features as FT; import sahkku_ai.punisher; "
         "raise SystemExit(0 if FT.N_GLOBAL >= 18 else 1)"],
        cwd=ROOT, capture_output=True)
    if g.returncode != 0:
        raise SystemExit("box runs OLD code (missing punisher / g[16]/g[17]) — "
                         "git pull first")

    # GPU selector: pin every child (trainer, export, bench) to the chosen
    # card via CUDA_VISIBLE_DEVICES — 'cpu' hides CUDA entirely, an index
    # picks that nvidia-smi card as the only visible device.
    dev, cvd = resolve_device(p.get("gpu", "auto"))
    if cvd is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        say(f"GPU selector: {p.get('gpu', 'auto')} → CUDA_VISIBLE_DEVICES="
            f"{cvd!r}")
    using_gpu = not dev.startswith("CPU")
    if p.get("gpu", "auto") not in ("auto", "", None) and dev.startswith("CPU"):
        say(f"note: GPU '{p.get('gpu')}' not usable — running CPU")
    workers = p["threads"]
    if using_gpu:
        try:
            import torch
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            maxw = max(2, int((total_gb * 0.8 - 1.0) / 0.45))
            auto = workers or n_cores()
            workers = min(auto, maxw)
            if auto > maxw:
                say(f"VRAM {total_gb:.1f} GB → workers {auto} → {workers} "
                    f"(each worker holds a CUDA context)")
        except Exception:
            workers = workers or n_cores()
    else:
        workers = workers or n_cores()
    say(f"device: {dev} · workers: {workers} · cores: {n_cores()}")

    done = last_cycle(run) if p["resume"] else 0
    if done:
        say(f"resuming at cycle {done + 1} ({done} completed)")
    if done >= p["cycles"]:
        say(f"cycles 1..{p['cycles']} already done — nothing to do")
        return

    k = done + 1
    t_start = time.time()
    fails = 0
    prog = None
    if sys.stdout.isatty():
        prog = Progress(run, p["cycles"], log_path)
        prog.set_games_per_cycle(p["games"])

    def bench_now(k):
        """Run the bench for cycle k (progress callback feeds the bar)."""
        label = ",".join(o for o, _ in parse_bench_spec(bench_spec))
        pr = prog                                   # narrowed for the closure
        if pr:
            pr.set_bench(0.0, label)
        res = bench_cycle(run, k, bench_spec, p["bench_seed"], log_path,
                          progress=(lambda f: pr.set_bench(f, label))
                          if pr else None)
        if pr:
            pr.set_bench(1.0, label)
        return res

    while k <= p["cycles"]:
        if p["hours"] > 0 and time.time() - t_start >= p["hours"] * 3600:
            say(f"time budget {p['hours']}h used — stopping after c{k - 1}")
            break
        say(f"-- cycle {k} of {p['cycles']} " + "-" * 30)
        if using_gpu:
            gpu_wait(p["gpu_wait_mb"], log_path)

        cmd = cycle_cmd(run, k, init, p, workers)
        say("$ " + " ".join(shlex.quote(x) for x in cmd))
        with open(log_path, "a") as lf:
            r = subprocess.run(cmd, cwd=ROOT, stdout=lf,
                               stderr=subprocess.STDOUT, text=True)
        if r.returncode != 0:
            fails += 1
            tail = open(log_path).read().strip().splitlines()[-5:]
            say(f"cycle {k} FAILED ({fails} in a row):")
            for l in tail:
                say(f"  | {l}")
            if fails >= 5:
                say("5 consecutive failures — stopping")
                break
            time.sleep(min(fails, 5) * 10)
            continue
        fails = 0
        if p["dry"]:
            say(f"(dry) would export c{k}, gate, bench, push")
            k += 1
            continue

        export_and_gate(run, k, log_path)          # export + parity + S-UCI
        if bench_each:
            bench_now(k)
        write_cycle_row(run, k, init, p["opponent"], p["hand_style"], log_path)
        commit_and_push(run, k, p["cycles"], p["push_every"],
                        init, p["opponent"], p["hand_style"],
                        p["no_push"], log_path)
        k += 1

    # bench_when='end': one bench on the last exported engine, after the last
    # cycle — the cheap option when the box should spend its hours training.
    if bench_spec and not bench_each and not p["dry"] and \
            (ROOT / f"engines/{run}-nn.js").exists():
        say(f"--- final bench (after c{k - 1}) " + "-" * 24)
        bench_now(k - 1)

    if prog:
        prog.stop()
    say(f"=== session done: through cycle {k - 1} of {p['cycles']} ===")
    say(f"report:  runs/{run}_report.md")
    say(f"watch:   {PY.relative_to(ROOT)} tools/train_status.py {run} -w")

# ── curses TUI ────────────────────────────────────────────────────────────
FIELDS = [
    # (key, label, type, choices)   'dynlist' = init_choices() at runtime
    ("run",          "Run name",              "text",    None),
    ("init",         "Init weight (←→ list)", "dynlist", None),
    ("gpu",          "GPU (←→ list)",         "gpulist", None),
    ("cycles",       "End cycle",             "int",     None),
    ("hours",        "Hours (0=no limit)",    "float",   None),
    ("games",        "Games per cycle",       "int",     None),
    ("sims",         "Sims per play",         "int",     None),
    ("steps",        "Train steps",           "int",     None),
    ("threads",      "Workers (0=auto)",      "int",     None),
    ("seed",         "Seed",                  "int",     None),
    ("opponent",     "Opponent",              "choice",
     ["", "punisher", "classic3"]),
    ("hand_style",   "Hand style",            "choice",
     ["", "hold", "few", "many", "kingfwd", "kingguard"]),
    ("nesseby",      "Nesseby rules",         "bool",    None),
    ("nesseby_frac", "Nesseby fraction",      "float",   None),
    ("aggressive",   "Aggressive side",       "bool",    None),
    ("styles",       "Style duels",           "bool",    None),
    ("push_every",   "Push every N cycles",   "int",     None),
    ("bench_list",   "Bench list (b: pick)",  "bench",   None),
    ("bench_when",   "Bench when",            "choice",  ["cycle", "end"]),
    ("bench_seed",   "Bench seed",            "int",     None),
    ("gpu_wait_mb",  "GPU wait MiB",          "int",     None),
    ("no_push",      "No push (local)",       "bool",    None),
    ("dry",          "Dry run",               "bool",    None),
]
ACTIONS = [
    ("start",    " START "),
    ("preset",   " load preset "),
    ("save",     " save preset "),
    ("defaults", " defaults "),
    ("quit",     " quit "),
]

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z=]|\x1b\][^\x07]*\x07")

def strip_ansi(s: str) -> str:
    return ANSI.sub("", s)

def gpu_choices() -> list:
    return ["auto", "cpu"] + [g["index"] for g in gpu_list()]

def gpu_shown(v) -> str:
    v = str(v if v not in (None, "") else "auto").strip().lower()
    if v in ("auto",):
        return f"auto → {gpu_name()}"
    if v in ("cpu", "none", "off"):
        return "CPU only (no GPU)"
    if v.isdigit():
        return gpu_label(v)
    return v

def parse_run_log(log_path: pathlib.Path, gpc_hint: int = 0) -> tuple:
    """(done, cur, now, games_per_cycle) from the session log — the exact
    lines the pipeline writes (same view as train_status.py)."""
    done, cur, now, gpc = 0, 1, 0, (gpc_hint or DEFAULTS["games"])
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                m = re.match(r"^\[c(\d+)\] self-play: (\d+) parti", line)
                if m:
                    cur, gpc = int(m.group(1)), int(m.group(2))
                    now = 0
                    continue
                m = re.match(r"^\[c(\d+)\] ferdig", line)
                if m:
                    done = max(done, int(m.group(1)))
                    now = gpc
                    continue
                m = re.match(r"^\[w\d+\] (\d+)/(\d+) parti", line)
                if m:
                    now = max(now, int(m.group(1)))
    except OSError:
        pass
    return done, cur, now, gpc

def bench_state(log_path: pathlib.Path) -> tuple:
    """(frac, label) of a running bench from the log: the last 'BENCH opp (N
    games' line opens a match; streamed 'parti' lines fill the bar; a
    'RESULT' closes it."""
    label, total, seen, open_ = "", 0, 0, False
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                m = re.search(r"BENCH \S+ vs (\S+) \((\d+) games", line)
                if m:
                    label, total, seen, open_ = m.group(1), int(m.group(2)), 0, True
                    continue
                if line.startswith("SKIP "):
                    open_ = False
                    continue
                if open_ and line.startswith("parti "):
                    seen += 1
                if open_ and line.startswith("RESULT "):
                    open_ = False
    except OSError:
        pass
    if not open_ and not label:
        return (0.0, "")
    if not open_:
        return (1.0, label)
    return (min(1.0, seen / max(1, total)), label)

def log_tail(log_path: pathlib.Path, n: int) -> list:
    """The last n readable lines of the session log, ANSI stripped."""
    try:
        with open(log_path, "rb") as f:
            try:
                f.seek(-16000, 2)
            except OSError:
                f.seek(0)
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    lines = [strip_ansi(l.rstrip()) for l in data.splitlines()]
    return [l for l in lines if l][-n:]

def bar(frac: float, width: int) -> str:
    n = max(0, min(width, int(round(frac * width))))
    return "█" * n + "░" * (width - n)

def tui(stdscr):
    curses.curs_set(0)
    stdscr.keypad(True)
    if curses.has_colors():
        def pair(n, fg):
            try:
                curses.init_extended_pair(n, fg, -1)
            except Exception:
                try:
                    curses.init_pair(n, fg, curses.COLOR_BLACK)
                except Exception:
                    pass
        pair(1, curses.COLOR_CYAN)      # header
        pair(2, curses.COLOR_GREEN)     # start / running
        pair(3, curses.COLOR_YELLOW)    # warnings
        pair(4, curses.COLOR_MAGENTA)   # values
        pair(5, curses.COLOR_RED)       # stop / danger
    CP = lambda n: (curses.color_pair(n) if curses.has_colors() else 0)
    A_BOLD = curses.A_BOLD
    A_DIM = curses.A_DIM
    A_REV = curses.A_REVERSE

    vals = dict(DEFAULTS)   # carries keys the form has no field for (resume)
    sel = 0
    status = ""
    n_f, n_a = len(FIELDS), len(ACTIONS)

    def safe(y, x, s, attr=0):
        try:
            stdscr.addstr(y, x, s, attr)
        except curses.error:
            pass            # clipped at panel edge / bottom-right corner

    def panel(y0, y1, title="", attr=0):
        """Rounded-ish box from row y0 to y1, full width minus margins."""
        h, w = stdscr.getmaxyx()
        x1 = w - 2
        head = "┌" + (f" {title} " if title else "")
        safe(y0, 1, (head + "─" * max(0, x1 - len(head) - 1) + "┐")[:x1], attr)
        for y in range(y0 + 1, y1):
            safe(y, 1, "│", attr)
        safe(y1, 1, "└" + "─" * (x1 - 3) + "┘", attr)

    # ── modal helpers ─────────────────────────────────────────────────────
    def ask(prompt: str) -> str:
        h, w = stdscr.getmaxyx()
        row = min(h - 2, 3 + len(FIELDS) + 3)
        curses.echo()
        curses.curs_set(1)
        safe(row, 2, " " * (w - 3))
        safe(row, 2, prompt, A_BOLD)
        stdscr.move(row, 2 + len(prompt))
        raw = b""
        try:
            raw = stdscr.getstr(row, 2 + len(prompt), 40)
        except Exception:
            pass
        curses.noecho()
        curses.curs_set(0)
        return raw.decode("utf-8", "replace").strip()

    def warn_overlay(issues: list) -> bool:
        """Compatibility gate: list the issues, y = start anyway, else back."""
        h, w = stdscr.getmaxyx()
        wrapped = []
        for s in issues:
            while len(s) > w - 8:
                cut = s.rfind(" ", 0, w - 8)
                cut = w - 8 if cut < 1 else cut
                wrapped.append("! " + s[:cut].strip())
                s = s[cut:].strip()
            wrapped.append("! " + s)
        wrapped = wrapped[:max(1, h - 10)]
        y0 = max(2, (h - len(wrapped) - 6) // 2)
        stdscr.erase()
        panel(y0, y0 + len(wrapped) + 4, "COMPATIBILITY", CP(3) | A_BOLD)
        for i, s in enumerate(wrapped):
            safe(y0 + 2 + i, 3, s[:w - 5], CP(3))
        row = y0 + len(wrapped) + 2
        safe(row, 3, "These options conflict. The session will run, parts of", CP(3) | A_DIM)
        safe(row + 1, 3, "it may be pointless — fix them, or proceed.", CP(3) | A_DIM)
        safe(row + 2, 3, "press y to START anyway · any other key to go back", A_BOLD)
        stdscr.refresh()
        return stdscr.getch() in (ord("y"), ord("Y"))

    def bench_picker():
        """Modal bench picker: the POPULATED opponent list (opp_choices),
        space toggles a pick, ←/→ step the game count, max MAX_BENCH."""
        nonlocal status
        cursor = 0
        picked = dict(parse_bench_spec(vals["bench_list"]))
        while True:
            chs = opp_choices(vals.get("run", ""))
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            panel(0, h - 2, "BENCH OPPONENTS", CP(1) | A_BOLD)
            safe(2, 3, "space: pick (max " + str(MAX_BENCH) +
                 ")   ←/→: games   enter: apply   esc: cancel", A_DIM)
            rows = max(1, h - 7)
            top = max(0, min(cursor - rows + 1, len(chs) - rows))
            for i in range(rows):
                j = top + i
                if j >= len(chs):
                    break
                opp = chs[j]
                g = picked.get(opp)
                line = (f" [{'x' if g is not None else ' '}] {opp:<12}"
                        f" {g if g is not None else '-':>4} games")
                safe(4 + i, 3, line[:w - 5],
                     A_REV if j == cursor else curses.A_NORMAL)
            spec = ",".join(f"{o}:{g}" for o, g in picked.items())
            safe(h - 4, 3, f" picked {len(picked)}/{MAX_BENCH}:"
                 f" {spec or '(none)'}"[:w - 5], A_DIM)
            stdscr.refresh()
            ch = stdscr.getch()
            if ch in (27, ord("q")):
                return
            if ch in (curses.KEY_UP, ord("k")):
                cursor = (cursor - 1) % len(chs)
            elif ch in (curses.KEY_DOWN, ord("j")):
                cursor = (cursor + 1) % len(chs)
            elif ch == ord(" ") and chs:
                opp = chs[cursor]
                if opp in picked:
                    del picked[opp]
                elif len(picked) < MAX_BENCH:
                    picked[opp] = 30
            elif ch in (curses.KEY_LEFT, curses.KEY_RIGHT) and chs:
                opp = chs[cursor]
                if opp not in picked:
                    if len(picked) >= MAX_BENCH:
                        status = (f"bench: {MAX_BENCH} picks — unselect "
                                  "one first (space)")
                        continue
                    picked[opp] = 10 if ch == curses.KEY_RIGHT else 0
                else:
                    step = 10 if ch == curses.KEY_RIGHT else -10
                    picked[opp] = max(0, min(100, picked[opp] + step))
                    if picked[opp] == 0:
                        del picked[opp]
            elif ch in (10, curses.KEY_ENTER):
                order = [(o, picked[o]) for o in chs if o in picked]
                order += [(o, g) for o, g in picked.items() if o not in chs]
                vals["bench_list"] = ",".join(
                    f"{o}:{g}" for o, g in order[:MAX_BENCH])
                return

    # ── value rendering ───────────────────────────────────────────────────
    def shown_val(key, ftype):
        v = vals[key]
        if ftype == "bool":
            return "yes" if v else "no"
        if ftype == "gpulist":
            return gpu_shown(v)
        if ftype == "bench":
            parts = parse_bench_spec(v)
            if not parts:
                return "(no bench)"
            spec = ",".join(f"{o}:{g}" for o, g in parts[:MAX_BENCH])
            return f"{spec}  [{vals.get('bench_when', 'cycle')}]"
        if ftype == "dynlist" and key == "init":
            if not v:
                ch = init_choices()
                return f"(newest: {pathlib.Path(ch[0]).name})" if ch else "(none)"
            return pathlib.Path(v).name
        return v if v != "" else "(none)"

    # ── live dashboard: the session runs, the TUI STAYS ───────────────────
    def dashboard(proc, log_path: pathlib.Path, p: dict):
        """Draw the running session IN the TUI: progress bars for cycles and
        the bench, log tail, and STOP. The session is a child process; the
        TUI is a dashboard around it, not a terminal emulator."""
        nonlocal status
        stdscr.timeout(1500)          # redraw every 1.5 s even with no input
        stop_ask = False
        cycles = p["cycles"]
        done_note = ""
        while True:
            rc = proc.poll()
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            title = (f"RUNNING · {p['run']} · c..{cycles}"
                     if rc is None else
                     f"SESSION ENDED · {p['run']} · exit {rc}")
            safe(0, 2, title, (CP(2) if rc is None else CP(1)) | A_BOLD)
            panel(2, 7, "PROGRESS", CP(1))
            done, cur, now, gpc = parse_run_log(log_path, p["games"])
            frac = (done * gpc + min(now, gpc)) / max(1, cycles * gpc)
            bw = max(10, min(40, w - 36))
            safe(4, 3, f" cycles  [{bar(frac, bw)}] "
                 f"c{min(cur, cycles)}/{cycles}  {frac * 100:5.1f} %"[:w - 5])
            bfrac, blabel = bench_state(log_path)
            if blabel:
                safe(5, 3, f" bench   [{bar(bfrac, bw)}] "
                     f"{blabel}  {bfrac * 100:5.1f} %"[:w - 5], CP(4))
            else:
                safe(5, 3, " bench   — not running", A_DIM)
            rows = max(4, h - 12)
            panel(8, 9 + rows, "LOG", CP(1))
            for i, line in enumerate(log_tail(log_path, rows - 1)):
                safe(10 + i, 3, " " + line[:w - 6],
                     CP(5) if "FAIL" in line else curses.A_NORMAL)
            keyrow = min(h - 2, 11 + rows)
            if rc is None:
                if stop_ask:
                    safe(keyrow, 3, "STOP the session? press q again to "
                         "kill it (esc = keep running)", CP(5) | A_BOLD)
                else:
                    safe(keyrow, 3, " q: stop session   (the TUI stays — "
                         "this is the live dashboard)", A_DIM)
            else:
                safe(keyrow, 3, " enter: back to the form   q: quit   "
                     f"watch: tools/train_status.py {p['run']} -w",
                     CP(2) | A_BOLD)
            stdscr.refresh()

            ch = stdscr.getch()
            if ch == -1:
                continue
            if rc is not None:
                if ch in (curses.KEY_F10, 27, ord("q")):
                    stdscr.timeout(-1)
                    return "quit"
                if ch in (10, curses.KEY_ENTER, curses.KEY_F5, ord("n")):
                    stdscr.timeout(-1)
                    done_note = "session ended — back in the form"
                    return "form"
                continue
            if ch in (curses.KEY_F10, 27):
                if stop_ask:
                    stop_ask = False          # esc on the confirm = keep running
                # esc/q on the plain bar: ask first (stopping is dangerous)
                stop_ask = True
            elif ch == ord("q"):
                if stop_ask:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        for _ in range(10):
                            if proc.poll() is not None:
                                break
                            time.sleep(0.5)
                        if proc.poll() is None:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    stdscr.timeout(-1)
                    status = "session stopped — back in the form"
                    return "form"
                stop_ask = True

    def start_session():
        """The START button. The TUI does NOT leave: run_session starts as a
        child process (--go, same params file), the TUI becomes its
        dashboard."""
        nonlocal status
        p = dict(vals)
        issues = compat_issues(p)
        if issues and not warn_overlay(issues):
            status = "start aborted — fix the compatibility warnings first"
            return
        path = write_params_file(p["run"], p)
        log_path = ROOT / f"runs/{p['run']}_run.log"
        lf = open(log_path, "a")
        proc = subprocess.Popen(
            [str(PY), "tools/train.py", "--go", "--params", str(path)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=lf,
            stderr=subprocess.STDOUT, start_new_session=True)
        out = dashboard(proc, log_path, p)
        lf.close()
        if out == "quit":
            raise SystemExit(0)

    # ── main loop ─────────────────────────────────────────────────────────
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        safe(0, 2, " SAHKKU TRAINING SESSION ", CP(1) | A_BOLD)
        info = f"{gpu_shown(vals['gpu'])} · {n_cores()} cores"
        safe(0, max(2, w - len(info) - 3), info, CP(1) | A_DIM)

        issues = compat_issues(vals)
        plan_rows = 3 + min(3, len(issues)) + 1
        actions_y = h - 3
        fields_y0 = 3
        fields_y1 = max(fields_y0 + 3, actions_y - plan_rows - 2)
        panel(fields_y0, fields_y1, "CONFIG", CP(1))
        rows = max(1, fields_y1 - fields_y0 - 1)
        f_sel = min(sel, n_f - 1)
        top = max(0, min(f_sel - rows + 1, n_f - rows)) if sel < n_f else max(0, n_f - rows)
        col_val = 37
        for i in range(rows):
            j = top + i
            if j >= n_f:
                break
            key, label, ftype, _ch = FIELDS[j]
            shown = shown_val(key, ftype)
            line = f" {label:<32}{shown}"
            safe(fields_y0 + 1 + i, 2, line[:w - 4],
                 A_REV if j == sel else curses.A_NORMAL)

        done = last_cycle(vals["run"]) if vals["run"] else 0
        cy = est_cycle_min(vals["games"], vals["sims"],
                           vals["threads"] or n_cores())
        n_left = max(0, vals["cycles"] - done)
        total = n_left * cy
        py = fields_y1 + 1
        panel(py, py + plan_rows, "PLAN", CP(1))
        safe(py + 1, 3, (f" c{done + 1}..{vals['cycles']} · {n_left} cycles "
                         f"~{cy} min ≈ {total / 60:.1f} h · push every "
                         f"{vals['push_every']} · bench: "
                         f"{vals['bench_list'] or 'none'} "
                         f"[{vals.get('bench_when', 'cycle')}]"
                         )[:w - 5], A_DIM)
        for i, s in enumerate(issues[:3]):
            safe(py + 2 + i, 3, (f" ! {s}")[:w - 5], CP(3))
        if len(issues) > 3:
            safe(py + 5, 3, f" ! (+{len(issues) - 3} more)", CP(3))

        safe(actions_y - 1, 2, status[:w - 4], A_BOLD)
        # action bar: the START button and friends
        x = 2
        for i, (_a, label) in enumerate(ACTIONS):
            a = len(label) + 4
            if x + a > w - 2:
                break
            tag = f" [{label}] "
            attr = (CP(2) | A_REV | A_BOLD) if sel == n_f + i else CP(2)
            safe(actions_y, x, tag, attr)
            x += a
        safe(actions_y, x, "  F5=start · b=bench · p/s=preset · Del=default",
             A_DIM)
        stdscr.refresh()

        ch = stdscr.getch()
        key, label, ftype, choices = (FIELDS[sel] if sel < n_f
                                      else (None, None, "action", None))
        status = ""
        if ch in (curses.KEY_F10,):
            return None
        if sel >= n_f and ch in (10, curses.KEY_ENTER, ord(" ")):
            act = ACTIONS[sel - n_f][0]
            if act == "start":
                start_session()
            elif act == "quit":
                return None
            elif act == "defaults":
                vals = dict(DEFAULTS)
                status = "all fields reset to defaults"
            elif act == "preset":
                pres = load_presets()
                if pres:
                    name = ask("preset: " + ", ".join(sorted(pres)) + " › ")
                    if name in pres:
                        vals.update({k: pres[name][k] for k in DEFAULTS
                                     if k in pres[name]})
                        status = f"loaded preset «{name}»"
                    elif name:
                        status = f"no preset «{name}»"
                else:
                    status = "no presets saved (save preset →)"
            elif act == "save":
                name = ask("save preset as: ")
                if name:
                    save_preset(name, vals)
                    status = f"preset «{name}» saved to tools/train_presets.json"
            continue
        if ch == ord("q") and sel < n_f:
            return None
        if ch in (curses.KEY_UP, ord("k")):
            sel = (sel - 1) % (n_f + n_a)
        elif ch in (curses.KEY_DOWN, ord("j")):
            sel = (sel + 1) % (n_f + n_a)
        elif ch == curses.KEY_HOME:
            sel = 0
        elif ch == curses.KEY_END:
            sel = n_f            # the START button
        elif ch in (curses.KEY_F5, 10, curses.KEY_ENTER) and sel < n_f:
            start_session()
        elif ch == ord("b") and sel < n_f:
            bench_picker()
            status = f"bench: {vals['bench_list'] or '(none)'}"
        elif ch == ord("p") and sel < n_f:
            pres = load_presets()
            if pres:
                name = ask("preset: " + ", ".join(sorted(pres)) + " › ")
                if name in pres:
                    vals.update({k: pres[name][k] for k in DEFAULTS
                                 if k in pres[name]})
                    status = f"loaded preset «{name}»"
                elif name:
                    status = f"no preset «{name}»"
            else:
                status = "no presets saved (s to save one)"
        elif ch == ord("s") and sel < n_f:
            name = ask("save preset as: ")
            if name:
                save_preset(name, vals)
                status = f"preset «{name}» saved to tools/train_presets.json"
        elif ch == curses.KEY_DC:
            if sel < n_f:
                vals[key] = DEFAULTS[key]
                status = f"{label}: back to default"
        elif ch in (curses.KEY_LEFT, curses.KEY_RIGHT, ord(" ")) and sel < n_f:
            if ftype == "bool":
                vals[key] = not vals[key]
            elif ftype == "dynlist":
                chs = init_choices()
                if chs:
                    cur = chs.index(vals[key]) if vals[key] in chs else -1
                    step = 1 if ch == curses.KEY_RIGHT else -1
                    vals[key] = chs[(cur + step) % len(chs)]
            elif ftype == "gpulist":
                chs = gpu_choices()
                cur = chs.index(str(vals[key])) if str(vals[key]) in chs else -1
                step = 1 if ch == curses.KEY_RIGHT else -1
                vals[key] = chs[(cur + step) % len(chs)]
            elif ftype == "choice" and choices:
                cur = choices.index(vals[key]) if vals[key] in choices else 0
                step = 1 if ch == curses.KEY_RIGHT else -1
                vals[key] = choices[(cur + step) % len(choices)]
            elif ftype == "bench":
                bench_picker()
                status = f"bench: {vals['bench_list'] or '(none)'}"
            else:
                status = f"{label}: type a value to edit"
        elif ftype in ("text", "int", "float") and sel < n_f:
            curses.echo()
            curses.curs_set(1)
            row = fields_y0 + 1 + (sel - top)
            stdscr.addstr(row, col_val, " " * (w - col_val - 2))
            stdscr.addstr(row, col_val, str(vals[key]))
            stdscr.move(row, col_val + len(str(vals[key])))
            raw = b""
            try:
                raw = stdscr.getstr(row, col_val, w - col_val - 2)
            except Exception:
                pass
            curses.noecho()
            curses.curs_set(0)
            s = raw.decode("utf-8", "replace").strip()
            if s:
                try:
                    if ftype == "int":
                        vals[key] = int(s)
                    elif ftype == "float":
                        vals[key] = float(s)
                    else:
                        vals[key] = s
                except ValueError:
                    status = f"{label}: «{s}» is not a valid {ftype} value"
        elif 32 <= ch < 127 and sel < n_f:
            status = f"{label}: press a key on a text/number field, ←→ on a toggle"


# ── headless CLI ──────────────────────────────────────────────────────────
def build_parser():
    ap = argparse.ArgumentParser(
        description="Unified sáhkku trainer: run with no args for the TUI, "
                    "with --go for a headless session.")
    ap.add_argument("--go", action="store_true", help="headless: run, no TUI")
    ap.add_argument("--preset", metavar="NAME",
                    help="start from a saved preset (tools/train_presets.json); "
                         "flags after it override")
    ap.add_argument("--save-preset", dest="save_preset", metavar="NAME",
                    help="write the resulting parameters (presets + flags) "
                         "to the preset NAME and exit")
    ap.add_argument("--params", metavar="FILE",
                    help="read parameters from runs/<run>_params.json "
                         "(written by every session); flags override")
    ap.add_argument("--list-weights", action="store_true",
                    help="print the selectable init weights, newest first, "
                         "and exit")
    ap.add_argument("--list-opponents", action="store_true",
                    help="print the selectable bench opponents, one per line, "
                         "and exit")
    ap.add_argument("--run", default=DEFAULTS["run"])
    ap.add_argument("--init", default=DEFAULTS["init"])
    ap.add_argument("--cycles", type=int, default=DEFAULTS["cycles"],
                    help="END cycle number; the run resumes where it stopped")
    ap.add_argument("--hours", type=float, default=DEFAULTS["hours"],
                    help="wall-clock budget, checked between cycles "
                         "(0 = no limit)")
    ap.add_argument("--games", type=int, default=DEFAULTS["games"])
    ap.add_argument("--sims", type=int, default=DEFAULTS["sims"])
    ap.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    ap.add_argument("--threads", type=int, default=DEFAULTS["threads"],
                    help="workers (0 = auto: physical cores, VRAM-capped)")
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--opponent", default=DEFAULTS["opponent"],
                    choices=["", "punisher", "classic3"])
    ap.add_argument("--hand-style", dest="hand_style",
                    default=DEFAULTS["hand_style"],
                    choices=["", "hold", "few", "many", "kingfwd", "kingguard"])
    ap.add_argument("--nesseby", action="store_true", default=DEFAULTS["nesseby"])
    ap.add_argument("--no-nesseby", dest="nesseby", action="store_false")
    ap.add_argument("--nesseby-fraction", dest="nesseby_frac",
                    type=float, default=DEFAULTS["nesseby_frac"])
    ap.add_argument("--aggressive", action="store_true",
                    default=DEFAULTS["aggressive"])
    ap.add_argument("--styles", action="store_true", default=DEFAULTS["styles"])
    ap.add_argument("--push-every", dest="push_every", type=int,
                    default=DEFAULTS["push_every"],
                    help="push every N-th cycle (0 = only the last)")
    ap.add_argument("--gpu", default=DEFAULTS["gpu"],
                    help='"auto", "cpu", or a nvidia-smi GPU index '
                         '(see --list-gpus) — pins CUDA_VISIBLE_DEVICES')
    ap.add_argument("--list-gpus", dest="list_gpus", action="store_true",
                    help="print the selectable GPUs, one per line, and exit")
    ap.add_argument("--bench-list", dest="bench_list",
                    default=DEFAULTS["bench_list"],
                    help='bench spec "opp:games,opp:games" — pick from the '
                         'populated list (--list-opponents: classic3, base, '
                         'every exported engine id); "0" = no local bench. '
                         "Max three matches. The same list is handed to the "
                         "Nox gate.")
    ap.add_argument("--bench-when", dest="bench_when",
                    choices=["cycle", "end"], default=DEFAULTS["bench_when"],
                    help='"cycle" = bench after each training cycle, '
                         '"end" = once, after the last cycle of the session')
    ap.add_argument("--bench-seed", dest="bench_seed", type=int,
                    default=DEFAULTS["bench_seed"])
    ap.add_argument("--gpu-wait-mb", dest="gpu_wait_mb", type=int,
                    default=DEFAULTS["gpu_wait_mb"])
    ap.add_argument("--no-push", dest="no_push", action="store_true",
                    default=DEFAULTS["no_push"])
    ap.add_argument("--dry", action="store_true", default=DEFAULTS["dry"])
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    default=DEFAULTS["resume"])
    return ap

def headless_command(p: dict) -> str:
    parts = ["./tools/train.py --go"]
    flags = {"run": "--run", "init": "--init", "cycles": "--cycles",
             "hours": "--hours", "games": "--games", "sims": "--sims",
             "steps": "--steps", "threads": "--threads", "seed": "--seed",
             "opponent": "--opponent", "hand_style": "--hand-style",
             "nesseby_frac": "--nesseby-fraction",
             "push_every": "--push-every", "bench_list": "--bench-list",
             "gpu": "--gpu",
             "bench_when": "--bench-when",
             "bench_seed": "--bench-seed",
             "gpu_wait_mb": "--gpu-wait-mb"}
    for key, flag in flags.items():
        if p.get(key, DEFAULTS[key]) != DEFAULTS[key]:
            parts.append(f"{flag} {shlex.quote(str(p[key]))}")
    for key, on, off in (("nesseby", None, "--no-nesseby"),
                         ("aggressive", "--aggressive", None),
                         ("styles", "--styles", None),
                         ("no_push", "--no-push", None),
                         ("dry", "--dry", None),
                         ("resume", None, "--no-resume")):
        if p.get(key, DEFAULTS[key]) and on:
            parts.append(on)
        if not p.get(key, DEFAULTS[key]) and off:
            parts.append(off)
    return " \\\n    ".join(parts)

def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.list_weights:
        for i, w in enumerate(init_choices()):
            print(f"[{i}] {w}")
        return

    if args.list_opponents:
        for i, o in enumerate(opp_choices(args.run)):
            print(f"[{i}] {o}")
        return

    if args.list_gpus:
        print("[auto]")
        print("[cpu]")
        for g in gpu_list():
            print(f"[{g['index']}] {g['name']} ({g['free']}/{g['total']} MiB free)")
        return

    # parameter precedence: DEFAULTS < --params file < --preset < explicit flags
    p = dict(DEFAULTS)
    if args.params:
        try:
            saved = json.loads(pathlib.Path(args.params).read_text())
        except Exception as e:
            raise SystemExit(f"--params: cannot read {args.params}: {e}")
        p.update({k: saved[k] for k in DEFAULTS if k in saved})
    if args.preset:
        load_preset(args.preset, p)
    for k in DEFAULTS:
        v = getattr(args, k, None)
        if v is not None and v != ap.get_default(k):
            p[k] = v

    if args.save_preset:
        save_preset(args.save_preset, p)
        print(f"preset «{args.save_preset}» saved to {PRESETS.relative_to(ROOT)}")
        print(json.dumps(p, indent=2, ensure_ascii=False))
        return

    if args.go:
        run_session(p)
        return
    if not sys.stdin.isatty():
        raise SystemExit("TUI needs a terminal — use --go with flags "
                         "for headless runs")
    result = curses.wrapper(tui)
    if result is None:
        print("quit — nothing started")
        return

if __name__ == "__main__":
    main()
