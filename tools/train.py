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
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = ROOT / ".venv/bin/python3"
PRESETS = ROOT / "tools" / "train_presets.json"

# ── defaults ──────────────────────────────────────────────────────────────
DEFAULTS: dict = {
    "run":          "blanka3",
    "init":         "engines/blanka2-nn.pt",  # '' = newest file (mtime); blanka
    #                                          # is the family trained now
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
    # Bench list: "opponent:games" comma-separated, run every cycle AND by
    # the Nox gate (sent to it in runs/<run>_bench.txt).
    #   base      = the current default engine (gate resolves it)
    #   classic3  = the classic CPU at level 3 (sahkku-ref)
    #   <id>      = any engine id (nais8, blanka2, kaiser1, …): the app's
    #               engines/<id>.js if the checkout is next door, else this
    #               repo's engines/<id>-nn.js (the box is self-sufficient)
    # Empty string / "0" = no local bench.
    "bench_list":   "base:30,classic3:40",
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
                log_path: pathlib.Path) -> list:
    """Run the local bench list; returns [(opp, games, pct)] parsed from
    bench_list.py RESULT lines. 'base' self-matches on the box: the freshly
    exported engine plays its own .js — same file, ~50 % — so 'base' is only
    meaningful with an explicit opponent id; keep the spec app-agnostic."""
    say = lambda m: _log(m, log_path)
    eng = ROOT / f"engines/{run}-nn.js"
    if not eng.exists():
        say(f"[c{k}] bench skipped (no {eng.name})")
        return []
    env = os.environ.copy()
    env["SAHKKU_APP"] = str(ROOT.parent / "SnartSahkku")  # if the app is here
    r = subprocess.run(
        [str(PY), "tools/bench_list.py", run, f"engines/{run}-nn.js",
         spec, str(seed_base + k)],
        cwd=ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    results = []
    for m in re.finditer(r"^RESULT (\S+) (\d+) ([\d.]+)", out, re.M):
        results.append((m.group(1), int(m.group(2)), float(m.group(3))))
    for line in out.splitlines():
        if line.startswith(("BENCH ", "SKIP ", "RESULT ")):
            say(f"[c{k}] {line}")
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
        f"bench {p['bench_list'] or 'none'} ===")

    # the Nox gate reads the bench spec from runs/<run>_bench.txt (committed
    # with each cycle) — so the publish bench plays exactly these opponents
    spec_file = ROOT / f"runs/{run}_bench.txt"
    if p["bench_list"] and p["bench_list"] != "0":
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

    dev = gpu_name()
    using_gpu = dev != "CPU"
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
        if p["bench_list"] and p["bench_list"] != "0":
            bench_cycle(run, k, p["bench_list"],
                        p["bench_seed"], log_path)
        write_cycle_row(run, k, init, p["opponent"], p["hand_style"], log_path)
        commit_and_push(run, k, p["cycles"], p["push_every"],
                        init, p["opponent"], p["hand_style"],
                        p["no_push"], log_path)
        k += 1

    say(f"=== session done: through cycle {k - 1} of {p['cycles']} ===")
    say(f"report:  runs/{run}_report.md")
    say(f"watch:   {PY.relative_to(ROOT)} tools/train_status.py {run} -w")

# ── curses TUI ────────────────────────────────────────────────────────────
FIELDS = [
    # (key, label, type, choices)   'dynlist' = init_choices() at runtime
    ("run",          "Run name",            "text",  None),
    ("init",         "Init weight (←→ list)", "dynlist", None),
    ("cycles",       "End cycle",           "int",   None),
    ("hours",        "Hours (0=no limit)",  "float", None),
    ("games",        "Games per cycle",     "int",   None),
    ("sims",         "Sims per play",       "int",   None),
    ("steps",        "Train steps",         "int",   None),
    ("threads",      "Workers (0=auto)",    "int",   None),
    ("seed",         "Seed",                "int",   None),
    ("opponent",     "Opponent",            "choice", ["", "punisher", "classic3"]),
    ("hand_style",   "Hand style",          "choice",
     ["", "hold", "few", "many", "kingfwd", "kingguard"]),
    ("nesseby",      "Nesseby rules",       "bool",  None),
    ("nesseby_frac", "Nesseby fraction",    "float", None),
    ("aggressive",   "Aggressive side",     "bool",  None),
    ("styles",       "Style duels",         "bool",  None),
    ("push_every",   "Push every N cycles", "int",   None),
    ("bench_list",   "Bench list",          "text",  None),
    ("bench_seed",   "Bench seed",          "int",   None),
    ("gpu_wait_mb",  "GPU wait MiB",        "int",   None),
    ("no_push",      "No push (local)",     "bool",  None),
    ("dry",          "Dry run",             "bool",  None),
]

def tui(stdscr):
    curses.curs_set(0)
    stdscr.keypad(True)
    vals = dict(DEFAULTS)   # carries keys the form has no field for (resume)
    sel = 0
    status = ""

    def ask(prompt: str) -> str:
        curses.echo()
        curses.curs_set(1)
        row = min(stdscr.getmaxyx()[0] - 1, 3 + len(FIELDS) + 3)
        stdscr.addstr(row, 2, " " * (stdscr.getmaxyx()[1] - 3))
        stdscr.addstr(row, 2, prompt)
        stdscr.move(row, 2 + len(prompt))
        raw = b""
        try:
            raw = stdscr.getstr(row, 2 + len(prompt), 40)
        except Exception:
            pass
        curses.noecho()
        curses.curs_set(0)
        return raw.decode("utf-8", "replace").strip()

    def shown_val(key, ftype):
        v = vals[key]
        if ftype == "bool":
            return "yes" if v else "no"
        if ftype == "dynlist" and key == "init":
            if not v:
                ch = init_choices()
                return f"(newest: {pathlib.Path(ch[0]).name})" if ch else "(none)"
            return pathlib.Path(v).name
        return v if v != "" else "(none)"
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        stdscr.addstr(0, 2, "SAHKKU TRAINING SESSION", curses.A_BOLD)
        stdscr.addstr(1, 2, "up/down: field   left/right/space: toggle/list   "
                            "type: edit   p: preset   s: save preset   "
                            "Del: default   F5: start   F10: quit",
                      curses.A_DIM)
        col_val, top = 34, 3
        for i, (key, label, ftype, _ch) in enumerate(FIELDS):
            if i + top > h - 5:
                break
            shown = shown_val(key, ftype)
            line = f"  {label:<30} {shown}"
            stdscr.addstr(i + top, 2, line[:w - 3],
                          curses.A_REVERSE if i == sel else curses.A_NORMAL)

        done = last_cycle(vals["run"]) if vals["run"] else 0
        cy = est_cycle_min(vals["games"], vals["sims"],
                           vals["threads"] or n_cores())
        n_left = max(0, vals["cycles"] - done)
        total = n_left * cy
        stdscr.addstr(min(h - 4, top + len(FIELDS) + 1), 2, (
            f"  plan: c{done + 1}..{vals['cycles']} = {n_left} cycles "
            f"~{cy} min each = ~{total / 60:.1f} h   GPU: {gpu_name()}   "
            f"push every {vals['push_every']}   bench: {vals['bench_list'] or 'none'}")[:w - 3], curses.A_DIM)
        if status:
            stdscr.addstr(min(h - 3, top + len(FIELDS) + 2), 2,
                          status[:w - 3], curses.A_BOLD)
        stdscr.addstr(min(h - 2, top + len(FIELDS) + 3), 2, (
            f"  resume: cycle {done + 1}   diet: {vals['opponent'] or 'self-play'}"
            f"{' + hand ' + vals['hand_style'] if vals['hand_style'] else ''}"
            f"   log: runs/{vals['run']}_run.log")[:w - 3], curses.A_DIM)
        stdscr.refresh()

        ch = stdscr.getch()
        key, label, ftype, choices = FIELDS[sel]
        status = ""
        if ch in (curses.KEY_F10, 27):
            return None
        if ch in (curses.KEY_F5, 10, curses.KEY_ENTER):
            return dict(vals)
        if ch in (curses.KEY_UP, ord("k")):
            sel = (sel - 1) % len(FIELDS)
        elif ch in (curses.KEY_DOWN, ord("j")):
            sel = (sel + 1) % len(FIELDS)
        elif ch in (curses.KEY_LEFT, curses.KEY_RIGHT, ord(" ")):
            if ftype == "bool":
                vals[key] = not vals[key]
            elif ftype == "dynlist":
                chs = init_choices()
                if chs:
                    cur = chs.index(vals[key]) if vals[key] in chs else -1
                    step = 1 if ch == curses.KEY_RIGHT else -1
                    vals[key] = chs[(cur + step) % len(chs)]
            elif ftype == "choice" and choices:
                cur = choices.index(vals[key]) if vals[key] in choices else 0
                step = 1 if ch == curses.KEY_RIGHT else -1
                vals[key] = choices[(cur + step) % len(choices)]
            else:
                status = f"{label}: type a value to edit"
        elif ch == ord("p"):
            pres = load_presets()
            if pres:
                names = sorted(pres)
                name = ask("preset: " + ", ".join(names) + " › ")
                if name in pres:
                    vals.update({k: pres[name][k] for k in DEFAULTS
                                 if k in pres[name]})
                    status = f"loaded preset «{name}»"
                elif name:
                    status = f"no preset «{name}»"
            else:
                status = "no presets saved (s to save one)"
        elif ch == ord("s"):
            name = ask("save preset as: ")
            if name:
                save_preset(name, vals)
                status = f"preset «{name}» saved to tools/train_presets.json"
        elif ch == curses.KEY_DC:
            vals[key] = DEFAULTS[key]
        elif ftype in ("text", "int", "float"):
            curses.echo()
            curses.curs_set(1)
            row = sel + top
            stdscr.addstr(row, col_val, " " * (w - col_val - 1))
            stdscr.addstr(row, col_val, str(vals[key]))
            stdscr.move(row, col_val + len(str(vals[key])))
            raw = b""
            try:
                raw = stdscr.getstr(row, col_val, w - col_val - 1)
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
        elif 32 <= ch < 127:
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
    ap.add_argument("--bench-list", dest="bench_list",
                    default=DEFAULTS["bench_list"],
                    help='bench spec "opp:games,opp:games" — opp is base '
                         '(current default engine), classic3 (FLOKK/L3), or '
                         'any engine id (nais8, blanka2, …); "0" = no local '
                         'bench. The same list is handed to the Nox gate.')
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
    print("--- session plan ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\nheadless equivalent (same session, no TUI):")
    print("  " + headless_command(result))
    print()
    try:
        ans = input("Start this session now? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"
    if ans == "y":
        run_session(result)
    else:
        print("not started — copy the command above, best under tmux: "
              "tmux new -s train './tools/train.py --go ...'")

if __name__ == "__main__":
    main()
