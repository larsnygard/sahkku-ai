"""Framdrift for ein treningskøyring — les den LEVANDE prosessen og loggen.

Loggen har éi linje per arbeidar («[w3] 15/25 parti»), men ingen samla
framdrift. Dette verktøyet finn køyringa (via /proc, so vi veit kor mange
syklusar og parti som var planlagt), les loggen, og skriv éi linje med
prosent, syklus, parti, tid og ETA.

Køyr:
    .venv/bin/python tools/train_status.py            # siste køyring
    .venv/bin/python tools/train_status.py ail08      # namn på køyringa
    .venv/bin/python tools/train_status.py ail08 -w   # oppdater kvar 5. sek
"""
import glob
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_run(name=None):
    logs = sorted(glob.glob(os.path.join(ROOT, "runs", "*_run.log")),
                  key=os.path.getmtime, reverse=True)
    if name:
        cand = os.path.join(ROOT, "runs", f"{name}_run.log")
        if os.path.exists(cand):
            return name, cand
    if not logs:
        raise SystemExit("fann ingen runs/*_run.log")
    return os.path.basename(logs[0])[:-8], logs[0]


def find_pid(name):
    """Finn prosessen som køyrer denne køyringa, og les argumenta hennar."""
    for d in glob.glob("/proc/[0-9]*"):
        try:
            with open(os.path.join(d, "cmdline"), "rb") as f:
                parts = [p.decode("utf-8", "replace") for p in f.read().split(b"\0") if p]
        except OSError:
            continue
        if "--run" in parts:
            i = parts.index("--run")
            if i + 1 < len(parts) and parts[i + 1] == name:
                return int(os.path.basename(d)), parts
    return None, None


def arg(parts, flag, default=None, cast=int):
    if parts and flag in parts:
        i = parts.index(flag)
        if i + 1 < len(parts):
            try:
                return cast(parts[i + 1])
            except (TypeError, ValueError):
                return default
    return default


def bar(frac, width=24):
    n = max(0, min(width, int(round(frac * width))))
    return "[" + "#" * n + "." * (width - n) + "]"


def read_log(path):
    st = {"cycles_done": 0, "current": None, "games_per_cycle": None,
          "workers": {}, "samples": 0, "finished": False, "stopped": False,
          "t_first": None}
    cre = re.compile(r"^\[c(\d+)\] self-play: (\d+) parti")
    done = re.compile(r"^\[c(\d+)\] ferdig")
    wk = re.compile(r"^\[w(\d+)\] (\d+)/(\d+) parti")
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.rstrip()
                m = cre.match(line)
                if m:
                    st["current"] = int(m.group(1))
                    st["games_per_cycle"] = int(m.group(2))
                    st["workers"] = {}
                    continue
                m = done.match(line)
                if m:
                    st["cycles_done"] = max(st["cycles_done"], int(m.group(1)))
                    st["workers"] = {}
                    continue
                m = wk.match(line)
                if m:
                    st["workers"][int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
                    continue
                if line.startswith("[stopp]"):
                    st["stopped"] = True
                if "alt ferdig" in line:
                    st["finished"] = True
    except OSError:
        pass
    st["t_first"] = os.path.getmtime(path)
    return st


def started_at(pid, fallback):
    """Når byrja prosessen? /proc/<pid> har oppstartstida si som mtime."""
    if pid:
        try:
            return os.path.getmtime(f"/proc/{pid}")
        except OSError:
            pass
    return fallback


def samples_done(run):
    n = 0
    for p in glob.glob(os.path.join(ROOT, "runs", run, "cycle*", "*.npz")):
        try:
            import numpy as np
            n += int(np.load(p)["value"].shape[0])
        except Exception:  # noqa: BLE001
            pass
    return n


def report(name, cycles_total=None, watch=False):
    run, log = find_run(name)
    pid, parts = find_pid(run)
    if parts:
        cycles_total = cycles_total or arg(parts, "--cycles", 1)
        games_arg = arg(parts, "--games", 0)
        interval = arg(parts, "--start-cycle", 1)
    else:
        games_arg = None

    st = read_log(log)
    if games_arg:
        gpc = games_arg
    else:
        gpc = st["games_per_cycle"] or 0
    cycles_total = cycles_total or max(st["cycles_done"] + (0 if st["finished"] else 1), 1)

    cur = st["current"] or 1
    done_now = sum(g for g, _ in st["workers"].values())
    if st["cycles_done"] >= cur:            # mellom syklusar
        done_now = gpc
    elapsed = time.time() - started_at(pid, st["t_first"])
    cycles_done = st["cycles_done"]

    if st["finished"]:
        frac, status = 1.0, "ferdig"
    elif st["stopped"]:
        frac = cycles_done / max(1, cycles_total)
        status = "stoppa på tidsbudsjettet"
    else:
        done_total = cycles_done * gpc + done_now
        frac = done_total / max(1, cycles_total * gpc)
        status = "køyrer"

    pct = 100.0 * frac
    eta = ""
    if 0 < frac < 1 and elapsed > 60:
        eta_s = elapsed * (1 - frac) / frac
        eta = f" · ETA {eta_s/3600:.1f} t" if eta_s > 3600 else f" · ETA {eta_s/60:.0f} min"

    live = f" · pid {pid}" if pid else " · (ingen levande prosess)"
    print(f"{run}  {bar(frac)} {pct:5.1f} %  {status}{live}")
    print(f"  syklus {cur}/{cycles_total} · parti i syklusen {done_now}/{gpc}"
          f" ({(100.0*done_now/gpc if gpc else 0):4.1f} %)"
          f" · parti totalt {cycles_done*gpc + done_now}/{cycles_total*gpc}")
    note = "" if st["workers"] else "  (ingen arbeidar har rapportert enno — fyrste linje kjem etter éin kamp)"
    print(f"  køyrt {elapsed/60:.0f} min{eta} · prøver på disk {samples_done(run)}"
          f" · arbeidarar {len(st['workers'])}{note}")
    return frac


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    watch = "-w" in sys.argv or "--watch" in sys.argv
    name = args[0] if args else None
    cycles = None
    if "--cycles" in sys.argv:
        cycles = int(sys.argv[sys.argv.index("--cycles") + 1])
    if not watch:
        report(name, cycles)
        return
    try:
        while True:
            os.system("clear")
            frac = report(name, cycles)
            if frac >= 1.0:
                break
            print("\n(oppdaterer kvart 5. sekund — Ctrl+C for å stoppe)")
            time.sleep(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
