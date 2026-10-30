"""End-to-end AlphaZero-lite pipeline: init → (self-play → train → export → eval)ⁿ.

One command runs the whole loop that the README documents step by step:

    python -m sahkku_ai.pipeline --run r1 --cycles 3 --games 200 --workers 4 \
        --steps 1500 --ch 64 --blocks 6 --fc 128 --device cuda --eval-games 20

Per cycle:
  1. SELF-PLAY  current net plays itself on the GPU; every decision is written
                as a sample (planes, globals, policy target π, value z) into
                runs/<run>/cycle<N>/shard_*.npz.
  2. TRAIN      resume from the previous cycle's checkpoint, fit policy (masked
                cross-entropy on π) + value (MSE on z) on the last `--window`
                cycles of shards (+ optional SPG shards).
  3. EXPORT     checkpoints/<run>/latest.pt → engines/sahkku-nn.{js,weights.json}
                (identical graph in pure JS, real JS rules injected).
  4. EVAL       optional match vs the reference engine, recorded in the log.

Everything is resumable: a cycle whose shards exist is not regenerated, and
`--start-cycle` skips ahead. Progress lands in runs/<run>/training_log.jsonl.
"""
from __future__ import annotations
import argparse, glob as globmod, json, os, re, shutil, subprocess, sys, time

from . import data as D
from . import selfplay as SP
from .model import Net
from .train import train as train_net

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def logline(run: str, rec: dict):
    os.makedirs(os.path.join("runs", run), exist_ok=True)
    p = os.path.join("runs", run, "training_log.jsonl")
    rec = dict(rec, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(p, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return p


def parse_eval(out: str):
    """Pull the score line out of tools/eval_match.js output."""
    m = re.search(r"SUM: A=(\d+)\s+B=(\d+)\s+uavgjort=(\d+)(?:\s+feil=(\d+))?", out)
    s = re.search(r"A-poengsum: ([0-9.]+) %", out)
    if not m:
        return None
    return {"a_wins": int(m.group(1)), "b_wins": int(m.group(2)),
            "draws": int(m.group(3)), "errors": int(m.group(4) or 0),
            "score_pct": float(s.group(1)) if s else None}


def run_loop(args):
    run = args.run
    rundir = os.path.join("runs", run)
    ckdir = os.path.join("checkpoints", run)
    os.makedirs(rundir, exist_ok=True)
    os.makedirs(ckdir, exist_ok=True)

    # ---- menneskeparti: eit lite shard som vert blanda inn i kvar minibatch ---
    # Policy-målet er berre vinnaren sine trekk; taparsida kjem med jamn
    # fordeling og verdi -1. So lærer nettet idéane til vinnaren utan å lære
    # dei trekkja som tapte.
    human_shard = None
    if args.human_games and not (0 < args.human_weight < 1):
        raise SystemExit(f"--human-weight must be between 0 and 1 (got {args.human_weight}); "
                         f"a weight >= 1 leaves no room for self-play in a batch")
    if args.human_games:
        human_shard = os.path.join(rundir, "human_shard.npz")
        samples = []
        # kvar sti kan vere ei fil (med eitt eller fleire parti) eller ein katalog
        paths = []
        for fp in args.human_games:
            if os.path.isdir(fp):
                paths.extend(sorted(globmod.glob(os.path.join(fp, "*.spg"))))
            else:
                paths.append(fp)
        for fp in paths:
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read()
                got = list(D.spg_samples(txt, winner_policy=True))
            except OSError as e:
                print(f"[human] cannot read {fp}: {e}", flush=True)
                continue
            ng = D.count_games(txt)
            print(f"[human] {os.path.basename(fp)}: {ng} games -> {len(got)} samples", flush=True)
            if ng and not got:
                print(f"[human] WARNING: {fp} holds {ng} game(s) but none replays on the "
                      f"engine — the file is NOT used", flush=True)
            samples.extend(got)
        if samples:
            D.save_shard(samples, human_shard)
            print(f"[human] {len(samples)} samples -> {human_shard} "
                  f"(weight {args.human_weight:.2f} of each batch)", flush=True)
        else:
            print("[human] no usable samples — running without", flush=True)
            human_shard = None

    # ---- optional supervised shard from SPG collections -------------------
    spg_shard = None
    if args.spg:
        spg_shard = args.spg_out or os.path.join("data", f"spg_{run}.npz")
        n = D.dataset_from_spg(args.spg, spg_shard)
        if not n:
            spg_shard = None

    # ---- starting point ---------------------------------------------------
    # Med --start-cycle N må løkka halde fram frå syklus N-1, ikkje frå
    # init.pt (elles trenar ein på nytt frå tilfeldig nett).
    if args.init:
        net_path = args.init
        print(f"[init] continuing from {net_path}", flush=True)
    else:
        p = os.path.join(ckdir, "init.pt")
        done = sorted(globmod.glob(os.path.join(ckdir, "c*.pt")),
                      key=lambda q: int(re.sub(r"\D", "", os.path.basename(q)) or 0))
        latest = [q for q in done
                  if int(re.sub(r"\D", "", os.path.basename(q)) or 0) < args.start_cycle]
        if latest and not args.force_init:
            net_path = latest[-1]
            print(f"[init] continuing from {net_path} (last completed cycle)", flush=True)
        elif os.path.exists(p) and not args.force_init:
            net_path = p
            print(f"[init] reusing {p}", flush=True)
        else:
            net = Net(ch=args.ch, blocks=args.blocks, fc=args.fc)
            net.meta = {"run": run, "cycle": 0, "steps": 0, "history": []}
            net.save(p, net.meta)
            net_path = p
            print(f"[init] ny nett {net.cfg} -> {p}", flush=True)

    t_run = time.time()
    last_cycle_s = 0.0            # kor lang den FØRRE syklusen vart
    for cycle in range(args.start_cycle, args.cycles + 1):
        # Tidsbudsjett: vi stoppar mellom syklusar (ein halv shard er verdilaus),
        # aldri midt i ei datagenerering.
        elapsed = time.time() - t_run
        if args.max_seconds and elapsed >= args.max_seconds:
            print(f"[stop] time budget ({args.max_seconds}s) used up "
                  f"after cycle {cycle - 1}", flush=True)
            break
        # …og vi startar ikkje ein syklus som ikkje REKK å bli ferdig: ein syklus
        # som byrjar 60 s før fristen gjer at køyringa gjekk 57 minutt over
        # budsjettet i staden for å stoppe nesten på slaget.
        if args.max_seconds and last_cycle_s and \
                elapsed + last_cycle_s > args.max_seconds:
            print(f"[stop] the next cycle needs ~{last_cycle_s:.0f}s but only "
                  f"{args.max_seconds - elapsed:.0f}s remain — stopping so the "
                  f"budget is honoured", flush=True)
            break
        cyc_dir = os.path.join(rundir, f"cycle{cycle}")
        t0 = time.time()
        games = args.games if cycle > 1 else args.games_first or args.games
        # -- 1. self-play -----------------------------------------------
        shards = sorted(globmod.glob(os.path.join(cyc_dir, "shard_*.npz")))
        if shards and not args.force_selfplay:
            print(f"[c{cycle}] self-play already present: {len(shards)} shards", flush=True)
        else:
            print(f"[c{cycle}] self-play: {games} games, {args.workers} workers, "
                  f"device={args.device}", flush=True)
            SP.run_selfplay(net_path, cyc_dir, device=args.device, games=games,
                            workers=args.workers, rollouts=args.rollouts,
                            temp=args.temp, noise_alpha=args.noise_alpha,
                            noise_eps=args.noise_eps, seed=args.seed + cycle,
                            prior_eps=args.prior_eps, tau_q=args.tau_q,
                            sims=args.sims, c_puct=args.c_puct,
                            leaf_batch=args.leaf_batch,
                            opponent=args.train_opponent,
                            opponent_level=args.train_opponent_level,
                            queen_delay=args.queen_delay,
                            queen_delay_soldiers=args.queen_delay_soldiers,
                            max_plies=args.max_plies,
                            nesseby=args.nesseby,
                            nesseby_fraction=args.nesseby_fraction,
                            aggressive=args.aggressive,
                            aggressive_fraction=args.aggressive_fraction,
                            styles=args.styles,
                            style_fraction=args.style_fraction)

        # -- 2. train ---------------------------------------------------
        pats = [os.path.join(rundir, f"cycle{c}", "*.npz")
                for c in range(max(1, cycle - args.window + 1), cycle + 1)]
        if spg_shard:
            pats.append(spg_shard)
        shard_expr = " ".join(pats)
        ck = os.path.join(ckdir, f"c{cycle}.pt")
        if os.path.exists(ck) and not args.force_train:
            print(f"[c{cycle}] trening finst alt: {ck}", flush=True)
        else:
            train_net(f"{run}_c{cycle}", shard_expr, init=net_path,
                      steps=args.steps, batch=args.batch, lr=args.lr,
                      wd=args.wd, warmup=args.warmup, device=args.device,
                      seed=args.seed, log_every=args.log_every,
                      save_every=args.save_every,
                      extra=human_shard, extra_weight=args.human_weight)
            src = os.path.join("checkpoints", f"{run}_c{cycle}", "latest.pt")
            shutil.copyfile(src, ck)
        net_path = ck

        # -- 3. export --------------------------------------------------
        from .export import export
        rpath, wpath = export(ck, args.out, "fp32")

        # -- 4. eval (valfritt) -----------------------------------------
        ev = None
        if args.eval_games:
            cmd = ["node", os.path.join("tools", "eval_match.js"),
                   f"node {rpath}", f"node {args.opponent}",
                   str(args.eval_games), str(args.seed + cycle)]
            print(f"[c{cycle}] eval: {' '.join(cmd)}", flush=True)
            try:
                out = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                                     text=True, timeout=args.eval_timeout).stdout
                ev = parse_eval(out)
                print("[c{0}] eval: {1}".format(cycle, ev), flush=True)
                with open(os.path.join(rundir, f"eval_c{cycle}.txt"), "w") as f:
                    f.write(out)
            except Exception as e:                      # noqa: BLE001
                print(f"[c{cycle}] eval feila: {e}", flush=True)

        n_samples = 0
        for s in globmod.glob(os.path.join(cyc_dir, "shard_*.npz")):
            import numpy as np
            n_samples += int(np.load(s)["value"].shape[0])
        rec = {"cycle": cycle, "games": games, "new_samples": n_samples,
               "shards": len(globmod.glob(os.path.join(cyc_dir, "shard_*.npz"))),
               "checkpoint": ck, "engine": rpath, "eval": ev,
               "elapsed_s": round(time.time() - t0, 1)}
        p = logline(run, rec)
        last_cycle_s = float(rec["elapsed_s"])
        print(f"[c{cycle}] done in {rec['elapsed_s']}s -> {p}", flush=True)
    print(f"all done: {args.out}.js (+ .weights.json) and {ckdir}/", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="r1")
    ap.add_argument("--human-games", nargs="+", default=None,
                    help="SPG files with human games to mix in")
    ap.add_argument("--human-weight", type=float, default=0.10,
                    help="share of each minibatch drawn from --human-games")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--start-cycle", type=int, default=1)
    ap.add_argument("--games", type=int, default=200, help="games per cycle")
    ap.add_argument("--games-first", type=int, default=0,
                    help="games in the first cycle (default: --games)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--noise-alpha", type=float, default=0.3)
    ap.add_argument("--noise-eps", type=float, default=0.25)
    ap.add_argument("--prior-eps", type=float, default=0.2)
    ap.add_argument("--tau-q", type=float, default=1.0)
    ap.add_argument("--sims", type=int, default=0,
                    help="MCTS simulations per decision in self-play (0 = playouts)")
    ap.add_argument("--c-puct", type=float, default=1.6)
    ap.add_argument("--leaf-batch", type=int, default=8)
    ap.add_argument("--train-opponent", dest="train_opponent", default="",
                    help="train against the classic CPU instead of self-play (e.g. \"classic3\")")
    ap.add_argument("--train-opponent-level", dest="train_opponent_level", type=int, default=3)
    ap.add_argument("--queen-delay", type=int, default=-1,
                    help="self-play handicap: -1 off, 0/1 fixed side, 2 = random side "
                         "per game. That side may not activate its queen until it has "
                         "--queen-delay-soldiers soldatar att (AIL0.6-eksperimentet)")
    ap.add_argument("--queen-delay-soldiers", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=300,
                    help="ply cap per game during data generation")
    ap.add_argument("--nesseby", action="store_true",
                    help="self-play with the Nesseby rule (the king cannot be captured by "
                         "landing until a soldier has reached the opponent's home row)")
    ap.add_argument("--nesseby-fraction", type=float, default=1.0,
                    help="share of games played with Nesseby when --nesseby is on "
                         "(0.5 = mixed; one net then learns both rule sets)")
    ap.add_argument("--styles", action="store_true",
                    help="taktikk-duellar: kvar kamp spelar éi side «few» eller «kingfwd» "
                         "mot «many» eller «kingguard» (sjå selfplay.PAIRS)")
    ap.add_argument("--style-fraction", type=float, default=1.0,
                    help="del av partia som er stil-duellar")
    ap.add_argument("--aggressive", action="store_true",
                    help="one side per game plays aggressively (weight on captures and "
                         "advances); the side alternates so the net learns both")
    ap.add_argument("--aggressive-fraction", type=float, default=1.0,
                    help="share of games with an aggressive side (1.0 = all)")
    ap.add_argument("--max-seconds", type=int, default=0,
                    help="stop after N seconds (0 = only --cycles). Checked between cycles.")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--window", type=int, default=2,
                    help="how many cycles of shards the trainer sees (replay buffer)")
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--fc", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--spg", default=None, help="mappe med .spg-samlingar (valfri)")
    ap.add_argument("--spg-out", default=None)
    ap.add_argument("--out", default="engines/sahkku-nn")
    ap.add_argument("--eval-games", type=int, default=0)
    ap.add_argument("--opponent", default="../SnartSahkku/engines/sahkku-ref.js")
    ap.add_argument("--eval-timeout", type=int, default=3600)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--force-init", action="store_true")
    ap.add_argument("--force-selfplay", action="store_true")
    ap.add_argument("--force-train", action="store_true")
    run_loop(ap.parse_args())


if __name__ == "__main__":
    main()
