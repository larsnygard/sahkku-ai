#!/usr/bin/env bash
# train-forever.sh — train cycle after cycle, and bench 20 games against the
# classic CPU (FLOKK, level 3) after EVERY cycle. Runs until stopped.
#
#   ./tools/train-forever.sh --run blanka3 --init engines/blanka2-nn.pt
#   ./tools/train-forever.sh --run blanka3 --games 400 --bench-games 20
#   ./tools/train-forever.sh --run blanka3 --max-cycles 1        # prøvekøyring
#   ./tools/train-forever.sh --run blanka3 --bench-only          # berre benk
#
# Kvar runde:
#   1. badass-trainer.sh --run <run> --cycles 1     (neste syklus, ikkje fyrste)
#   2. node tools/eval_match.js <run>-nn.js cpu_uci.js 20 <seed>
#   3. ei line i runs/<run>_bench.csv + loggen runs/<run>_forever.log
#
# GPU: Strata held ~15.7 av 16 GB VRAM. Scriptet ventar (i staden for å
# krasje på CUDA OOM) til det er --free-mb ledig, med mindre --no-wait.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

RUN=""
INIT=""
GAMES="400"
BENCH_GAMES="20"
SEED="7"
OPP="classic3"
FREE_MB="3000"
MAX_CYCLES="0"          # 0 = uendeleg
BENCH_ONLY="0"
NO_WAIT="0"
PAUSE="10"              # sekund mellom rundar / etter feil
EXTRA=()

die() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
say() { printf '%s\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --run)         RUN="$2"; shift 2 ;;
    --init)        INIT="$2"; shift 2 ;;
    --games)       GAMES="$2"; shift 2 ;;
    --bench-games) BENCH_GAMES="$2"; shift 2 ;;
    --seed)        SEED="$2"; shift 2 ;;
    --opponent)    OPP="$2"; shift 2 ;;
    --free-mb)     FREE_MB="$2"; shift 2 ;;
    --max-cycles)  MAX_CYCLES="$2"; shift 2 ;;
    --pause)       PAUSE="$2"; shift 2 ;;
    --bench-only)  BENCH_ONLY="1"; shift ;;
    --no-wait)     NO_WAIT="1"; shift ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    --)            shift; EXTRA+=("$@"); break ;;
    *)             EXTRA+=("$1"); shift ;;
  esac
done

[ -n "$RUN" ] || die "--run manglar (t.d. --run blanka3)"
export RUN GAMES BENCH_GAMES SEED OPP PAUSE BENCH_ONLY

LOG="$ROOT/runs/${RUN}_forever.log"
CSV="$ROOT/runs/${RUN}_bench.csv"
mkdir -p "$ROOT/runs"
[ -f "$CSV" ] || printf 'cycle,tid,poeng_pct,vinst,tap,uavgjort,seed,benk_log\n' > "$CSV"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' '{f=$1-$2; if (f>m) m=f} END {print (m==""?0:m+0)}'; }

wait_for_gpu() {
  local tries=0
  while :; do
    local f; f=$(free_mb)
    if [ "$f" -ge "$FREE_MB" ]; then
      [ "$tries" -gt 0 ] && log "GPU fri: ${f} MiB ledig"
      return 0
    fi
    if [ "$NO_WAIT" = "1" ]; then
      log "ÅTVARING: berre ${f} MiB ledig VRAM (< ${FREE_MB}); held fram (--no-wait)"
      return 0
    fi
    tries=$((tries+1))
    [ $((tries % 12)) -eq 1 ] && log "ventar på GPU: ${f} MiB ledig, treng ${FREE_MB} (Strata? stopp ho med: systemctl --user stop strata-swift)"
    sleep 20
  done
}

last_cycle() {
  python3 - "$ROOT/runs/$RUN/training_log.jsonl" <<'PY'
import json, sys
try:
    rows=[json.loads(l) for l in open(sys.argv[1])]
    print(max((r["cycle"] for r in rows), default=0))
except Exception:
    print(0)
PY
}

bench() {                       # bench <syklus> -> skriv CSV-line, ekko resultat
  local cyc="$1" eng="$ROOT/engines/${RUN}-nn.js"
  local seed=$(( SEED + cyc ))
  local out="$ROOT/runs/${RUN}_bench_c${cyc}.txt"
  if [ ! -f "$eng" ]; then
    log "c${cyc}: finn ikkje $eng — hoppar over benk"
    return 0
  fi
  log "c${cyc}: benk ${BENCH_GAMES} parti mot ${OPP} (seed ${seed})"
  node "$ROOT/tools/eval_match.js" "node $eng" "node $ROOT/tools/cpu_uci.js" \
       "$BENCH_GAMES" "$seed" > "$out" 2>&1
  local pct wins losses draws
  pct=$(grep -oE 'A-poengsum: [0-9.]+' "$out" | tail -1 | awk '{print $2}')
  wins=$(grep -oE 'SUM: A=[0-9]+' "$out" | tail -1 | cut -d= -f2)
  losses=$(grep -oE 'B=[0-9]+' "$out" | tail -1 | cut -d= -f2)
  draws=$(grep -oE 'uavgjort=[0-9]+' "$out" | tail -1 | cut -d= -f2)
  if [ -z "${pct:-}" ]; then
    log "c${cyc}: benken gav ingen poengsum — sjå $out"
    printf '%s,%s,,,,%s,%s,%s\n' "$cyc" "$(date '+%F %T')" "$seed" "" "$out" >> "$CSV"
    return 0
  fi
  log "c${cyc}: RESULTAT ${pct} % mot ${OPP}  (A=${wins:-?} B=${losses:-?} uavgjort=${draws:-?})"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$cyc" "$(date '+%F %T')" "$pct" \
         "${wins:-}" "${losses:-}" "${draws:-}" "$seed" "$out" >> "$CSV"
}

trap 'log "avbroten — stoppar (CSV: $CSV)"; exit 0' INT TERM

log "start: run=$RUN games=$GAMES bench=$BENCH_GAMES opp=$OPP max_cycles=$MAX_CYCLES"

if [ "$BENCH_ONLY" = "1" ]; then
  bench "$(last_cycle)"
  exit 0
fi

rounds=0
fails=0
while :; do
  rounds=$((rounds+1))
  if [ "$MAX_CYCLES" != "0" ] && [ "$rounds" -gt "$MAX_CYCLES" ]; then
    log "ferdig: ${MAX_CYCLES} runde(r) køyrd"
    exit 0
  fi
  wait_for_gpu

  # -- 1. éin treningssyklus -------------------------------------------
  before=$(last_cycle)
  log "runde ${rounds}: trenar syklus $((before+1)) (${GAMES} parti)"
  targs=( --run "$RUN" --games "$GAMES" --cycles 1 --train-opponent "$OPP" --hours 0 )
  [ -n "$INIT" ] && targs+=( --init "$INIT" )
  if ( cd "$ROOT" && ./badass-trainer.sh "${targs[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" \
        >> "$ROOT/runs/${RUN}_run.log" 2>&1 ); then
    fails=0
    after=$(last_cycle)
    if [ "$after" -le "$before" ]; then
      log "runde ${rounds}: ingen ny syklus vart fullført — ventar ${PAUSE}s"
      sleep "$PAUSE"; continue
    fi
    log "runde ${rounds}: syklus ${after} ferdig"
  else
    fails=$((fails+1))
    log "runde ${rounds}: TRENING FEILA (${fails} på rad) — siste 5 linjer av runs/${RUN}_run.log:"
    tail -5 "$ROOT/runs/${RUN}_run.log" 2>/dev/null | sed 's/^/    | /' | tee -a "$LOG"
    # INIT er berre for fyrste syklus; etter ein feil held vi fram på same run
    sleep $((PAUSE * (fails < 5 ? fails : 5)))
    continue
  fi

  # -- 2. benk 20 parti mot CPU-en --------------------------------------
  bench "$(last_cycle)"

  # -- 3. vidare ---------------------------------------------------------
  INIT=""            # syklus 1 er gjort: runen held fram frå eigne sjekkpunkt
  sleep "$PAUSE"
done
