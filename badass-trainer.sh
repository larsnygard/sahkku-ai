#!/usr/bin/env bash
# badass-trainer.sh — tren sáhkku-nettet på denne maskina.
#
#   ./badass-trainer.sh                 # 2 timar, held fram frå AIL0.9 → ail10
#   ./badass-trainer.sh --hours 6       # lengre økt
#   ./badass-trainer.sh --cycles 2      # NØYAKTIG to syklusar, so stoppar han
#   ./badass-trainer.sh --cycles 3 --hours 0   # berre syklusar, ingen tidsgrense
#   ./badass-trainer.sh --games 600     # fleire parti per syklus
#   ./badass-trainer.sh --run ail11 --init engines/ail10-nn.pt
#   ./badass-trainer.sh --dry           # vis berre kva som ville køyrt
#   ./badass-trainer.sh --nesseby       # blanda vanleg + Nesseby (0.5 av partia)
#   ./badass-trainer.sh --nesseby --nesseby-fraction 0.3   # 30 % Nesseby
#   ./badass-trainer.sh --threads 12                     # tving 12 arbeidarar
#       (auto = fysiske kjernar; ei eksplisitt vekt vinn over VRAM-/minnetaka,
#        men du får ei åtvaring om ho ligg over det maskina tilrår)
#   ./badass-trainer.sh --aggressive                     # éi side spelar aggressivt
#   ./badass-trainer.sh --styles                         # taktikk-duellar (NAIS.6):
#       few vs many (kor mange soldatar som vert aktiverte) og
#       kingfwd vs kingguard (kva kongen gjer etter at han er kapra)
#   ./badass-trainer.sh --human-games runs/human/*.spg   # blanda inn menneskeparti
#       (standardvekt 0,05; høgare gav klårt svakare nett i NAIS.3-forsøket)
#
# Skriptet finn sjølv ut av:
#   * fysiske kjernar (arbeidarar = kjernar, ikkje trådar)
#   * CUDA (GPU) eller ikkje — utan GPU vert sims/arbeidarar skrudde ned
#   * om ei delvis fullført syklus må ryddast (elles trenar han på avkutta data)
#
# Logg:  runs/<run>_run.log     Framdrift:  tools/train_status.py <run> -w
set -euo pipefail
cd "$(dirname "$0")"

RUN="ail10"
INIT="engines/ail09-nn.pt"
HOURS="2"
CYCLES=""              # sett med --cycles: nøyaktig så mange syklusar
GAMES="400"
SIMS="45"
WORKERS=""
THREADS_OVERRIDE=""      # sett med --threads/--workers; vinn over alle automatiske tak
STEPS="900"
BATCH="256"
LR="3.0e-4"
LEAFBATCH="8"
WINDOW="2"
SEED="81"
MAXPLIES="2500"
OUT=""
NESSBY="0"
NBFRAC=""
AGGR="0"
STYLES="0"
STYLEFRAC=""
AGGRFRAC=""
HUMANGAMES=""
HUMANWEIGHT="0.05"
DRY="0"

die() { printf '\033[31mFEIL:\033[0m %s\n' "$*" >&2; exit 1; }
# Set -e kan stoppe skriptet kvar som helst UTAN eit ord. Denne fella gjer kvar
# slik stopp til ein feilmelding med kommandoen og linja, so ein veit kva som
# braut i staden for å gjette ut frå fire linjer vanleg utskrift.
trap 'rc=$?; printf "\033[31m──── ABORTED (exit %s) at line %s\033[0m\n  command: %s\n" \
        "$rc" "$LINENO" "$BASH_COMMAND" >&2' ERR
say() { printf '\033[36m▸\033[0m %s\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --run)        RUN="$2"; shift 2 ;;
    --init)       INIT="$2"; shift 2 ;;
    --hours)      HOURS="$2"; shift 2 ;;
    --cycles)     CYCLES="$2"; shift 2 ;;
    --games)      GAMES="$2"; shift 2 ;;
    --sims)       SIMS="$2"; shift 2 ;;
    --threads|--workers) THREADS_OVERRIDE="$2"; shift 2 ;;
    --steps)      STEPS="$2"; shift 2 ;;
    --lr)         LR="$2"; shift 2 ;;
    --leaf-batch) LEAFBATCH="$2"; shift 2 ;;
    --seed)       SEED="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    --nesseby)    NESSBY="1"; shift ;;
    --nesseby-fraction) NBFRAC="$2"; shift 2 ;;
    --styles)     STYLES="1"; shift ;;
    --style-fraction) STYLEFRAC="$2"; shift 2 ;;
    --aggressive) AGGR="1"; shift ;;
    --aggressive-fraction) AGGRFRAC="$2"; shift 2 ;;
    --human-games)  shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do
                      HUMANGAMES="$HUMANGAMES $1"; shift; done ;;
    --human-weight) HUMANWEIGHT="$2"; shift 2 ;;
    --dry)        DRY="1"; shift ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *)            die "ukjent val «$1» (sjå --help)" ;;
  esac
done

# ---- miljø ---------------------------------------------------------------
[ -x .venv/bin/python3 ] || die "fann ingen .venv. Køyr fyrst:
  python3 -m venv .venv && . .venv/bin/activate
  pip install torch --index-url https://download.pytorch.org/whl/cu126   # eller /cpu
  pip install numpy"
PY=".venv/bin/python3"
$PY -c 'import torch, numpy' 2>/dev/null || die "torch/numpy manglar i .venv"
[ -f "$INIT" ] || die "finn ikkje startvekta «$INIT» (--init)"

# Startvekta må ha same tal globale trekk som features.py. Er ho smalare
# (t.d. 15 då koden no vil ha 16), utvidar vi henne med nullkolonnar:
# nettet reknar då nøyaktig som før, men kan lære dei nye trekka.
WANT_G=$($PY -c 'from sahkku_ai.features import N_GLOBAL; print(N_GLOBAL)' 2>/dev/null || echo "?")
HAVE_G=$($PY - <<EOF
import torch
ck = torch.load("$INIT", map_location="cpu", weights_only=False)
print(ck["state"]["pol_fc.weight"].shape[1] - 360)
EOF
)
if [ "$HAVE_G" != "$WANT_G" ]; then
  if [ "$HAVE_G" -lt "$WANT_G" ] 2>/dev/null; then
    # Legg den utvida vekta VED sida av kjelda (ikkje i checkpoints/ rot):
    # elles kolliderer c3-g16.pt frå to ulike køyringar.
    case "$INIT" in
      checkpoints/*/*) WIDE="${INIT%.pt}-g${WANT_G}.pt" ;;
      *)               WIDE="checkpoints/$(basename "${INIT%.pt}")-g${WANT_G}.pt" ;;
    esac
    if [ ! -f "$WIDE" ]; then
      say "startvekta har ${HAVE_G} globale trekk, koden vil ha ${WANT_G} → utvidar til $WIDE"
      $PY tools/expand_net.py "$INIT" "$WIDE" >/dev/null || die "utviding feila"
    fi
    INIT="$WIDE"
  else
    die "startvekta har ${HAVE_G} globale trekk, men koden vil ha ${WANT_G} (nyare sjekkpunkt? bruk --init)"
  fi
fi

# ---- maskin -------------------------------------------------------------
THREADS=$(nproc)
CORES=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "${CORES:-0}" -gt 0 ] || CORES=$(( THREADS / 2 > 0 ? THREADS / 2 : 1 ))
if [ -n "$THREADS_OVERRIDE" ]; then
  case "$THREADS_OVERRIDE" in
    ''|*[!0-9]*) die "--threads must be a whole number >= 1 (got «$THREADS_OVERRIDE»)" ;;
  esac
  [ "$THREADS_OVERRIDE" -ge 1 ] || die "--threads must be >= 1"
  WORKERS="$THREADS_OVERRIDE"
else
  WORKERS="$CORES"                          # éin arbeidar per fysisk kjerne
fi

GPU=$($PY - <<'EOF'
import torch
if not torch.cuda.is_available():
    print("none"); raise SystemExit
p = torch.cuda.get_device_properties(0)
print(f"{p.name}|{p.total_memory/1024**3:.1f}GB|{p.major}.{p.minor}")
EOF
)
DEVICE="cuda"
if [ "$GPU" != "none" ]; then
  # VRAM er grensa, ikkje modellen: kvart arbeidar-prosess held sin eigen
  # CUDA-kontekst (~0,4 GB). 8 GB ≈ 12-14 arbeidarar; 4 GB ≈ 4-5.
  VRAM_GB=$(echo "$GPU" | cut -d'|' -f2 | tr -d 'GB')
  MAXW=$($PY -c "print(max(2, int((float('$VRAM_GB')*0.8 - 1.0) / 0.45)))")
  if [ "$WORKERS" -gt "$MAXW" ]; then
    if [ -n "$THREADS_OVERRIDE" ]; then
      say "ÅTVARING: $WORKERS arbeidarar er over det VRAM tilrår (${VRAM_GB} GB → ~$MAXW); held fram med $WORKERS fordi --threads vart gjeve"
    else
      say "VRAM ${VRAM_GB} GB → arbeidarar skorne frå $WORKERS til $MAXW (kvar prosess held ein CUDA-kontekst)"
      WORKERS="$MAXW"
    fi
  fi
  # Ampere+ (cc ≥ 8): TF32 gjev raskare konvolusjonar utan å røre eksporten.
  CC=$(echo "$GPU" | cut -d'|' -f3)
  CC_MAJOR=$(echo "$CC" | cut -d. -f1)
  if [ "$CC_MAJOR" -ge 8 ] 2>/dev/null; then
    export NVIDIA_TF32_OVERRIDE=1
    say "cc ${CC} → TF32 på (raskare trening; vektene er uendra i eksporten)"
  fi
fi
if [ "$GPU" = "none" ]; then
  DEVICE="cpu"
  say "Ingen CUDA-GPU: køyrer på CPU. Skruar ned sims ($SIMS→20) og arbeidarar."
  SIMS=20
  if [ "$WORKERS" -gt "$CORES" ] && [ -z "$THREADS_OVERRIDE" ]; then
    WORKERS="$CORES"
  fi
fi

# CPU utan GPU er ~10x tregare per avgjerd: fleire arbeidarar gjev meir data
# per time, men kvar arbeidar treng ein kjerne.
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)

say "maskin:  $CORES fysiske kjernar / $THREADS trådar, ${RAM_GB} GB RAM"
if [ "$DEVICE" = "cuda" ]; then say "gpu:     ${GPU}"; else say "gpu:     ingen (CPU)"; fi
say "køyring: $RUN frå $INIT"
if [ "$NESSBY" = "1" ]; then
  [ -n "$NBFRAC" ] || NBFRAC="0.5"
  say "reglar:   vanleg sáhkku + Nesseby (${NBFRAC} av partia) — eitt nett, flaget g[15] skil dei"
else
  say "reglar:   vanleg sáhkku (Nesseby av)"
fi
if [ "$AGGR" = "1" ]; then
  [ -n "$AGGRFRAC" ] || AGGRFRAC="1.0"
  say "stil:     éi side aggressiv i ${AGGRFRAC} av partia (slag ×2,5, framrykk ×1,4)"
fi
if [ "$STYLES" = "1" ]; then
  [ -n "$STYLEFRAC" ] || STYLEFRAC="1.0"
  say "taktikk:  duellar i ${STYLEFRAC} av partia — few mot many, kingfwd mot kingguard"
fi
if [ -n "$HUMANGAMES" ]; then
  NHUM=$(ls -1 $HUMANGAMES 2>/dev/null | wc -l || true)
  say "menneske: ${NHUM} filer blanda inn, vekt ${HUMANWEIGHT} av kvar batch"
fi
if [ -n "$THREADS_OVERRIDE" ]; then
  WSORCE="${WORKERS} arbeidarar (--threads)"
else
  WSORCE="${WORKERS} arbeidarar (auto: fysiske kjernar)"
fi
# Kor lang tid éin syklus tek, og passar han i tidsbudsjettet?
# Målt på P100 med 6 arbeidarar: 3,6-4,3 parti/min ved 45 sims, dvs. ~0,6
# parti/min per arbeidar. Tida skalerer om lag lineært med sims.
# --hours 0 = inga tidsgrense (berre --cycles styrer). SECS må reknast her,
# for plan-linja og åtvaringa nedanfor treng å vite om det finst ei grense.
SECS=""
if $PY -c "import sys; sys.exit(0 if float('$HOURS') > 0 else 1)"; then
  SECS=$($PY -c "print(int(float('$HOURS')*3600))")
fi
RATE_PW=$($PY -c "print(round(27.0 / max(1, $SIMS), 3))")          # parti/min/arbeidar
CYCLE_MIN=$($PY -c "print(int(round($GAMES / (max(1,$WORKERS) * float('$RATE_PW')))))")
BUDGET_MIN=$($PY -c "print(int(round(float('$HOURS') * 60)))")
if [ -n "$CYCLES" ]; then
  TOTAL_MIN=$($PY -c "print(int($CYCLES * $CYCLE_MIN))")
  WORDSYK=$([ "$CYCLES" -eq 1 ] && echo syklus || echo syklusar)
  say "plan:    ${CYCLES} ${WORDSYK} (~${TOTAL_MIN} min) · ${GAMES} parti/syklus (~${CYCLE_MIN} min) · ${WSORCE} · ${SIMS} sims · $DEVICE"
else
  say "plan:    ${HOURS} t · ${GAMES} parti/syklus (~${CYCLE_MIN} min) · ${WSORCE} · ${SIMS} sims · $DEVICE"
fi
# Tidsgrense OG syklustal samstundes: sei kven som kjem til å vinne
if [ -n "$CYCLES" ] && [ -n "$SECS" ] && [ "$TOTAL_MIN" -gt "$BUDGET_MIN" ]; then
  printf '\033[33m'
  say "ÅTVARING: ${CYCLES} ${WORDSYK} tek ~${TOTAL_MIN} min, men tidsgrensa er ${BUDGET_MIN} min."
  REACH=$(($BUDGET_MIN / $CYCLE_MIN))
  say "          Tidsgrensa vinn — køyringa stoppar etter ~${REACH} $([ "$REACH" -eq 1 ] && echo syklus || echo syklusar)."
  say "          Vil du ha alle ${CYCLES}: bruk --hours 0 (eller --hours $(($TOTAL_MIN / 60 + 1)))."
  printf '\033[0m'
fi
if [ -n "$SECS" ] && [ "$CYCLE_MIN" -gt "$BUDGET_MIN" ]; then
  FIT=$($PY -c "print(max(20, int($BUDGET_MIN * max(1,$WORKERS) * float('$RATE_PW') * 0.9)))")
  printf '\033[33m'
  say "ÅTVARING: éin syklus tek ~${CYCLE_MIN} min, men budsjettet er ${BUDGET_MIN} min."
  say "          Tidsbudsjettet vert sjekka MELLOM syklusar — syklus 1 går ferdig uansett,"
  say "          so køyringa endar ~${CYCLE_MIN} min etter start, ikkje etter ${HOURS} t."
  say "          Vil du halde deg innanfor: --games ${FIT} (då stoppar han etter éin syklus)."
  printf '\033[0m'
fi

# Maskinvaretilpassing: minne til 8 arbeidarar er ~1-2 GB; 64 GB er rikeleg.
if [ "$RAM_GB" -lt 16 ] && [ -z "$THREADS_OVERRIDE" ]; then
  WORKERS=$(( WORKERS < 4 ? WORKERS : 4 ))
  say "lite minne → ${WORKERS} arbeidarar"
fi

# ---- rydde delvis syklus -------------------------------------------------
# Pipeline reknar «det finst shard-filer» som «syklusen er ferdig». Ei avbrote
# økt må difor ryddast, elles trenar han på avkutta data.
LAST_DONE=0
if [ -f "runs/$RUN/training_log.jsonl" ]; then
  LAST_DONE=$($PY - <<EOF
import json
try:
    rows=[json.loads(l) for l in open("runs/$RUN/training_log.jsonl")]
    print(max((r["cycle"] for r in rows), default=0))
except Exception:
    print(0)
EOF
)
fi
NEXT=$(( LAST_DONE + 1 ))
for d in runs/"$RUN"/cycle*; do
  [ -d "$d" ] || continue
  c=${d##*cycle}
  if [ "$c" -gt "$LAST_DONE" ]; then
    say "ryddar avbrote syklus $c ($(ls "$d"/*.npz 2>/dev/null | wc -l) delvise shard-filer)"
    rm -rf "$d"
  fi
done

[ -n "$OUT" ] || OUT="engines/${RUN}-nn"

CMD=(
  "$PY" -m sahkku_ai.pipeline
  --run "$RUN" --start-cycle "$NEXT" --cycles "${CYCLES:-999}"
  --init "$INIT"
  --games "$GAMES" --workers "$WORKERS" --sims "$SIMS" --leaf-batch "$LEAFBATCH"
  --steps "$STEPS" --batch "$BATCH" --lr "$LR" --warmup 100
  --ch 64 --blocks 6 --fc 128 --device "$DEVICE" --window "$WINDOW"
  --seed "$SEED" --max-plies "$MAXPLIES"
  --log-every 300 --save-every 450 --eval-games 0
  --out "$OUT"
)
if [ -n "$SECS" ]; then
  CMD+=(--max-seconds "$SECS")
fi
if [ "$NESSBY" = "1" ]; then
  [ -n "$NBFRAC" ] || NBFRAC="0.5"
  CMD+=(--nesseby --nesseby-fraction "$NBFRAC")
fi
if [ "$AGGR" = "1" ]; then
  [ -n "$AGGRFRAC" ] || AGGRFRAC="1.0"
  CMD+=(--aggressive --aggressive-fraction "$AGGRFRAC")
fi
if [ "$STYLES" = "1" ]; then
  [ -n "$STYLEFRAC" ] || STYLEFRAC="1.0"
  CMD+=(--styles --style-fraction "$STYLEFRAC")
fi
# Menneskeparti: eit lite, kuratert shard som vert blanda inn per minibatch.
# Vekta er med vilje låg — 12 % gjorde nettet klårt svakare (NAIS.3), sjå README.
if [ -n "$HUMANGAMES" ]; then
  # vekt >= 1 gjer batch - n_ex <= 0 og numpy krasjar midt i treninga
  $PY -c "import sys; w=float('$HUMANWEIGHT'); sys.exit(0 if 0 < w < 1 else 1)" \
    || die "--human-weight must be between 0 and 1 (got $HUMANWEIGHT); 0.05 is the measured default"
  # shellcheck disable=SC2086
  CMD+=(--human-games $HUMANGAMES --human-weight "$HUMANWEIGHT")
fi

if [ "$DRY" = "1" ]; then
  printf '\n%s\n' "${CMD[*]}"
  exit 0
fi

# ---- pre-flight: a failure here costs seconds, not a wasted run ----------
if [ "$DEVICE" = "cuda" ] && ! $PY -c 'import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  die "CUDA was requested but torch sees no GPU. Check: nvidia-smi · driver · that the venv has a CUDA build of torch (not the CPU wheel). Use --device cpu to train on CPU."
fi
if [ -n "$HUMANGAMES" ]; then
  # shellcheck disable=SC2086
  MISSING=""
  for f in $HUMANGAMES; do
    [ -e "$f" ] || MISSING="$MISSING $f"
  done
  [ -z "$MISSING" ] || die "human game file(s) not found:$MISSING"
  # count the samples NOW instead of discovering an empty shard an hour in
  # shellcheck disable=SC2086
  set +e
  HG=$($PY -m sahkku_ai.data --spg $HUMANGAMES --out /tmp/_human_preflight.npz --winner-policy 2>&1 | tail -1)
  HGRC=$?
  set -e
  [ "$HGRC" -eq 0 ] || die "kunne ikkje lese menneskefilene (data-CLI exit $HGRC): $HG"
  say "pre-flight human games: $HG"
  case "$HG" in
    *" 0 samples")
      die "the human game files produced 0 usable samples — the run would be plain self-play. Check that the games replay on the engine: python -m sahkku_ai.data --spg <file> --out /tmp/x.npz --winner-policy" ;;
  esac
fi
mkdir -p "$(dirname "$OUT")" 2>/dev/null || die "cannot create the output directory for $OUT"

mkdir -p runs
LOG="runs/${RUN}_run.log"
say "starting (Ctrl-C stops after the current cycle; progress: tools/train_status.py $RUN -w)"
printf '\n' > "$LOG"
START=$(date +%s)
set -o pipefail
set +e                       # vi les exit-koden sjølve og skriv feilmeldinga
"${CMD[@]}" 2>&1 | tee -a "$LOG"
RC=$?
set -e
ELAPSED=$(( $(date +%s) - START ))

# ---- did it start at all? ------------------------------------------------
if [ "$RC" -ne 0 ]; then
  echo
  printf '\033[31m──── RUN FAILED ────────────────────────────────────────────\033[0m\n'
  printf 'exit code %s after %ss\n\n' "$RC" "$ELAPSED"
  CAUSE=""
  if grep -qi "CUDA out of memory\|out of memory" "$LOG"; then
    CAUSE="GPU out of memory. Each worker holds its own CUDA context (~0.4 GB), so lower --workers before touching the model; --games also helps."
  elif grep -qi "ModuleNotFoundError\|No module named" "$LOG"; then
    CAUSE="a Python module is missing from .venv. The exact install lines are in the header of this script."
  elif grep -qi "unrecognized arguments\|error: argument\|conflicting option" "$LOG"; then
    CAUSE="the pipeline rejected a flag: this wrapper and sahkku_ai/pipeline.py are out of step. Compare the wrapper's CMD with the pipeline's add_argument list."
  elif grep -qi "size mismatch\|load_state_dict\|Missing key" "$LOG"; then
    CAUSE="the start weights do not match the current feature vector. The widening step covers narrower nets; a WIDER checkpoint cannot be shrunk — start from a matching one."
  elif grep -qi "Address already in use\|Errno 98" "$LOG"; then
    CAUSE="a port is taken — another run or eval is probably still alive."
  elif grep -qi "No such file or directory" "$LOG"; then
    CAUSE="a path does not exist (checkpoint, shard directory or human game file)."
  elif grep -qi "Killed\|SIGKILL" "$LOG"; then
    CAUSE="the process was killed — usually the kernel OOM killer. Lower --workers."
  elif [ "$ELAPSED" -lt 30 ]; then
    CAUSE="it died within 30 s, before any training work — read the last lines below; the first exception is the real cause."
  fi
  if [ -n "$CAUSE" ]; then printf 'likely cause: %s\n\n' "$CAUSE"; fi
  printf 'last 25 lines of %s:\n' "$LOG"
  tail -25 "$LOG" | sed 's/^/  | /'
  echo
  printf '\033[31mNothing was published. Progress so far is kept in checkpoints/%s/ and runs/%s/.\033[0m\n' "$RUN" "$RUN"
  exit "$RC"
fi

# ---- exit 0 but no cycle finished (usually the budget ran out) -----------
if ! grep -q "\[c[0-9]*\] done in" "$LOG"; then
  echo
  printf '\033[33m──── STOPPED WITHOUT A FINISHED CYCLE ──────────────────────\033[0m\n'
  printf 'exit code 0, but no cycle completed in %ss — no new checkpoint was written.\n' "$ELAPSED"
  echo "If that was the time budget, raise --hours; if it stopped early, the tail says why:"
  tail -12 "$LOG" | sed 's/^/  | /'
  exit 1
fi

# ---- summary -------------------------------------------------------------
echo
DONE_N=$(grep -c "\[c[0-9]*\] done in" "$LOG" 2>/dev/null || echo 0)
say "done — ${DONE_N} syklus(ar) ferdige. Latest checkpoint:"
# `|| true`: utan ckpt. feilar ls med 2, og med set -e attende ville skriptet
# dødd her — nettopp i det høvet vi vil seie «skreiv ingenting»
CK=$(ls -1t checkpoints/"$RUN"/c*.pt 2>/dev/null | head -1 || true)
if [ -n "$CK" ]; then
  ls -1sh "$CK"
  say "next steps (export + engine validation):"
  cat <<EOF
  $PY -m sahkku_ai.export --net $CK --out $OUT --name "NAIS.x"
  SAHKKU_ENGINE="node $OUT.js" node ../SnartSahkku/tests/suci.test.js
  EVAL_NESSBY=1 node tools/eval_match.js "node $OUT.js" "node ../SnartSahkku/engines/nais2.js" 30 <seed>
EOF
else
  printf '\033[33mno c*.pt under checkpoints/%s — the run finished but wrote nothing; see %s\033[0m\n' "$RUN" "$LOG"
  exit 1
fi
