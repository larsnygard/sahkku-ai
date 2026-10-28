#!/usr/bin/env bash
# badass-trainer.sh — tren sáhkku-nettet på denne maskina.
#
#   ./badass-trainer.sh                 # 2 timar, held fram frå AIL0.9 → ail10
#   ./badass-trainer.sh --hours 6       # lengre økt
#   ./badass-trainer.sh --games 600     # fleire parti per syklus
#   ./badass-trainer.sh --run ail11 --init engines/ail10-nn.pt
#   ./badass-trainer.sh --dry           # vis berre kva som ville køyrt
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
GAMES="400"
SIMS="45"
WORKERS=""
STEPS="900"
BATCH="256"
LR="3.0e-4"
LEAFBATCH="8"
WINDOW="2"
SEED="81"
MAXPLIES="2500"
OUT=""
DRY="0"

die() { printf '\033[31mFEIL:\033[0m %s\n' "$*" >&2; exit 1; }
say() { printf '\033[36m▸\033[0m %s\n' "$*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --run)        RUN="$2"; shift 2 ;;
    --init)       INIT="$2"; shift 2 ;;
    --hours)      HOURS="$2"; shift 2 ;;
    --games)      GAMES="$2"; shift 2 ;;
    --sims)       SIMS="$2"; shift 2 ;;
    --workers)    WORKERS="$2"; shift 2 ;;
    --steps)      STEPS="$2"; shift 2 ;;
    --lr)         LR="$2"; shift 2 ;;
    --leaf-batch) LEAFBATCH="$2"; shift 2 ;;
    --seed)       SEED="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
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

# ---- maskin -------------------------------------------------------------
THREADS=$(nproc)
CORES=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[ "${CORES:-0}" -gt 0 ] || CORES=$(( THREADS / 2 > 0 ? THREADS / 2 : 1 ))
[ -n "$WORKERS" ] || WORKERS="$CORES"       # éin arbeidar per fysisk kjerne

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
    say "VRAM ${VRAM_GB} GB → arbeidarar skorne frå $WORKERS til $MAXW (kvar prosess held ein CUDA-kontekst)"
    WORKERS="$MAXW"
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
  [ "$WORKERS" -gt "$CORES" ] && WORKERS="$CORES"
fi

# CPU utan GPU er ~10x tregare per avgjerd: fleire arbeidarar gjev meir data
# per time, men kvar arbeidar treng ein kjerne.
RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)

say "maskin:  $CORES fysiske kjernar / $THREADS trådar, ${RAM_GB} GB RAM"
if [ "$DEVICE" = "cuda" ]; then say "gpu:     ${GPU}"; else say "gpu:     ingen (CPU)"; fi
say "køyring: $RUN frå $INIT"
say "plan:    ${HOURS} t · ${GAMES} parti/syklus · ${WORKERS} arbeidarar · ${SIMS} sims · $DEVICE"

# Maskinvaretilpassing: minne til 8 arbeidarar er ~1-2 GB; 64 GB er rikeleg.
if [ "$RAM_GB" -lt 16 ]; then
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
SECS=$($PY -c "print(int(float('$HOURS')*3600))")

CMD=(
  "$PY" -m sahkku_ai.pipeline
  --run "$RUN" --start-cycle "$NEXT" --cycles 999
  --init "$INIT"
  --games "$GAMES" --workers "$WORKERS" --sims "$SIMS" --leaf-batch "$LEAFBATCH"
  --steps "$STEPS" --batch "$BATCH" --lr "$LR" --warmup 100
  --ch 64 --blocks 6 --fc 128 --device "$DEVICE" --window "$WINDOW"
  --seed "$SEED" --max-plies "$MAXPLIES" --max-seconds "$SECS"
  --log-every 300 --save-every 450 --eval-games 0
  --out "$OUT"
)

if [ "$DRY" = "1" ]; then
  printf '\n%s\n' "${CMD[*]}"
  exit 0
fi

mkdir -p runs
say "startar (Ctrl-C stoppar etter inneverande syklus; framdrift: tools/train_status.py $RUN -w)"
printf '\n' > "runs/${RUN}_run.log"
"${CMD[@]}" 2>&1 | tee -a "runs/${RUN}_run.log"

# ---- oppsummering --------------------------------------------------------
echo
say "ferdig. Siste sjekkpunkt:"
ls -1t checkpoints/"$RUN"/c*.pt 2>/dev/null | head -1 || say "  (ingen c*.pt — sjå loggen)"
say "neste steg (eksport + motorvalidering):"
cat <<EOF
  $PY -m sahkku_ai.export --net \$(ls -1t checkpoints/$RUN/c*.pt | head -1) \\
      --out $OUT --name "AIL${RUN#ail}"
  node ../SnartSahkku/tests/suci.test.js   # krev S-UCI-testen (set SAHKKU_ENGINE)
EOF
