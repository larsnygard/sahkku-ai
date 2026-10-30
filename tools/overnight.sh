#!/usr/bin/env bash
# Full økt i éin kommando: tren → eksporter → commit → push.
#
# Kvifor eit skript: badass-trainer gjer jobben, men etterpå sit ein att med
# tre ting som må gjerast rett kvar gong — eksporter frå SISTE sjekkpunkt (med
# rett namn, so vektfila i .js stemmer), commit berre motor-filene (ikkje
# 12 MB-duplikat av .tar.xz), og push etter ein rebase (ein annan klone har
# nesten alltid pushe noko sidan sist).
#
#   tools/overnight.sh                          # NAIS.7-standard: 8 syklusar / 9 t
#   tools/overnight.sh --cycles 4 --hours 4     # kortare blokk
#   tools/overnight.sh --dry                    # vis kva som ville skje
#
# Alt etter --cycles/--hours/--games/--sims/--threads går rett til badass-trainer.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=nais7
INIT=engines/nais5-nn.pt
CYCLES=8
HOURS=9
GAMES=400
SIMS=45
EXTRA=()
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --run)     RUN="$2"; shift 2 ;;
    --init)    INIT="$2"; shift 2 ;;
    --cycles)  CYCLES="$2"; shift 2 ;;
    --hours)   HOURS="$2"; shift 2 ;;
    --games)   GAMES="$2"; shift 2 ;;
    --sims)    SIMS="$2"; shift 2 ;;
    --threads) EXTRA+=(--threads "$2"); shift 2 ;;
    --dry)     DRY=1; shift ;;
    *)         EXTRA+=("$1"); shift ;;
  esac
done

say() { printf '\033[36m▸\033[0m %s\n' "$*"; }
die() { printf '\033[31mFEIL:\033[0m %s\n' "$*" >&2; exit 1; }

[ -x .venv/bin/python3 ] || die "fann ingen .venv i $(pwd)"
[ -f "$INIT" ] || die "finn ikkje startvekta «$INIT»"

# ---- 1. hent andre sine commits fyrst, so pushen går gjennom etterpå ------
if [ "$DRY" = "0" ] && git rev-parse --git-dir >/dev/null 2>&1; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    die "arbeidstreet er ikkje reint — commit eller stash fyrst, elles blandar økta di eigne endringar inn i NAIS-commiten"
  fi
  say "hentar origin (rebase) før køyringa"
  git pull --rebase --autostash origin "$(git rev-parse --abbrev-ref HEAD)" >/dev/null \
    || die "git pull --rebase feila — fiks det no, ikkje etter 9 timar"
fi

# ---- 2. treninga ---------------------------------------------------------
say "trenar: $RUN i $CYCLES syklusar / maks $HOURS t (few på båe sider, Nesseby)"
DRYARG=()
[ "$DRY" = "1" ] && DRYARG=(--dry)
./badass-trainer.sh --run "$RUN" --init "$INIT" \
    --cycles "$CYCLES" --hours "$HOURS" --games "$GAMES" --sims "$SIMS" \
    --nesseby --nesseby-fraction 1.0 --style-both few "${EXTRA[@]}" "${DRYARG[@]}" \
    || die "treninga stoppa med feil (sjå loggen over) — ingen commit/push gjort"

# ---- 3. eksporter frå siste sjekkpunkt -----------------------------------
CK=$(ls -1t "checkpoints/$RUN"/c*.pt 2>/dev/null | head -1 || true)
[ -n "$CK" ] || die "fann ingen c*.pt under checkpoints/$RUN — køyringa skreiv ingen vekt"
say "eksporterer $CK → engines/${RUN}-nn"
.venv/bin/python3 -m sahkku_ai.export --net "$CK" --out "engines/${RUN}-nn" --name "${RUN^^}" >/dev/null
.venv/bin/python3 tests/test_net_parity.py "engines/${RUN}-nn.pt" --js "engines/${RUN}-nn.js" 2>&1 | tail -1

# ---- 4. commit + push (berre motor-filene) -------------------------------
if [ "$DRY" = "1" ]; then
  say "(dry) ville committa: engines/${RUN}-nn.js .pt .weights.json"
  exit 0
fi
git add "engines/${RUN}-nn.js" "engines/${RUN}-nn.pt" "engines/${RUN}-nn.weights.json"
if git diff --cached --quiet; then
  say "inga endring i motor-filene — hoppar over commit"
else
  N=$(grep -c "^\[c[0-9]*\] done in" "runs/${RUN}_run.log" 2>/dev/null || echo 0)
  S=$(python3 -c "
import json
try:
  rows=[json.loads(l) for l in open('runs/${RUN}/training_log.jsonl')]
  print(sum(r['new_samples'] for r in rows))
except Exception: print('?')")
  git commit -q -m "$RUN: $N syklusar, $S prøver ($CK)

Trena med --style-both few oppå $(basename "$INIT") — aktiveringsdoktrina frå
NAIS.6-duellane (few slo many 322-48, 87 %) på BÅE sider, so all data dreg same
veg. Eksportert frå $CK; nett-paritet køyrd."
  say "committa $RUN"
fi
for i in 1 2 3; do
  if git push -q origin "$(git rev-parse --abbrev-ref HEAD)"; then say "pusht"; break; fi
  say "push feila (forsøk $i) — prøver rebase og push att"
  git pull --rebase --autostash origin "$(git rev-parse --abbrev-ref HEAD)" || true
  sleep 5
done

# ---- 5. kva no? ----------------------------------------------------------
say "ferdig. Morgonporten (køyr der det er GPU):"
cat <<EOF
  .venv/bin/python tools/train_status.py $RUN          # kor mange syklusar vart det
  .venv/bin/python tools/style_report.py $RUN          # løysingar/100 halvtrekk (skal liggje under ~6)
  EVAL_NESSBY=1 node tools/eval_match.js "node engines/${RUN}-nn.js" \\
      "node engines/nais5-nn.js" 30 <seed>              # ≥55 % → standardmotor
  EVAL_SETOPT_B=Level=3 node tools/eval_match.js "node engines/${RUN}-nn.js" \\
      "node engines/sahkku-ref.js" 40 <seed>            # ≥90 %
EOF
