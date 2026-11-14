#!/usr/bin/env bash
# blanka4 — første økt etter nattleginga (analysis/blanka3_overnight/FINDINGS.md).
#
# Kvar plan-punkt er eit flagg her, so ei køyring kan etterprøvast:
#   P1  --z-blend 0.75 --z-clip 0.9   avmetta verdihovud (55 % |v|>0.9 → mål <30 %)
#   P2  --nesseby --nesseby-fraction 0.5  blanda reglar (eitt nett, g[15] vel)
#       --train-opponent classic3 --opponent-fraction 0.5   halvparten mot L3
#   P3  --side-aug                    Giera-sida lært av Menn-sida (symmetry.py,
#                                     sanning prova i tests/test_side_flip.py)
#   P4  morgonporten nedanfor: to regelvariantar, spegel per side, dronningtal,
#       value_health — alle som tal, ikkje som skjønn.
#
#   tools/overnight-blanka4.sh                # 8 syklusar / 8 t
#   tools/overnight-blanka4.sh --dry          # vis kva som ville skje
#   tools/overnight-blanka4.sh --cycles 4 --hours 4
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=blanka4
INIT=engines/blanka3-nn.pt
CYCLES=8
HOURS=8
GAMES=400
SIMS=45
STEPS=900
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
    --steps)   STEPS="$2"; shift 2 ;;
    --dry)     DRY=1; shift ;;
    *)         EXTRA+=("$1"); shift ;;
  esac
done

say() { printf '\033[36m▸\033[0m %s\n' "$*"; }
die() { printf '\033[31mFEIL:\033[0m %s\n' "$*" >&2; exit 1; }

[ -x .venv/bin/python3 ] || die "fann ingen .venv i $(pwd)"
[ -f "$INIT" ] || die "finn ikkje startvekta «$INIT»"

# Symmetrien må vere sann FØR han vert brukt til treningsdata.
say "sjekkar side-symmetrien (tests/test_side_flip.py)"
.venv/bin/python3 tests/test_side_flip.py || die "side-speglinga er ikkje ei symmetri — ikkje køyr med --side-aug"

if [ "$DRY" = "0" ] && git rev-parse --git-dir >/dev/null 2>&1; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    die "arbeidstreet er ikkje reint — commit eller stash fyrst"
  fi
  say "hentar origin (rebase) før køyringa"
  git pull --rebase --autostash origin "$(git rev-parse --abbrev-ref HEAD)" >/dev/null \
    || die "git pull --rebase feila"
fi

DRYARG=()
[ "$DRY" = "1" ] && DRYARG=(--dry)
say "trenar: $RUN i $CYCLES syklusar / maks $HOURS t (blanda reglar, 50 % classic3, side-aug, avmetta z)"
./badass-trainer.sh --run "$RUN" --init "$INIT" \
    --cycles "$CYCLES" --hours "$HOURS" --games "$GAMES" --sims "$SIMS" --steps "$STEPS" \
    --nesseby --nesseby-fraction 0.5 \
    --train-opponent classic3 --train-opponent-level 3 --opponent-fraction 0.5 \
    --z-blend 0.75 --z-clip 0.9 --side-aug \
    "${EXTRA[@]}" "${DRYARG[@]}" \
    || die "treninga stoppa med feil — ingen commit/push gjort"

if [ "$DRY" = "1" ]; then
  say "(dry) ville committa: engines/${RUN}-nn.js .pt .weights.json"
  exit 0
fi

CK=$(ls -1t "checkpoints/$RUN"/c*.pt 2>/dev/null | head -1 || true)
[ -n "$CK" ] || die "fann ingen c*.pt under checkpoints/$RUN"
say "eksporterer $CK → engines/${RUN}-nn"
.venv/bin/python3 -m sahkku_ai.export --net "$CK" --out "engines/${RUN}-nn" --name "${RUN^^}" >/dev/null
.venv/bin/python3 tests/test_net_parity.py "engines/${RUN}-nn.pt" --js "engines/${RUN}-nn.js" 2>&1 | tail -1

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

Trena etter nattleginga (analysis/blanka3_overnight/FINDINGS.md, P1–P3):
z_blend 0.75/z_clip 0.9 (avmetta verdihovud), blanda reglar (nesseby 0.5),
50 % kampar mot FLOKK L3, side-augmentering (symmetry.py). Eksportert frå $CK;
nett-paritet køyrd."
  say "committa $RUN"
fi
for i in 1 2 3; do
  if git push -q origin "$(git rev-parse --abbrev-ref HEAD)"; then say "pusht"; break; fi
  say "push feila (forsøk $i) — prøver rebase og push att"
  git pull --rebase --autostash origin "$(git rev-parse --abbrev-ref HEAD)" || true
  sleep 5
done

say "ferdig. Morgonporten (køyr der det er GPU):"
cat <<EOF
  .venv/bin/python tools/train_status.py $RUN
  # P0: begge regelvariantane, same CPU — krav ≥55 % på KVAR av dei
  node tools/bench_list.py $RUN "engines/${RUN}-nn.js" "classic3c:40,classic3n:40" 2026
  # P1: verdihovudet — krav <30 %
  .venv/bin/python tools/value_health.py --tag $RUN --side-split \\
      runs/eval/${RUN}_classic3c.spg engines/${RUN}-nn.pt
  # P3: spegel — krav ≥48 % på BÅE sider (blanka3: 83/17)
  EVAL_NESSBY=1 node tools/eval_match.js "node engines/${RUN}-nn.js" \\
      "node engines/${RUN}-nn.js" 24 777
  # P4: dronninga — krav: eiga dronning teken i <50 % av partia
  .venv/bin/python tools/punish_rate.py runs/eval/${RUN}_classic3c.spg
  .venv/bin/python tools/queen_autopsy.py runs/eval/${RUN}_classic3c.spg \\
      engines/${RUN}-nn.pt 4
  # ABIASLAR på det nye nettet (LA0/LA4/LA8/Deep2) — no skal djupn VERKE
  bash analysis/blanka3_overnight/run_ab.sh engines/${RUN}-nn.js $RUN
EOF
