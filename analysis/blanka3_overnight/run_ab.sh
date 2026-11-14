#!/bin/bash
# Abiaslar: kva veg regel-luka i JS-motoren? Kvar køyring 16 parti mot FLOKK L3
# (klassiske reglar), same seed 31337 — same kast, samanliknbare parti.
#
#   bash analysis/blanka3_overnight/run_ab.sh [MOTOR_JS] [TAG]
# t.d.   bash analysis/blanka3_overnight/run_ab.sh engines/blanka4-nn.js blanka4
cd ~/src/sahkku-ai
ENG=${1:-engines/blanka3-nn.js}
TAG=${2:-blanka3}
AN=analysis/blanka3_overnight
run() {
  local tag="$1"; shift
  echo "$(date '+%F %H:%M:%S') === $TAG/$tag ===" >> "$AN/ab.log"
  env "$@" EVAL_SPG="$AN/ab_${TAG}_$tag.spg" \
    node tools/eval_match.js "node $ENG" "node vendor/sahkku-ref.js" 16 31337 \
    > "$AN/ab_${TAG}_$tag.log" 2>&1
  grep -E "SUM|poengsum" "$AN/ab_${TAG}_$tag.log" >> "$AN/ab.log"
}
run la0   EVAL_SETOPT_A="Lookahead=0"  EVAL_SETOPT_B="Level=3"
run base  EVAL_SETOPT_B="Level=3"
run la8   EVAL_SETOPT_A="Lookahead=8"  EVAL_SETOPT_B="Level=3"
run deep2 EVAL_SETOPT_A="Deep=2"       EVAL_SETOPT_B="Level=3"
echo "$(date '+%F %H:%M:%S') === $TAG AB DONE ===" >> "$AN/ab.log"
