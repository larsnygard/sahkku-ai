#!/bin/bash
# Abiaslar: kva veg regel-luka i JS-motoren? Kvar køyring 16 parti mot FLOKK L3
# (klassiske reglar), same seed 31337 — same kast, samanliknbare parti.
cd ~/src/sahkku-ai
AN=analysis/blanka3_overnight
run() {
  local tag="$1"; shift
  echo "$(date '+%F %H:%M:%S') === $tag ===" >> "$AN/ab.log"
  env "$@" EVAL_SPG="$AN/ab_$tag.spg" \
    node tools/eval_match.js "node engines/blanka3-nn.js" "node vendor/sahkku-ref.js" 16 31337 \
    > "$AN/ab_$tag.log" 2>&1
  grep -E "SUM|poengsum" "$AN/ab_$tag.log" >> "$AN/ab.log"
}
run la0   EVAL_SETOPT_A="Lookahead=0"  EVAL_SETOPT_B="Level=3"
run deep2 EVAL_SETOPT_A="Deep=2"       EVAL_SETOPT_B="Level=3"
run la8   EVAL_SETOPT_A="Lookahead=8"  EVAL_SETOPT_B="Level=3"
echo "$(date '+%F %H:%M:%S') === AB DONE ===" >> "$AN/ab.log"
