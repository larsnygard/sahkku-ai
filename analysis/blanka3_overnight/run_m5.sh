#!/bin/bash
cd ~/src/sahkku-ai
AN=analysis/blanka3_overnight
echo "$(date '+%F %H:%M:%S') === M5: blanka3 vs classic3 L3, 16 games, SPG, seed 31337 ===" >> "$AN/sim.log"
EVAL_SETOPT_B=Level=3 EVAL_SPG="$AN/b3_vs_classic3_L3.spg" \
  node tools/eval_match.js "node engines/blanka3-nn.js" "node vendor/sahkku-ref.js" 16 31337 \
  > "$AN/m5.log" 2>&1
tail -6 "$AN/m5.log" >> "$AN/sim.log"
echo "$(date '+%F %H:%M:%S') === M5 DONE ===" >> "$AN/sim.log"
