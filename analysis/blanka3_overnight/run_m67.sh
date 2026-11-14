#!/bin/bash
cd ~/src/sahkku-ai
AN=analysis/blanka3_overnight
echo "$(date '+%F %H:%M:%S') === M6: classic-reglar, CPU Level=2, 16 parti, seed 31337 ===" >> "$AN/sim.log"
EVAL_SETOPT_B="Level=2" EVAL_SPG="$AN/b3_vs_classic3_L2.spg" \
  node tools/eval_match.js "node engines/blanka3-nn.js" "node vendor/sahkku-ref.js" 16 31337 \
  > "$AN/m6.log" 2>&1
grep -E "SUM|poengsum" "$AN/m6.log" >> "$AN/sim.log"
echo "$(date '+%F %H:%M:%S') === M7: Nesseby, CPU Level=3, 16 parti, seed 31337 ===" >> "$AN/sim.log"
EVAL_NESSBY=1 EVAL_SETOPT_B="Level=3" EVAL_SPG="$AN/b3_vs_classic3_L3_nesseby2.spg" \
  node tools/eval_match.js "node engines/blanka3-nn.js" "node vendor/sahkku-ref.js" 16 31337 \
  > "$AN/m7.log" 2>&1
grep -E "SUM|poengsum|FEIL" "$AN/m7.log" | head -20 >> "$AN/sim.log"
echo "$(date '+%F %H:%M:%S') === M67 DONE ===" >> "$AN/sim.log"
