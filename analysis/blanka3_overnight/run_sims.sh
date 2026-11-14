#!/bin/bash
# Overnight sim: blanka3 vs CPU(FLOKK L3), blanka3 mirror, CPU mirror. SPG recorded.
cd ~/src/sahkku-ai
AN=analysis/blanka3_overnight
mkdir -p "$AN"
TS() { date '+%F %H:%M:%S'; }

log() { echo "$(TS) $*" | tee -a "$AN/sim.log"; }

log "=== MATCH 1: blanka3 (A) vs classic3 FLOKK L3 (B), 30 games, seed 2026 ==="
EVAL_SPG="$AN/b3_vs_classic3.spg" \
  node tools/eval_match.js "node engines/blanka3-nn.js" "node vendor/sahkku-ref.js" 30 2026 \
  > "$AN/m1.log" 2>&1
tail -12 "$AN/m1.log" >> "$AN/sim.log"

log "=== MATCH 2: blanka3 mirror self-play, 24 games, seed 777 ==="
EVAL_NESSBY=1 EVAL_SPG="$AN/b3_mirror.spg" \
  node tools/eval_match.js "node engines/blanka3-nn.js" "node engines/blanka3-nn.js" 24 777 \
  > "$AN/m2.log" 2>&1
tail -12 "$AN/m2.log" >> "$AN/sim.log"

log "=== MATCH 3: classic3 CPU mirror, 14 games, seed 555 ==="
EVAL_NESSBY=0 EVAL_SETOPT_A="Level=3" EVAL_SETOPT_B="Level=3" EVAL_SPG="$AN/classic3_mirror.spg" \
  node tools/eval_match.js "node vendor/sahkku-ref.js" "node vendor/sahkku-ref.js" 14 555 \
  > "$AN/m3.log" 2>&1
tail -12 "$AN/m3.log" >> "$AN/sim.log"

log "=== ALL SIMS DONE ==="
