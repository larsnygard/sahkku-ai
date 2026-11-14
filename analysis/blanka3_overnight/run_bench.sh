#!/bin/bash
# Reproduser bench-kommandoen for blanka3 (Level=3-gata) med fersk seed.
cd ~/src/sahkku-ai
AN=analysis/blanka3_overnight
log() { echo "$(date '+%F %H:%M:%S') $*" | tee -a "$AN/bench.log"; }

log "=== GATE-KOMMANDO: blanka3 vs classic3 FLOKK L3, 40 parti, seed 2026 ==="
EVAL_SETOPT_B=Level=3 \
  node tools/eval_match.js "node engines/blanka3-nn.js" "node vendor/sahkku-ref.js" 40 2026 \
  > "$AN/gate.log" 2>&1
tail -8 "$AN/gate.log" >> "$AN/bench.log"

log "=== SAME UANSEN SETOPT (Nesseby-variant), 40 parti, seed 2026 ==="
EVAL_NESSBY=1 \
  node tools/eval_match.js "node engines/blanka3-nn.js" "node vendor/sahkku-ref.js" 40 2026 \
  > "$AN/nessby.log" 2>&1
tail -8 "$AN/nessby.log" >> "$AN/bench.log"

log "=== BENCH REPRO DONE ==="
