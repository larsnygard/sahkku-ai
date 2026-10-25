#!/bin/sh
# Kopier dei kanoniske regelfilene frå ein SnartSahkku-checkout inn i vendor/.
#
# vendor/engine.js og vendor/fen.js er VERBATIM-kopiar (ikkje avskrivne) av
# regelmotoren i SnartSahkku. Dei blir
#   * brukt av paritetstestane (tests/fuzz_parity.py → tools/parity_probe.js)
#   * INJISERTE i engines/sahkku-nn.js ved eksport, slik at den nevrale
#     motoren dømmer etter nøyaktig same regler som PWA-en.
#
# Køyr etter at SnartSahkku sine regler er endra:
#   ./scripts/sync_vendor.sh [sti-til-SnartSahkku]
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=${1:-$HERE/../SnartSahkku}

for f in engine.js fen.js; do
  cp "$SRC/js/$f" "$HERE/vendor/$f"
done
printf '%s\n' "vendor oppdatert frå $SRC/js:" \
  "  engine.js $(wc -l < "$HERE/vendor/engine.js") linjer" \
  "  fen.js    $(wc -l < "$HERE/vendor/fen.js") linjer"
echo "køyr testane på nytt:  .venv/bin/python -m tests.test_features && node -e 'require(\"./vendor/engine.js\")'"
