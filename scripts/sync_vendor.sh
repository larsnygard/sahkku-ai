#!/bin/sh
# Kopier dei kanoniske filene frå ein SnartSahkku-checkout inn i vendor/.
#
# vendor/ er VERBATIM-kopiar (ikkje avskrivne) av SnartSahkku si kode. Dette
# er den einaste staden sahkku-ai «ser» appen — resten av repoet køyrer HEILT
# åleine, også på treningsboksen utan app-checkout ved sidan av.
#
#   vendor/engine.js, vendor/fen.js    reglane (dommen) — injiserte i
#                                      engines/*.js ved eksport og brukte av
#                                      paritetstestane (1:1 med PWA-en)
#   vendor/cpu.js                      FLOKK, den klassiske CPU-en (nivå 1-3)
#   vendor/spg.js                      SPG-skriveren (EVAL_SPG i eval_match)
#   vendor/sahkku-ref.js               S-UCI-motoren rundt FLOKK — benke-
#                                      motstandaren (eval_match, bench_list,
#                                      pipeline --opponent, train.py classic3)
#   vendor/tests/suci.test.js          den EKTE GUI-testen: heile parti mot
#                                      ein S-UCI-motor (badass-trainer, gate)
#
# Dei to sistnemnde får require-stiene sine skrivne om til vendor/ — sjå sed
# under. Filenes innhald elles er ordrett som i appen.
#
# Køyr etter at SnartSahkku si kode er endra:
#   ./scripts/sync_vendor.sh [sti-til-SnartSahkku]
# og køyr testane:  ./tests/run_all.sh
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=${1:-$HERE/../SnartSahkku}
[ -d "$SRC/js" ] || { echo "finst ikkje: $SRC/js — klone SnartSahkku eller gje staden"; exit 1; }

cp "$SRC/js/engine.js" "$HERE/vendor/engine.js"
cp "$SRC/js/fen.js"    "$HERE/vendor/fen.js"
cp "$SRC/js/cpu.js"    "$SRC/js/spg.js" "$HERE/vendor/"

# sahkku-ref: same fil, men krev reglane frå vendor/ i staden for ../js/
sed "s|require('../js/|require('./|g" "$SRC/engines/sahkku-ref.js" \
  > "$HERE/vendor/sahkku-ref.js"

# suci.test.js: same test, men krev reglane frå vendor/ og ref-motoren der
# (testen ligg i vendor/tests/, så '../engine.js' = vendor/engine.js)
mkdir -p "$HERE/vendor/tests"
sed -e "s|require('../js/engine.js')|require('../engine.js')|" \
    -e "s|require('../js/fen.js')|require('../fen.js')|" \
    -e "s|path.join(__dirname, '..', 'engines', 'sahkku-ref.js')|path.join(__dirname, '..', 'sahkku-ref.js')|" \
    "$SRC/tests/suci.test.js" > "$HERE/vendor/tests/suci.test.js"

printf '%s\n' "vendor oppdatert frå $SRC:"
for f in engine.js fen.js cpu.js spg.js sahkku-ref.js tests/suci.test.js; do
  printf '  %-22s %5s linjer  %s\n' "$f" "$(wc -l < "$HERE/vendor/$f")" \
    "$(md5sum "$HERE/vendor/$f" | cut -c1-8)"
done
node -e "require('$HERE/vendor/engine.js'); require('$HERE/vendor/fen.js');
         console.log('vendor engine.js + fen.js: OK')"
echo "køyr testane:  ./tests/run_all.sh"
