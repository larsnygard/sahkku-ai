#!/bin/sh
# Køyr alle testane i repoet (reglar, regelparitet, nett-paritet, S-UCI/GUI-test).
#
#   ./tests/run_all.sh              # 40 fuzz-parti
#   FUZZ_GAMES=200 ./tests/run_all.sh
#
# Miljø: PY=.venv/bin/python (standard), SAHKKU=<sti til SnartSahkku>
set -eu
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PY:-.venv/bin/python}
SAHKKU=${SAHKKU:-$ROOT/../SnartSahkku}
ENGINE=$ROOT/engines/sahkku-nn.js

echo "== 1/6 terningreglane som køyrbar spesifikasjon (X vekkjer, omkast, X>III>II) ="
$PY -m tests.test_rules

echo "== 2/6 features, aksjonsrom, FEN-runde, modellmasker =========================="
$PY -m tests.test_features

echo "== 3/6 regelparitet Python <-> JS (fuzz, ${FUZZ_GAMES:-40} parti) ============="
$PY tests/fuzz_parity.py "${FUZZ_GAMES:-40}"

echo "== 4/6 nett-paritet PyTorch <-> rein JS (planes, legal, policy, value) ======="
$PY -m tests.test_net_parity

echo "== 5/6 nettlesar-stien: modulen utan Node-globalar + vekter via loadWeights =="
node tests/test_browser_path.js "$ENGINE"

echo "== 6/6 S-UCI: den ekte GUI-testen frå SnartSahkku (heile parti) ============="
if [ -d "$SAHKKU" ]; then
  ( cd "$SAHKKU" && SAHKKU_ENGINE="node $ENGINE" node tests/suci.test.js )
else
  echo "  hoppar over: fann ikkje $SAHKKU (set SAHKKU=...)"
fi

echo
echo "ALT GRØNT"
