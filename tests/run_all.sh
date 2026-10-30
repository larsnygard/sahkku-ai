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

echo "== 1/10 terningreglane som køyrbar spesifikasjon (X vekkjer, omkast, X>III>II) ="
$PY -m tests.test_rules

echo "== 2/10 features, aksjonsrom, FEN-runde, modellmasker =========================="
$PY -m tests.test_features

echo "== 3/10 regelparitet Python <-> JS (fuzz, ${FUZZ_GAMES:-40} parti) ============="
$PY tests/fuzz_parity.py "${FUZZ_GAMES:-40}"

echo "== 4/10 nett-paritet PyTorch <-> rein JS (planes, legal, policy, value) ======="
$PY -m tests.test_net_parity

echo "== 5/10 nettlesar-stien: modulen utan Node-globalar + vekter via loadWeights =="
node tests/test_browser_path.js "$ENGINE"

echo "== 6/10 S-UCI: den ekte GUI-testen frå SnartSahkku (heile parti) ============="
if [ -d "$SAHKKU" ]; then
  ( cd "$SAHKKU" && SAHKKU_ENGINE="node $ENGINE" node tests/suci.test.js )
else
  echo "  hoppar over: fann ikkje $SAHKKU (set SAHKKU=...)"
fi

echo "== 7/10 CLI-tekst i Python-verktøya er engelsk ================================"
$PY -m tests.test_cli_english

echo "== 8/10 badass-trainer: feilmeldingane når ei køyring ikkje startar ============"
$PY -m tests.test_trainer_faults

echo "== 9/10 tidsbudsjettet vert halde (ikkje berre «klokka er ute») ================"
$PY -m tests.test_budget
echo "== 10/10 taktikk-stilar (few/many, kingfwd/kingguard) ========================="
$PY -m tests.test_styles

echo
echo "ALT GRØNT"
