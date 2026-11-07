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

echo "== 1/12 terningreglane som køyrbar spesifikasjon (X vekkjer, omkast, X>III>II) ="
$PY -m tests.test_rules

echo "== 2/12 features, aksjonsrom, FEN-runde, modellmasker =========================="
$PY -m tests.test_features

echo "== 3/12 regelparitet Python <-> JS (fuzz, ${FUZZ_GAMES:-40} parti) ============="
$PY tests/fuzz_parity.py "${FUZZ_GAMES:-40}"

echo "== 4/12 nett-paritet PyTorch <-> rein JS (planes, legal, policy, value) ======="
$PY -m tests.test_net_parity

echo "== 5/12 nettlesar-stien: modulen utan Node-globalar + vekter via loadWeights =="
node tests/test_browser_path.js "$ENGINE"

echo "== 6/12 S-UCI: den ekte GUI-testen (heile parti) — frå appen eller vendor/ =="
if [ -d "$SAHKKU" ]; then
  ( cd "$SAHKKU" && SAHKKU_ENGINE="node $ENGINE" node tests/suci.test.js )
elif [ -f "$ROOT/vendor/tests/suci.test.js" ]; then
  # same test, vendor-kopi (scripts/sync_vendor.sh) — treningsboksen heilt åleine
  ( cd "$ROOT/vendor/tests" && SAHKKU_ENGINE="node $ENGINE" node suci.test.js )
else
  echo "  hoppar over: verken $SAHKKU eller vendor/tests/suci.test.js"
fi

echo "== 7/12 CLI-tekst i Python-verktøya er engelsk ================================"
$PY -m tests.test_cli_english

echo "== 8/12 badass-trainer: feilmeldingane når ei køyring ikkje startar ============"
$PY -m tests.test_trainer_faults

echo "== 9/12 tidsbudsjettet vert halde (ikkje berre «klokka er ute») ================"
# pipeline-løkka tek GPU dersom ho finst; testen verifiserer TIDSLOGIKK, ikkje
# fart — på CPU køyrer han uansett kva ein annan prosess (trening) har tatt av VRAM
CUDA_VISIBLE_DEVICES="" $PY -m tests.test_budget
echo "== 10/12 taktikk-stilar (few/many, kingfwd/kingguard) ========================="
$PY -m tests.test_styles

echo "== 11/12 indektert speltdatabase (SCID-metoden: .sci + .sgt) =================="
$PY -m tests.test_gamedb

echo "== 12/12 treningsløkka: train/eval-modus, val, minstemål ====================="
CUDA_VISIBLE_DEVICES="" $PY -m tests.test_train
CUDA_VISIBLE_DEVICES="" $PY -m tests.test_resume

echo
echo "ALT GRØNT"
