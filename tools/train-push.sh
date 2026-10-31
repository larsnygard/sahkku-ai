#!/usr/bin/env bash
# Heile økta på treningsboksen: tren → eksporter → lokale portar → commit → push.
# Porten på den andre sida (tools/gate-publish.sh, køyrd som cron) plukkar opp
# motoren når han ligg på origin/main, benkar han og deployar han.
#
# Kvifor eit eige skript og ikkje overnight.sh: overnight.sh trenar
# AKTIVERINGSDOKTRINA FRA NAIS.6 (--style-both few på båe sider). Det var feil
# veg — `few` er målt til 1.14 utløyste soldatar ved trekk 40, og NAIS.7 er
# trena nettopp slik, so ho spelar den doktrina godt. Her er dietten snudd:
# nettsida vert tvinga inn i `hold` (aktiver fleire, hald deg utanfor
# rekkjevidd) medan straffaren tek det som stikk fram.
#
#   tools/train-push.sh                        # NAIS.8: 10 syklusar × 200 parti
#   tools/train-push.sh --hours 3              # kort blokk om kvelden
#   tools/train-push.sh --cycles 10 --hours 12 # nattblokke: HELD FRAM der ho slapp
#   tools/train-push.sh --dry                  # vis kommandoen, gjer ingenting
#   tools/train-push.sh --no-push              # tren + eksporter, ikkje commit
#
# Blokker: --cycles er SISTE syklusnummer, ikkje «kor mange til». Køyrer du
# same kommandoen att, held pipeline fram frå siste fullførte syklus
# (checkpoints/<run>/cN.pt) — ikkje frå --init. So kort blokk om kvelden og
# lang blokk om natta er same kommando med anna tidsbudsjett, og kvar blokk
# vert pusha og publisert for seg.
#
# Alt ukjent vert sendt rett til badass-trainer.sh.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

RUN=nais8
INIT=engines/nais7-nn.pt       # sterkaste modellen; vert breidd ut til 18 globale
CYCLES=10
HOURS=0                        # 0 = inga tidsgrense, --cycles avgjer
GAMES=200
SIMS=45
STEPS=450
OPP=punisher
HAND=hold
EXTRA=()
DRY=0
NO_PUSH=0

while [ $# -gt 0 ]; do
  case "$1" in
    --run)        RUN="$2"; shift 2 ;;
    --init)       INIT="$2"; shift 2 ;;
    --cycles)     CYCLES="$2"; shift 2 ;;
    --hours)      HOURS="$2"; shift 2 ;;
    --games)      GAMES="$2"; shift 2 ;;
    --sims)       SIMS="$2"; shift 2 ;;
    --steps)      STEPS="$2"; shift 2 ;;
    --opponent)   OPP="$2"; shift 2 ;;
    --hand-style) HAND="$2"; shift 2 ;;
    --threads)    EXTRA+=(--threads "$2"); shift 2 ;;
    --dry)        DRY=1; shift ;;
    --no-push)    NO_PUSH=1; shift ;;
    *)            EXTRA+=("$1"); shift ;;
  esac
done

say() { printf '\033[36m▸\033[0m %s\n' "$*"; }
die() { printf '\033[31mFEIL:\033[0m %s\n' "$*" >&2; exit 1; }

# ---- 0. førehandssjekk: alt som kan gå gale no, gjer det no ---------------
[ -x .venv/bin/python3 ] || die "fann ingen .venv i $(pwd) — køyr frå sahkku-ai-repoet"
[ -f "$INIT" ]           || die "finn ikkje startvekta «$INIT»"
command -v git >/dev/null || die "git manglar"
git rev-parse --git-dir >/dev/null 2>&1 || die "dette er ikkje eit git-arbeidstre"
BR=$(git rev-parse --abbrev-ref HEAD)
[ "$BR" = "main" ] || die "står på «$BR», ikkje main — porten les berre origin/main"
if [ "$NO_PUSH" = "0" ] && [ "$DRY" = "0" ]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    die "arbeidstreet er ikkje reint — commit eller stash fyrst, elles blandar økta di eigne endringar inn i motor-commiten"
  fi
  git ls-remote --exit-code origin >/dev/null 2>&1 \
    || die "når ikkje origin — sjekk nøkkel/nett FØR treninga, ikkje etter 3 timar"
fi

# Same kode på begge maskinene: elles trenar boksen med gamle trekk (t.d. utan
# g[16]/g[17]) og porten på andre sida lastar noko anna enn det som vart trena.
if [ "$DRY" = "0" ] && [ "$NO_PUSH" = "0" ]; then
  git pull -q --rebase --autostash origin "$BR" \
    || die "git pull --rebase feila — fiks det no, ikkje etter tre timar"
fi

# Kodevakt: dei nye trekka og straffaren MÅ finnast i arbeidstreet.
.venv/bin/python3 - <<'PY' || die "boksen køyrer gammal kode (manglar punisher eller g[16]/g[17]) — git pull først"
import sys
from sahkku_ai import features as FT
import sahkku_ai.punisher  # noqa: F401
sys.exit(0 if FT.N_GLOBAL == 18 else 1)
PY

.venv/bin/python3 - "$INIT" <<'PY' || die "startvekta let seg ikkje laste (manglar ho ei fil?)"
import sys
from sahkku_ai.model import Net
Net.load(sys.argv[1], map_location="cpu")
PY

GPU=$(.venv/bin/python3 -c "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null)
say "startvekt: $INIT ($(.venv/bin/python3 -c "
from sahkku_ai.model import Net
n=Net.load('$INIT', map_location='cpu'); print(n.meta.get('run','?'), '→', n.meta.get('steps','?'), 'steg')" 2>/dev/null))"
say "diett: $OPP som motstandar, nettsida tvinga inn i «$HAND»"
say "plan: $CYCLES syklusar × $GAMES parti, $SIMS sims, $STEPS steg/syklus, $GPU"
if [ "$HOURS" != "0" ]; then
  say "tidsbudsjett: $HOURS t (ein syklus kan fullførast etter at budsjettet er brukt)"
else
  say "tidsbudsjett: ingen — --cycles $CYCLES avgjer"
fi

[ "$DRY" = "1" ] && say "(dry) ingen trening, ingen commit, ingen push"

# ---- 1. treninga ---------------------------------------------------------
DRYARG=(); [ "$DRY" = "1" ] && DRYARG=(--dry)
./badass-trainer.sh --run "$RUN" --init "$INIT" \
    --cycles "$CYCLES" --hours "$HOURS" --games "$GAMES" --sims "$SIMS" --steps "$STEPS" \
    --nesseby --nesseby-fraction 1.0 \
    --train-opponent "$OPP" --hand-style "$HAND" \
    "${EXTRA[@]}" "${DRYARG[@]}" \
  || die "treninga stoppa (sjå loggen over) — ingen commit/push gjort"

[ "$DRY" = "1" ] && exit 0

# ---- 2. eksporter frå SISTE sjekkpunkt -----------------------------------
CK=$(ls -1t "checkpoints/$RUN"/c*.pt 2>/dev/null | head -1 || true)
[ -n "$CK" ] || die "fann ingen c*.pt under checkpoints/$RUN — økta skreiv inga vekt"
say "eksporterer $CK → engines/${RUN}-nn"
.venv/bin/python3 -m sahkku_ai.export --net "$CK" --out "engines/${RUN}-nn" --name "${RUN^^}" >/dev/null \
  || die "eksporten feila"

# ---- 3. lokale portar (publiser aldri utan dei) --------------------------
PAR=$(.venv/bin/python3 tests/test_net_parity.py "engines/${RUN}-nn.pt" --js "engines/${RUN}-nn.js" 2>&1 | tail -1)
say "nett-paritet: $PAR"
case "$PAR" in *OK*) ;; *) die "nett-paritet feila — .pt og .js stemmer ikkje; ikkje pusht" ;; esac

SUCI=$( { printf 'uci\nposition startpos\ngo depth 1\nquit\n' | node "engines/${RUN}-nn.js"; } 2>&1 | grep -c "uciok")
[ "$SUCI" -ge 1 ] || die "motoren svarar ikkje på S-UCI (uciok manglar); ikkje pusht"
say "S-UCI: svarar"

# ---- 4. rapport (porten legg benken til same fila) -----------------------
.venv/bin/python3 - "$RUN" "$CK" "$INIT" "$OPP" "$HAND" <<'PY' || true
import json, pathlib, sys, torch
run, ck, init, opp, hand = sys.argv[1:6]
rows = []
p = pathlib.Path(f"runs/{run}/training_log.jsonl")
if p.exists():
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
samples = sum(r.get("new_samples", 0) for r in rows)
# val-tapet ligg i sjekkpunktet sitt meta, ikkje i loggen
try:
    meta = torch.load(ck, map_location="cpu", weights_only=False).get("meta", {})
except Exception:
    meta = {}
best = meta.get("best_val")
body = [f"# {run.upper()} — treningsrapport", "",
        f"- startvekt: `{init}`",
        f"- sjekkpunkt: `{ck}`",
        f"- diett: motstandar **{opp}**, nettsida tvinga inn i **{hand}**",
        f"- syklusar fullførte: {len(rows)}",
        f"- prøver: {samples}",
        (f"- val-tap (beste): {best:.3f}" if isinstance(best, float) else "- val-tap: ?"),
        "", "Benken og publiseringa kjem frå porten (tools/gate-publish.sh).", ""]
pathlib.Path(f"runs/{run}_report.md").write_text("\n".join(body))
print("\n".join(body))
PY

# ---- 5. commit + push (berre motor-filene og rapporten) ------------------
if [ "$NO_PUSH" = "1" ]; then
  say "--no-push: hoppar over commit/push. Motor ligg i engines/${RUN}-nn.*"
  exit 0
fi

git add "engines/${RUN}-nn.js" "engines/${RUN}-nn.pt" "engines/${RUN}-nn.weights.json" "runs/${RUN}_report.md"
if git diff --cached --quiet; then
  say "inga endring i motor-filene — ingenting å committe"
else
  N=$(grep -c "^\[c[0-9]*\] done in" "runs/${RUN}_run.log" 2>/dev/null); N=${N:-0}
  git commit -q -m "$RUN: $N syklusar, $GAMES parti/syklus ($CK)

Trena med --train-opponent $OPP og --hand-style $HAND oppå $(basename "$INIT"):
nettsida vert tvinga inn i «$HAND» (aktiver fleire soldatar, hald dei utanfor
motstandaren si rekkjevidd) medan motstandaren tek det som stikk fram.
Dette snur NAIS.7-dietten (--style-both few, som lærte ~1 utløyst soldat);
same vektene, motsett doktrine. g[16]/g[17] (bonderisiko) er nye og startar på
null, so nettet spelar som før til det lærer noko nyttigt.

Eksportert frå $CK; nett-paritet køyrd lokalt; S-UCI svarar. Benk og deploy
gjer porten på Nox-sida når denne committen ligg på origin/main." \
    || die "git commit feila"
  say "committa $RUN"
fi

for i in 1 2 3; do
  if git push -q origin "$BR"; then say "pusht til origin/$BR — porten tek over innan 20 min"; break; fi
  say "push feila (forsøk $i) — rebaser og prøver att"
  git pull -q --rebase --autostash origin "$BR" || true
  sleep 5
  [ "$i" = "3" ] && die "push feila tre gonger — committen ligg lokalt, køyr: git push origin $BR"
done

cat <<EOF

Neste steg (automatisk på Nox-sida):
  1. porten ser engines/${RUN}-nn.pt på origin/main
  2. S-UCI + nett-paritet, så benk: Nesseby mot nais7 (30) og klassisk L3 (40)
  3. vert standardmotor berre om ho slår nais7 (≥ 55 %), elles valfri
  4. service-worker-cache + versjon vert bumpa, deploy via sendsahkku.sh
  5. rapport i runs/${RUN}_report.md og i chatten

Om du vil sjå framgangen sjølv:
  .venv/bin/python3 tools/train_status.py $RUN
EOF
