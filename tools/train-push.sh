#!/usr/bin/env bash
# Heile økta på treningsboksen, SYKLUS FOR SYKLUS:
#   tren éin syklus → eksporter → lokale portar → commit → push → neste syklus
# Porten på den andre sida (tools/gate-publish.sh, køyrd som cron) plukkar opp
# kvar push, benkar han og deployar han. Framgangen er difor synleg undervegs,
# og ei krasjande natt mistar berre den eine syklusen.
#
# Kvifor eit eige skript og ikkje overnight.sh: overnight.sh trenar
# AKTIVERINGSDOKTRINA FRÅ NAIS.6 (--style-both few på båe sider). Det var feil
# veg — `few` er målt til 1.14 utløyste soldatar ved trekk 40, og NAIS.7 er
# trena nettopp slik, so ho spelar den doktrina godt. Her er dietten snudd:
# nettsida vert tvinga inn i `hold` (aktiver fleire, hald deg utanfor
# rekkjevidd) medan straffaren tek det som stikk fram.
#
#   tools/train-push.sh                        # NAIS.8: syklus 1..10
#   tools/train-push.sh --cycles 3             # tre syklusar (~2,8 t)
#   tools/train-push.sh --cycles 10 --hours 12 # natt: held fram der ho slapp
#   tools/train-push.sh --dry                  # vis planar, gjer ingenting
#   tools/train-push.sh --no-push              # tren + eksporter, ikkje commit
#
# --cycles er SISTE syklusnummer. Køyrer du same kommandoen att, held han fram
# frå siste fullførte syklus (checkpoints/<run>/cN.pt), ikkje frå --init.
# Alt ukjent vert sendt rett til badass-trainer.sh.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

RUN=nais8
INIT=engines/nais7-nn.pt       # sterkaste modellen; vert breidd ut til 18 globale
CYCLES=10                      # siste syklusnummer
HOURS=0                        # 0 = inga tidsgrense, --cycles avgjer
GAMES=200
SIMS=45
STEPS=450
OPP=punisher
HAND=hold
EXTRA=()
DRY=0
NO_PUSH=0
PUSH_EVERY=1        # kvar N-te syklus vert pusht (0 = berre den siste)

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
    --push-every) PUSH_EVERY="$2"; shift 2 ;;
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
    || die "når ikkje origin — sjekk nøkkel/nett FØR treninga, ikkje etter fleire timar"
  git pull -q --rebase --autostash origin "$BR" \
    || die "git pull --rebase feila — fiks det no, ikkje etter tre timar"
fi

# Kodevakt: dei nye trekka og straffaren MÅ finnast i arbeidstreet, elles
# trenar boksen med gamle trekk og porten lastar noko anna enn det som vart trena.
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

# kvar er vi? (siste fullførte syklus, so ein avbroten køyring held fram rett)
DONE=$(.venv/bin/python3 - "$RUN" <<'PY'
import json, pathlib, sys
run = sys.argv[1]
p = pathlib.Path(f"runs/{run}/training_log.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
print(max((r.get("cycle", 0) for r in rows), default=0))
PY
)
GPU=$(.venv/bin/python3 -c "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null)

say "startvekt: $INIT ($(.venv/bin/python3 -c "
from sahkku_ai.model import Net
n=Net.load('$INIT', map_location='cpu'); print(n.meta.get('run','?'), '→', n.meta.get('steps','?'), 'steg')" 2>/dev/null))"
say "diett: $OPP som motstandar, nettsida tvinga inn i «$HAND»"
say "mål: syklus 1..$CYCLES · $GAMES parti · $SIMS sims · $STEPS steg · $GPU"
[ "$DONE" -gt 0 ] && say "held fram frå syklus $((DONE + 1)) ($DONE alt fullførte)"
say "kvar syklus vert eksportert, gata og pusht for seg"
[ "$HOURS" != "0" ] && say "tidsbudsjett: $HOURS t — stoppa etter siste HEILE syklus innanfor budsjettet"
[ "$DRY" = "1" ] && say "(dry) ingen trening, ingen commit, ingen push"

if [ "$DONE" -ge "$CYCLES" ]; then
  say "syklus 1..$CYCLES er alt ferdige — ingenting å gjere"
  exit 0
fi

# ---- hjelparar ----------------------------------------------------------
T0=$(date +%s)
budget_left() {
  [ "$HOURS" = "0" ] && return 0
  [ $(( $(date +%s) - T0 )) -lt $(( HOURS * 3600 )) ]
}

export_and_gate() {          # $1 = syklusnummer
  local k="$1"
  local ck="checkpoints/${RUN}/c${k}.pt"
  # Pipeline skriv c<k>.pt under checkpoints/<run>/; treningssteget skriv òg
  # checkpoints/<run>_c<k>/latest.pt. Ta den andre om den fyrste manglar, so
  # ein ikkje stoggar på ei fillene ein berre ikkje visste namnet på.
  [ -f "$ck" ] || ck="checkpoints/${RUN}_c${k}/latest.pt"
  [ -f "$ck" ] || die "syklus $k skreiv ingen vekt (verken checkpoints/${RUN}/c${k}.pt eller checkpoints/${RUN}_c${k}/latest.pt) — stoggar"
  say "[c$k] eksporterer $ck → engines/${RUN}-nn"
  .venv/bin/python3 -m sahkku_ai.export --net "$ck" --out "engines/${RUN}-nn" --name "${RUN^^}" >/dev/null \
    || die "eksporten feila for syklus $k"
  local par
  par=$(.venv/bin/python3 tests/test_net_parity.py "engines/${RUN}-nn.pt" --js "engines/${RUN}-nn.js" 2>&1 | tail -1)
  case "$par" in *OK*) say "[c$k] nett-paritet: OK" ;; *) die "[c$k] nett-paritet feila: $par" ;; esac
  local suci
  suci=$( { printf 'uci\nposition startpos\ngo depth 1\nquit\n' | node "engines/${RUN}-nn.js"; } 2>&1 | grep -c "uciok")
  [ "$suci" -ge 1 ] || die "[c$k] motoren svarar ikkje på S-UCI"
}

write_cycle_row() {          # $1 = syklusnummer
  .venv/bin/python3 - "$RUN" "$1" "$INIT" "$OPP" "$HAND" "$DONE" <<'PY' || true
import json, pathlib, sys, torch
run, k, init, opp, hand = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
done = int(sys.argv[6])
p = pathlib.Path(f"runs/{run}_report.md")
if not p.exists():
    head = [f"# {run.upper()} — sykluslogg", "",
            f"- startvekt: `{init}`",
            f"- diett: motstandar **{opp}**, nettsida tvinga inn i **{hand}**",
            "", "| syklus | prøver (til no) | val-tap | sjekkpunkt |", "|---|---|---|---|"]
    # Syklusar som alt var trena før dette skriptet tok over: før dei inn i
    # tabellen, so loggen ikkje ser ut som økta byrja midt i.
    if done > 0:
        try:
            rows_prev = [json.loads(l) for l in
                         pathlib.Path(f"runs/{run}/training_log.jsonl").read_text().splitlines() if l.strip()]
        except Exception:
            rows_prev = []
        cum = 0
        for r in rows_prev:
            if r.get("cycle", 0) > done:
                break
            cum += r.get("new_samples", 0)
            ck = f"checkpoints/{run}/c{r['cycle']}.pt"
            try:
                bv = torch.load(ck, map_location="cpu", weights_only=False).get("meta", {}).get("best_val")
            except Exception:
                bv = None
            head.append(f"| {r['cycle']} | {cum} | {bv:.3f} | `{ck}` |" if isinstance(bv, float)
                        else f"| {r['cycle']} | {cum} | ? | `{ck}` |")
    p.write_text("\n".join(head) + "\n")
try:
    rows = [json.loads(l) for l in pathlib.Path(f"runs/{run}/training_log.jsonl").read_text().splitlines() if l.strip()]
except Exception:
    rows = []
samples = sum(r.get("new_samples", 0) for r in rows)
ck = f"checkpoints/{run}/c{k}.pt"
try:
    best = torch.load(ck, map_location="cpu", weights_only=False).get("meta", {}).get("best_val")
except Exception:
    best = None
line = f"| {k} | {samples} | {best:.3f} | `{ck}` |" if isinstance(best, float) \
       else f"| {k} | {samples} | ? | `{ck}` |"
p.write_text(p.read_text().rstrip("\n") + "\n" + line + "\n")
print(line)
PY
}

commit_and_push() {          # $1 = syklusnummer
  local k="$1"
  [ "$NO_PUSH" = "1" ] && { say "[c$k] --no-push: motor ligg i engines/${RUN}-nn.*"; return 0; }
  # Kvar N-te syklus (og alltid den siste): 40 syklusar × 6,4 MB er 250 MB
  # git-historie, og det er ingenting å hente på å ha alle mellomstega der.
  if [ "$PUSH_EVERY" != "0" ] && [ "$PUSH_EVERY" != "1" ] \
     && [ $((k % PUSH_EVERY)) -ne 0 ] && [ "$k" -ne "$CYCLES" ]; then
    say "[c$k] hoppar over push (kvar ${PUSH_EVERY}. syklus; neste ved $(( (k / PUSH_EVERY + 1) * PUSH_EVERY )))"
    return 0
  fi
  # Berre sjekkpunktet og loggen i git. .js/.weights vert eksporterte av porten
  # på Nox-sida uansett, so dei treng ikkje liggje i historia — og dei er 8,5 MB
  # per syklus me ikkje får att.
  git add "engines/${RUN}-nn.pt" "runs/${RUN}_report.md"
  if git diff --cached --quiet; then
    say "[c$k] inga endring i motor-filene — hoppar over commit"
  else
    git commit -q -m "$RUN syklus $k: eksportert frå checkpoints/$RUN/c$k.pt

Trena med --train-opponent $OPP --hand-style $HAND oppå $(basename "$INIT")
(der ein av dei er brukt): nettsida vert tvinga inn i «$HAND» — aktiver fleire
soldatar, hald dei utanfor motstandaren si rekkjevidd — medan motstandaren tek
det som stikk fram. Motsett doktrine av NAIS.7 (--style-both few, ~1 utløyst
soldat). g[16]/g[17] er nye og startar på 0, so nettet spelar som før til det
lærer noko nyttig.

Nett-paritet og S-UCI grøne lokalt. Benk, versjon og deploy gjer porten på
Nox-sida når committen ligg på origin/main." || die "[c$k] git commit feila"
    say "[c$k] committa"
  fi
  local i
  for i in 1 2 3; do
    if git push -q origin "$BR"; then say "[c$k] pusht → porten tek over innan 20 min"; return 0; fi
    say "[c$k] push feila (forsøk $i) — rebaser og prøver att"
    git pull -q --rebase --autostash origin "$BR" || true
    sleep 5
  done
  die "[c$k] push feila tre gonger — committen ligg lokalt, køyr: git push origin $BR"
}

# ---- 1. løkka: éin syklus om gongen -------------------------------------
k=$((DONE + 1))
while [ "$k" -le "$CYCLES" ]; do
  if ! budget_left; then
    say "tidsbudsjettet ($HOURS t) er brukt — stoppar etter syklus $((k - 1)) av $CYCLES"
    break
  fi
  say "── syklus $k av $CYCLES ──────────────────────────────────"
  DRYARG=(); [ "$DRY" = "1" ] && DRYARG=(--dry)
  ./badass-trainer.sh --run "$RUN" --init "$INIT" \
      --cycles "$k" --hours 0 --games "$GAMES" --sims "$SIMS" --steps "$STEPS" \
      --nesseby --nesseby-fraction 1.0 \
      --train-opponent "$OPP" --hand-style "$HAND" \
      "${EXTRA[@]}" "${DRYARG[@]}" \
    || die "syklus $k stoppa (sjå loggen over) — ingenting frå denne syklusen er pusht"
  if [ "$DRY" = "1" ]; then
    say "(dry) ville no eksportert checkpoints/$RUN/c$k.pt og pusht"
    k=$((k + 1))
    continue
  fi
  export_and_gate "$k"
  write_cycle_row "$k"
  commit_and_push "$k"
  k=$((k + 1))
done

FINAL=$((k - 1))
say "ferdig med syklus $FINAL av $CYCLES"
cat <<EOF

Kvar syklus er pusht og vert publisert av porten på Nox-sida (S-UCI, paritet,
benk mot gjeldande standardmotor, versjonsbump, deploy). Rapport:
  runs/${RUN}_report.md   (sykluslogg + benk per publisering)

Sjå framgangen sjølv:
  .venv/bin/python3 tools/train_status.py $RUN
EOF
