#!/usr/bin/env bash
# Vakt + port: vent på at ein trena motor hamnar på GitHub, køyr porten, publiser.
#
# Køyrer som cron-tick (no_agent): skriv INGENTING før motoren finst, so ein
# tick utan nyhende er stille. Når motoren er på origin/main:
#   1. hent han, eksporter til appen med rett namn
#   2. S-UCI + nett-paritet (publiser aldri ein motor som ikkje er den trena)
#   3. benk: Nesseby mot førre modell (30) og klassisk nivå 3 (40)
#   4. legg han i motorlista — STANDARD berre om han slår førre modell (≥55 %)
#   5. bump service-worker-cache + versjon, commit, push, deploy
# og skriv ein rapport til stdout (som vert levert) + runs/<run>_report.md.
#
#   tools/gate-publish.sh nais7 nais5
set -uo pipefail
RUN=${1:-nais8}
BASE=${2:-auto}      # auto = den motoren som er standard no (kan skifte mellom blokker)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 0
APP=/home/lars/src/SnartSahkku
MARK="runs/${RUN}_published"     # inneheld shaen av det som vart publisert
# Fleire blokker kan pushe same RUN (kort blokk om kvelden, lang om natta).
# Difor: marker på motorFILA sin sha, ikkje på namnet — ei ny vekt = ny
# publisering, same vekt = stille.
LOCAL_SHA=$(sha256sum "engines/${RUN}-nn.pt" 2>/dev/null | cut -d' ' -f1)
[ -n "$LOCAL_SHA" ] && [ -f "$MARK" ] && [ "$(cat "$MARK")" = "$LOCAL_SHA" ] && exit 0
exec 9>"/tmp/gate-${RUN}.lock"; flock -n 9 || exit 0      # ein tick om gongen

say() { printf '%s\n' "$*"; }

git fetch -q origin 2>/dev/null || exit 0
git ls-tree -r --name-only origin/main -- engines 2>/dev/null | grep -qx "engines/${RUN}-nn.pt" || exit 0
BR=$(git rev-parse --abbrev-ref HEAD)
git pull -q --rebase --autostash origin "$BR" >/dev/null 2>&1 || git pull -q --rebase origin "$BR" >/dev/null 2>&1

# Kven er standardmotoren no? (Kan ha skifta sidan førre blokk.)
if [ "$BASE" = "auto" ]; then
  BASE=$( .venv/bin/python3 -c "
import json,sys
m=json.load(open('$APP/engines/engines.json'))
print(m.get('default') or '')" 2>/dev/null )
  [ -n "$BASE" ] || { say "FEIL: fann ikkje standardmotor i engines.json"; exit 1; }
  [ "$BASE" != "$RUN" ] || BASE=$( .venv/bin/python3 -c "
import json
m=json.load(open('$APP/engines/engines.json'))
ids=[e['id'] for e in m['engines'] if e.get('type')=='nn' and e['id']!='$RUN']
print(ids[-1] if ids else '')" 2>/dev/null )
fi
[ -n "$BASE" ] || { say "FEIL: ingen samanlikningsmotor"; exit 1; }

say "=== $RUN har nådd GitHub — køyrer porten (samanliknar mot $BASE) ==="
CK="engines/${RUN}-nn.pt"
[ -f "$CK" ] || { say "FEIL: $CK manglar etter pull"; exit 1; }
# Etter pull: er dette same vekta som alt vart publisert? (Vakta startar oss
# berre ved ny sha, men porten kan òg startast for hand.)
NOW_SHA=$(sha256sum "$CK" | cut -d' ' -f1)
[ -f "$MARK" ] && [ "$(cat "$MARK")" = "$NOW_SHA" ] && { say "$RUN: same vekt som sist — ingenting å gjere"; exit 0; }

# ---- 1. eksport til appen (rett namn → rett vektfil i .js) --------------
.venv/bin/python3 -m sahkku_ai.export --net "$CK" --out "$APP/engines/$RUN" --name "NAIS.${RUN#nais}" >/dev/null 2>&1 \
  || { say "FEIL: eksport feila"; exit 1; }
rm -f "$APP/engines/$RUN.pt"

# ---- 2. testane ---------------------------------------------------------
SUCI=$(cd "$APP" && SAHKKU_ENGINE="node engines/$RUN.js" node tests/suci.test.js 2>&1 | tail -1)
PAR=$( .venv/bin/python3 tests/test_net_parity.py "$CK" --js "$APP/engines/$RUN.js" 2>&1 | tail -1 )
say "S-UCI: $SUCI"
say "paritet: $PAR"
case "$SUCI" in *"0 fail"*) ;; *) say "FEIL: S-UCI feila — publiserer ikkje"; exit 1 ;; esac
case "$PAR" in *"OK"*) ;; *) say "FEIL: nett-paritet feila — publiserer ikkje"; exit 1 ;; esac

# ---- 3. benk ------------------------------------------------------------
say "benk: Nesseby mot $BASE (30) og klassisk nivå 3 (40) — tek ~1,5 t"
# Treningsboksen pushar berre sjekkpunktet (.pt) for å halde git-historia
# rimeleg. Difor bench-ar vi appen sin .js, som porten eksporterte rett over.
NEWJS="$APP/engines/${RUN}.js"
BASEJS="$APP/engines/${BASE}.js"
[ -f "$NEWJS" ] || { say "FEIL: $NEWJS manglar etter eksport"; exit 1; }
[ -f "$BASEJS" ] || { say "FEIL: $BASEJS manglar (base=$BASE)"; exit 1; }
EVAL_NESSBY=1 node tools/eval_match.js "node $NEWJS" "node $BASEJS" 30 181 \
  > "runs/eval/${RUN}_nesseby.txt" 2>&1
EVAL_SETOPT_B=Level=3 node tools/eval_match.js "node $NEWJS" "node $APP/engines/sahkku-ref.js" 40 182 \
  > "runs/eval/${RUN}_L3.txt" 2>&1
NB=$(grep -oE "A-poengsum: [0-9.]+" "runs/eval/${RUN}_nesseby.txt" | tail -1 | grep -oE "[0-9.]+")
L3=$(grep -oE "A-poengsum: [0-9.]+" "runs/eval/${RUN}_L3.txt" | tail -1 | grep -oE "[0-9.]+")
NB=${NB:-0}; L3=${L3:-0}
say "resultat: Nesseby $NB % mot $BASE · klassisk L3 $L3 %"

# Er dette framleis siste vekt? Under ein lang treningskøyring kjem det nye
# sjekkpunkt kvart 12. minutt, og då er det sløsing å publisere eit som alt er
# avløyst — neste tick tek den nye i staden.
git fetch -q origin 2>/dev/null || true
NOW_ORIGIN=$(git show "origin/main:engines/${RUN}-nn.pt" 2>/dev/null | sha256sum | cut -d' ' -f1)
if [ -n "$NOW_ORIGIN" ] && [ "$NOW_ORIGIN" != "$NOW_SHA" ]; then
  say "$RUN: nyare sjekkpunkt kom medan benken gjekk (${NOW_ORIGIN:0:12} mot ${NOW_SHA:0:12})"
  say "publiserer ikkje eit avløyst sjekkpunkt — neste tick tek den nye"
  exit 0
fi

# ---- 3b. menneskelege mål (ikkje-fatale, men skrivne i rapporten) -------
# Benken over spelar mot ein motor. Det mennesket straffar er noko anna:
# kor mange soldatar nettet held UTE, og kor mange av dei som vert tekne.
# Same probe som vart brukt til aa diagnostisere NAIS.7-partiet (0.8 ute).
HUM=""
if [ -f tools/punisher_probe.py ]; then
  say "menneskeleg mål: straffar mot $RUN (4 parti, --hand-style hold)"
  if .venv/bin/python3 tools/punisher_probe.py --net "$CK" --games 4 --sims 45 --hand-style hold \
       > "runs/eval/${RUN}_human.txt" 2>&1; then
    HUM=$(grep -E "nettet miste|for nettet" "runs/eval/${RUN}_human.txt" | tr '\n' ' ' | sed 's/  */ /g')
    say "  $HUM"
  else
    say "  (proben feila — sjå runs/eval/${RUN}_human.txt; publiserer likevel)"
  fi
fi

# ---- 4. motorlista ------------------------------------------------------
DEF=$( .venv/bin/python3 - "$RUN" "$NB" "$L3" "$BASE" <<'PY'
import json, pathlib, sys
run, nb, l3, base = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
p = pathlib.Path("/home/lars/src/SnartSahkku/engines/engines.json")
m = json.loads(p.read_text())
eid = run                      # nais7
name = "NAIS." + run.replace("nais", "")
better = nb >= 55.0
note = (f"Trena {run} (sjå sahkku-ai/docs/NAIS6-PLAN.md og runs/README) — "
        f"{nb} % mot {base} i Nesseby (n=30, 30 parti) og {l3} % mot klassisk nivå 3 (n=40). "
        + ("Standardmotor." if better else "Ikkje standardmotor (slår ikkje førre modell)."))
entry = {"id": eid, "name": name, "type": "nn", "script": f"engines/{eid}.js",
         "weights": f"engines/{eid}.weights.json", "lookahead": 4, "note": note}
m["engines"] = [e for e in m["engines"] if e["id"] != eid] + [entry]
if better:
    m["default"] = eid
p.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n")
print("ja" if better else "nei")
PY
)
say "standardmotor: $([ "$DEF" = "ja" ] && echo "JA ($RUN)" || echo "nei — $BASE står")"

# ---- 5. cache/versjon, commit, push, deploy ----------------------------
cd "$APP" || exit 1
python3 - <<'PY'
import pathlib, re, sys
sw = pathlib.Path("sw.js"); s = sw.read_text()
m = re.search(r"sahkku-v(\d+)", s)
if m: s = s.replace(m.group(0), f"sahkku-v{int(m.group(1)) + 1}")
sw.write_text(s)
main = pathlib.Path("js/main.js"); t = main.read_text()
m = re.search(r"APP_VERSION = 'v(\d+)\.(\d+)\.(\d+)'", t)
if not m:
    sys.exit("APP_VERSION vart ikkje funnen i js/main.js — publicerer ikkje utan versjonsbump")
new = f"v{m.group(1)}.{int(m.group(2)) + 1}.0"
t = t.replace(m.group(0), f"APP_VERSION = '{new}'")
main.write_text(t)
idx = pathlib.Path("index.html"); h = idx.read_text()
h2 = re.sub(r"v\d+\.\d+\.\d+ · AGPL", f"{new} · AGPL", h)
if h2 == h:
    sys.exit("fann ikkje versjonslinja i index.html — publicerer ikkje utan ho")
idx.write_text(h2)
print(new)
PY
VERBUMP=$?
[ "$VERBUMP" = "0" ] || { say "FEIL: versjons-/cache-bump feila — deployar ikkje ein motor brukarane ikkje får"; exit 1; }
git add -A >/dev/null 2>&1
git commit -q -m "Publiser $RUN: $NB % mot $BASE i Nesseby (n=30), $L3 % mot klassisk L3 (n=40). Standardmotor: $DEF. Eksportert frå engines/$RUN-nn.pt; S-UCI og nett-paritet grøne." || true
git push -q origin main 2>/dev/null || { git pull -q --rebase origin main >/dev/null 2>&1; git push -q origin main 2>/dev/null || say "ÅTVARING: push til appen feila"; }
VER=$(node -e "console.log(require('fs').readFileSync('js/main.js','utf8').match(/APP_VERSION = '([^']+)'/)[1])")
DEP=$(./sendsahkku.sh 2>&1 | tail -1)
say "deploy: $DEP"

sha256sum "$CK" | cut -d' ' -f1 > "$ROOT/$MARK"
{
  say ""
  say "**$RUN er publisert** — $VER"
  say "- Nesseby mot $BASE: **$NB %** (n=30)"
  say "- klassisk nivå 3: **$L3 %** (n=40)"
  say "- standardmotor: $([ "$DEF" = "ja" ] && echo "**ja**" || echo "nei — $BASE står")"
  say "- S-UCI: $SUCI"
  say "- deploy: $DEP"
  [ -z "$HUM" ] || say "- mot straffaren: $HUM"
} | tee -a "$ROOT/runs/${RUN}_report.md"
