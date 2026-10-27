# sáhkku-ai — nevral S-UCI-motor og treningsramverk for sáhkku

Et treningsramverk for **Lágesvuotna-sáhkku** (AlphaZero-aktig: sjølvspel →
policy/value-nett → eksport) og den ferdige motoren det gir:
`engines/sahkku-nn.js`, en **S-UCI 1.0**-motor som kjører på rein JavaScript
(Node 18+ *og* i nettleseren) uten npm-avhengigheter.

Motoren dømmer etter **nøyaktig samme regler som SnartSáhkku-PWA-en**:
regelfilene `js/engine.js` + `js/fen.js` blir *injisert* i motoren ved eksport
(`vendor/` er verbatim-kopier). Ingen avskrevet regelkopi, ingen drift.

- Protokoll: [`docs/SAHKKU-UCI.md`](../SnartSahkku/docs/SAHKKU-UCI.md) i SnartSahkku
- Spilleregler: [sahkku.no/docs](https://sahkku.no/docs)

**Verifisert** (kjørt, ikke påstått — `./tests/run_all.sh`):

| test | resultat |
|---|---|
| terningreglane (`tests/test_rules.py`) | **33 sjekkar**: X vekkjer berre, omkast berre når X er ubrukeleg, X > III > II, resten tapt |
| regelparitet Python ↔ JS (`tests/fuzz_parity.py`, 300 parti) | **14 483 trekkpunkt, 0 avvik** (pool + FEN + trekkutførelse) |
| nettparitet PyTorch ↔ rein JS (`tests/test_net_parity.py`) | **25 posisjoner**, Δpolicy/Δvalue < 1e-2 |
| nettlesar-stien (`tests/test_browser_path.js`) | modulen laster uten Node-globaler, Δ ≤ 5e-6 |
| S-UCI/GUI-testen til SnartSahkku (heilt parti) | **31 pass, 0 fail** |
| GPU-trening på Tesla P100 (cc 6.0) | sjølvspel + trening + eksport + styrkemåling, fleire syklusar (sjå «Resultater») |

```
         .spg-samlinger ─┐
                         ├─→ .npz-shards ─→ PyTorch-nett (GPU) ─→ checkpoint .pt
  sjølvspel (GPU) ───────┘                                          │
                                                                    ▼
                                        engines/sahkku-nn.js  ◀── eksport
                                        (+ vekter, + injiserte regler)
                                                    │
                                                    ▼
                                       S-UCI-GUI / SnartSahkku-testkit
```

## Spillereglene (terningane)

Kortversjonen som motoren implementerer — og som er køyrbar som test i
`tests/test_rules.py` (58 sjekkar):

1. **X er eit 1-tal.** Ho vekkjer éi inaktiv brikke (soldat eller dronning —
   flyttet er ein del av vekkinga) **og** flyttar ei utløyst brikke eitt steg:
   soldat framover i tallsyklusen, dronning og konge eitt steg langs syklusen.
2. **X kan alltid kastast om** — bortsett frå når aktivering er det einaste du
   kan gjere (inga brikker er utløyste). Elles kunne ein spelar som aldri
   aktiverer noko blokkere spelet for alltid. Etter eit omkast gjeld vanlege
   reglar, òg om det nye fjeset òg er ein X.
3. **Rekkjefylgja er X > III > II.** Blanke fjes er brukte med ein gong.
4. **Kan du ikkje flytte den aktive terningen, mistar du resten.** Får du III
   og II og ikkje kan flytte III, vert II ikkje spelt — turen sluttar.
5. **Konge og dronning går langs tallsyklusen** (åttetalet), begge vegar, og
   kan hoppe over brikker. **Kongen slår ut** fiendtlege soldatar og dronninga
   (tek han dronninga, vinn han), men landar aldri på eigne eller uutløyste
   brikker.
6. **Nesseby** (valfri): kongen kan ikkje kaprast ved landing før nokon har
   nådd motstandarens heimrad; den fyrste soldaten som når dit kaprar kongen.

X, III og II er elles 1, 3 og 2 steg. Vinst: fiendens dronning er slått ut,
eller alle fiendens soldatar er borte.

## Innhold

| fil | innhold |
|---|---|
| `sahkku_ai/engine.py` | ren Python-port av regelmotoren (paritetsverifisert) |
| `sahkku_ai/fen.py` | S-UCI-FEN: `to_fen` / `from_fen` |
| `sahkku_ai/spg.py` | SPG 1.0-leser → avspilling på motoren |
| `sahkku_ai/features.py` | 16 plan à 3×15 + 10 globale tall; aksjonsrom 8282 |
| `sahkku_ai/model.py` | PyTorch-nett (conv-trunk + resblokker, policy-/value-hode) |
| `sahkku_ai/selfplay.py` | sjølvspel på GPU med batcha utspillinger |
| `sahkku_ai/train.py` | treningsløkka (maskert policy-CE + value-MSE, gjenopptak) |
| `sahkku_ai/data.py` | SPG/sjølvspel → trening-shards |
| `sahkku_ai/export.py` | checkpoint → `engines/sahkku-nn.js` + vekter (JSON) |
| `sahkku_ai/pipeline.py` | heile løkka i én kommando |
| `engines/sahkku-nn.js` | den ferdige S-UCI-motoren (regler + nett, rein JS) |
| `engines/sahkku-nn.pt` | checkpointet motoren vart eksportert frå (same nett som vektene) |
| `engines/runner_template.js` | malen eksporten fyller inn regler/vekter i |
| `tools/eval_match.js` | styrkemåling: to motorer mot hverandre, JS-dommer |
| `tools/parity_probe.js` | JS-siden av regelpariteten |
| `tests/` | `run_all.sh` + de fem testsuitene |

## Krav

- **Python 3.10+** med PyTorch (CUDA-bygg hvis du vil trene på GPU) og numpy
- **Node 18+** (motoren, testene og styrkemålingen)

```sh
git clone git@github.com:larsnygard/sahkku-ai.git && cd sahkku-ai
python -m venv .venv
.venv/bin/pip install --upgrade pip
# GPU (CUDA 12.x; P100/cc 6.0 er dekket av cu126-hjulet):
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
# eller CPU-berre:
# .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy
```

Regelfilene hentes fra en SnartSahkku-checkout som ligger ved siden av
(`../SnartSahkku`). Er den ikke der, kopierer du bare `js/engine.js` og
`js/fen.js` inn i `vendor/` — eller kjører:

```sh
./scripts/sync_vendor.sh ../SnartSahkku    # oppdaterer vendor/ etter regelendringer
```

## Kjør alt i én kommando

```sh
.venv/bin/python -m sahkku_ai.pipeline \
    --run r1 --cycles 4 --games 600 --workers 4 --rollouts 8 \
    --steps 600 --batch 256 --lr 1.5e-3 \
    --ch 64 --blocks 6 --fc 128 \
    --device cuda --eval-games 20
```

Hver syklus gjør: **sjølvspel → trening (gjenopptak) → eksport → styrkemåling**,
og skriver `runs/r1/training_log.jsonl` + `runs/r1/eval_c<N>.txt`. Sykluser som
alt er kjørt hoppes over, så kommandoen kan avbrytes og startes igjen:

```sh
# hald fram der du slapp (startar frå checkpoints/r1/c<N-1>.pt)
.venv/bin/python -m sahkku_ai.pipeline --run r1 --start-cycle 5 --cycles 8 ...
```

Ferdig produkt: `engines/sahkku-nn.js` + `.weights.json` + `.pt` (same nett, tre deler).
Vil du ha det steg for steg (lettere å feilsøke), er det de fem avsnittene under.

---

## 1. Treningsdata fra SPG (valgfritt, men anbefalt)

SPG er notasjonsformatet til SnartSahkku (`docs/SPG.md`). Hver spilte brikke blir
ett treningspunkt: stilling → menneskets trekk (soft policy-mål: 0.9 på trekket,
0.1 fordelt på resten av den lovlige poolen) + partresultatet som value-mål.

```sh
.venv/bin/python -c "
from sahkku_ai.data import dataset_from_spg
dataset_from_spg('sti/til/spg-samling', 'data/spg_shard.npz')"
# SPG-filer: 1 fila, 0 hoppa over, 22 prøver
# SPG -> data/spg_shard.npz: 22 prøver
```

Partier som ikke kan spilles opp igjen på motoren (`strict=True`) hoppes over —
det skal ikke skje, siden motoren er paritetsverifisert mot JS.

## 2. Sjølvspel

Nettet spiller mot seg selv; hver beslutning blir ett treningspunkt.

```sh
.venv/bin/python -m sahkku_ai.selfplay \
    --net checkpoints/r1/init.pt --outdir runs/r1/cycle1 \
    --games 600 --workers 4 --rollouts 8 --device cuda
```

- **`--workers`** = parallelle prosesser, hver med eget nett på GPU (~330 MB).
  Bruker *spawn*, ikke fork (CUDA kan ikke re-initialiseres etter fork).
- **`--rollouts`** = antall utspillinger per beslutning. Hver utspilling er et
  helt parti fra den ene rotaksjonen, batcha over GPU-en (én forward per ply
  for alle utspillingene samtidig).
- Policy-målet er **utfallsvektet**, ikke bare netttes egen prior:
  `π[a] ∝ (n[a] + prior_eps·N·prior[a]) · exp(q[a]/tau_q)`, der `q[a]` er
  middelutfall (rotspillerens syn) for aksjon `a` i utspillingene. Uten denne
  vektingen blir målet bare netttes egen prior renormalisert (selvdistillasjon
  som aldri kan bli sterkere). Skru med `--prior-eps` og `--tau-q`.
- Aksjoner som **avgjør partiet** (dronning/kongekapring, utryddelse) får
  utfall ±1 direkte — de er de skarpeste målene i datasettet.

Målt på P100 med 64×6-nettet: **~13 parti/minutt per arbeider**, ~24 prøver per
parti, ~330 MB VRAM per arbeider.

## 3. Trening

```sh
.venv/bin/python -m sahkku_ai.train \
    --run r1_c1 --shards 'runs/r1/cycle1/*.npz data/spg_shard.npz' \
    --steps 600 --batch 256 --lr 1.5e-3 --device cuda
# device: Tesla P100-PCIE-16GB (cc 6.0, torch 2.14.1+cu126, fp32)
# steg 300/600  tap 2.28 (pol 1.70 verd 0.57)  lr 1.00e-03  8s
#   VAL: policy-xent 1.9036  verd-MSE 0.6271
# trening ferdig på 15s → checkpoints/r1_c1/latest.pt
```

- **Tap:** maskert policy-kryssentropi mot `π` (kun lovlige aksjoner) +
  MSE på value.
- **Gjenopptak:** `--init checkpoints/r1/latest.pt` laster vektene og
  fortsetter (meta følger med: `step`, `samples`, `best_val`, `history`).
- **Ingen AMP.** P100 er compute capability 6.0; fp16-konvolusjoner er
  halvferdige der, og fp32 holder treningen numerisk i takt med JS-eksporten
  (som også regner fp32).
- Sjekkpunkter: `checkpoints/<run>/step<N>.pt`, `latest.pt`, `best.pt`.

Sjekk at læringa faktisk når parametrane (overfit-test på 22 SPG-prøver — tapet
skal lande på entropien til målet, ~0.37, ikke over):

```sh
.venv/bin/python -m sahkku_ai.train --run overfit --shards data/spg_shard.npz \
    --steps 800 --batch 22 --lr 3e-3 --device cuda
# steg 800/800  tap 0.3726 (pol 0.3726 verd 0.0000)   <- H(π) ≈ 0.37
```

## 4. Eksport til S-UCI-motoren

```sh
.venv/bin/python -m sahkku_ai.export \
    --net checkpoints/ail01/c9.pt --out engines/sahkku-nn
# eksport: engines/sahkku-nn.js + engines/sahkku-nn.weights.json + engines/sahkku-nn.pt
```

Eksporten gjør fire ting:

0. kopierer checkpointet til `engines/sahkku-nn.pt`, så JS-kode, vekter og
   kjeldenett alltid hører sammen (og en fersk klone kan kjøre nettparitetstesten);
1. alle tensorer → **base64 float32** (én per tensor, med shapes) i `.weights.json`;
2. `engines/runner_template.js` får `vendor/engine.js` + `vendor/fen.js`
   **injisert** (§RULES§/§FEN§-markørene) — samme regler som PWA-en;
3. den ferdige fila skriver nettets graf i rein JS (conv, batchnorm, relu,
   linear, tanh — samme rekkefølge som `model.py`).

Motoren kan kjøres med en gang:

```sh
printf 'uci\nisready\nposition startpos\ndiceroll X 2 _\ngo\nquit\n' | node engines/sahkku-nn.js
# id name SahkkuNN 1.0
# id author sahkku-ai (PyTorch -> JS export)
# option name Debug type check default false
# option name Lookahead type spin default 4 min 0 max 12
# uciok
# readyok
# info movepool O1-O2
# info movepool L2-M2
# info movepool L2-K2
# bestmove L2-K2
```

### Motorvalg

| option | type | default | betydning |
|---|---|---|---|
| `Debug` | check | false | `info string` med verdi per trekk |
| `Lookahead` | spin 0–12 | **4** | 1-ply verdilookahead over de k+1 beste policy-trekkene |
| `Deep` | spin 0–4 | 0 | 2-ply: hvert kandidattrekk scores med det **verste** motstanderen kan gjøre mot oss blant hans m beste svar. Koster ~2× per trekk (målt 1,05 s → 1,68 s med m=2). **Målt dårligere: 20 % (6–24) mot samme vekter uten Deep**, så den er av som standard — verdinettet er for svakt til å bære minimax, og pessimismen gjør spillet passivt. Beholdt som option for forsøk med sterkere verdinett. |

`Lookahead` er ikke pynt: verdinettet blir trent, men policy-hodet alene bruker
det ikke. Med `k=4` spilles de fem beste policy-trekkene på en klon av den ekte
tilstanden, og barnestillingen verdsettes av verdinettet (med fortegn for hvem
som trekker) — motoren velger aksjonen med best *verdi* i stedet for best
*logit*. Målt med **samme vekter**:

```
node tools/eval_match.js "node engines/sahkku-nn.js" "node engines/sahkku-nn.js" 30 77
  (EVAL_SETOPT_A=Lookahead=4, EVAL_SETOPT_B=Lookahead=0)
SUM: A=25  B=5   A-poengsum: 83.3 %
```

`k=0` gir momentmotoren (rein policy), `k=8` ga samme styrke som `k=4` men dobbel
tid. Kostnad per trekk i rein JS (Node, 11 lovlige trekk i stillinga):

| Lookahead | ms/trekk |
|---|---|
| 0 (rein policy) | 117 |
| **4 (levert)** | **623** |
| 8 | 1030 |

Uten lookahead er nettet selv begrensningen (15 ms for 32×2, 76 ms for 64×4,
109 ms for 64×6, 238 ms for 96×6).

## 5. Styrkemåling

`tools/eval_match.js` spiller to S-UCI-motorer mot hverandre med **JS-dommeren
som referent**: den kaster terningene, kontrollerer hvert `bestmove` mot den
lovlige poolen, bruker `reroll`/`pass` etter reglene og bytter side på A hvert
parti (sáhkku er asymmetrisk).

```sh
node tools/eval_match.js "node engines/sahkku-nn.js" \
                         "node ../SnartSahkku/engines/sahkku-ref.js" 20 41
# parti 1 [A=Menn]: B (Giera) 33 halvtrekk {dronning}
# ...
# SUM: A=7  B=13  uavgjort=0  av 20  (82 s)
#   A som Menn: 1-9
#   A som Giera: 6-4
#   A-poengsum: 35.0 %  (seier=1, uavgjort=0,5)
```

Motorvalg settes med miljøvariabler (standard: begge motorer):

```sh
EVAL_SETOPT="Level=1"  node tools/eval_match.js …     # begge
EVAL_SETOPT_A="Lookahead=0" node tools/eval_match.js … # berre A
```

### Resultater (målt, ikke ønsketenkning)

Motstander: referansemotoren `engines/sahkku-ref.js`. Nivå 2 = grådig med ±125
støy (hver kjøring er derfor et nytt utvalg), nivå 3 = deterministisk grådig.
Kalibrering mot nivå 1 (uniformt tilfeldig), 20 parti hver: **L2 85 %**,
**L3 95 %** — altså ligger et tilfeldig nett på ~15 % mot L2.

| modell | data | val-xent | mot ref L2 | mot sin forgjengar |
|---|---|---|---|---|
| **AIL0.1** (`ail01/c9.pt`) | 9 sykl. / 120 093 prøver | 0.957 | 26.7 % (8–22, n=30) | — |
| **AIL0.2** (`ail02/c12.pt`) | 12 sykl. / 152 351 prøver | **0.840** | 16.7 % (5–25, n=30) | 46.7 % (14–16, n=30) |
| **AIL0.3** (`ail03/c3.pt`, MCTS-mål 25 sims) | 3 sykl. / 156 632 prøver | 1.441\* | **30.0 %** (9–21, n=30) · **22.5 % mot L3** (9–31, n=40) | 40.0 % (12–18) mot AIL0.2 |
| **AIL0.4** (`ail04/c8.pt`, trena MOT klassisk L3, MCTS-mål, nye globale trekk) | 8 sykl. / 91 934 prøver | 0.772 | **27.5 % mot L3** (11–29, n=40) · 25 % som Menn / 30 % som Giera | — |
| **AIL0.5 blokk 1** (`ail05/c6.pt`, held fram mot L3) | 6 sykl. til | 0.842 | **42.5 % mot L3** (17–23, n=40) · 30 % Menn / **55 % Giera** | — |
| **AIL0.5** (`ail05/c12.pt`, 12 sykl. mot L3, lr 4e-4 + vindauge 3) | 12 sykl. / 219 312 prøver | **0.7705** | **50.0 % mot L3** (20–20, n=40) · 11–9 Menn / 9–11 Giera | — |
| referanse: `Lookahead=4` mot `Lookahead=0`, same vekter | — | — | — | 85 % (17–3) |

\* AIL0.3 sitt tap er ikkje samanliknbart: MCTS-π er mjukare (vitjingstal
fordelt på fleire trekk), so kryssentropien *må* bli høgre. Verdi-MSE (0.06) er
heller ikkje samanliknbar — 88 % av partia når ply-taket, so målet er då
søkets eige anslag. Berre hovud-mot-hovud-tal er meiningsfulle her.

**AIL0.4 — endeleg eit grep som monnar, men ikkje nok.** Ho er trena *mot*
klassisk CPU nivå 3 (ikkje mot seg sjølv), med MCTS-mål, og med dei globale
tala endra: `g[3]/g[4]` tel **utløyste** soldatar (ei brikke som står heime er
trygg og gjev ingen mobilitet) og alle globale tal er POV-relative. Målt mot
same benk som AIL0.3: **27,5 % mot 22,5 %** — og sideskeivheita er borte
(25 % som Menn / 30 % som Giera, mot 15/30 før). Framleis tapande mot L3,
men den eine av to kjende veikskapar (sideskeivheita) er retta, og tapet er
no mindre enn feilmarginen på benken.

**Tre modellar, ~430 000 prøver, ingen målbar styrkeauke.** AIL0.3 får 30 %
mot referansen (AIL0.1: 26.7 %, AIL0.2: 16.7 %) men taper 40–60 mot AIL0.2
innbyrdes. Alle tala ligg i eit band der n=30 gjev ±11 prosentpoeng, so
skilnadene er ikkje avgjorde — men *ingen* av dei tre skil seg klart ut.
Det som faktisk ville avgjere det er fleire parti per benk (n≥100), ikkje
fleire treningsrundar.

**To timar meir av same metode gav betre tap og *inga* styrke.** AIL0.2 spår
si eiga spelargrein betre (0.957 → 0.840) men taper mot AIL0.1 (46,7 %) og
gjer det dårlegare mot den grådige referansen (16,7 % mot 26,7 %). Det er
den same diagnosen som tidlegare: måla kjem frå 8 tilfeldige utspelingar av
nettet sjølv, so meir trening finpussar vanane i staden for å lære noko nytt.
Betre mål — søk, ikkje fleire timar — er vegen vidare; AIL0.3 er det forsøket.

Treningskurva: val policy-xent **1.144 → 0.957** (value-MSE 0.83 → 0.57) over
ni syklusar — jamn framgang, ingen kollisjon. Styrken mot den grådige
referansemotoren er derimot *flat* (~27 %) samanlikna med den førre modellen,
så auken i tap ligg i kalibrering, ikkje i spelestyrke enno.

Dronning-aktivering (60 parti med AIL0.1): median utløyst **trekk 3** for båe
sider, og den som utløyser fyrst vinn 54 % — altså aktiverer nettet så tidleg
det kan og vinn ingenting målbart på å vere fyrst. No kan det i det minste
*sjå* skilnaden (før var flagget konstant).

Sideskillet er tydelig: som Menn 13 % (2–13), som Giera 40 % (6–9). Nettet ser
absolutte posisjoner (ingen speiling), så det må lære hver side for seg — med
halvparten av dataene til hver.

Treningsløkka per syklus (500 parti, 8 arbeidere, 800 steg):

| syklus | nye prøver | policy-xent (val) | value-MSE (val) |
|---|---|---|---|
| 1 | 12 799 | 0.971 | 0.696 |
| 2 | 13 748 | 0.962 | 0.615 |
| 3 | 12 416 | 0.965 | 0.680 |

**Les dette riktig:** rammeverket lærer og spiller gjenkjennelig sáhkku — det
tar dronninger, slår referansen av og til, og vinner 85 % mot sin egen
policy-only-tvilling — men det er ikke *sterkt*: 27 % mot en grådig heuristikk
er et stykke under «god motstander». Lookahead er den enkeltendringen som gir
mest (85 % mot seg selv uten), og den koster ~0,5 s per trekk. Se
«Begrensninger» nederst.

## MCTS (`--sims N`)

`sahkku_ai/mcts.py` er eit bladbunta PUCT-søk som erstattar utspelingane når
`--sims N` er sett (0 = dagens løype).

**Designet, og kvifor:**

* **Terningane er ikkje sjansjenodar i treet.** Turen sine terningar er kjende,
  og eit trekk som avsluttar turen er eit *løv* som verdsettast av verdinettet —
  det ukjende neste kastet ligg dermed implisitt i verdien, og treet slepp
  expectimax over 4³ terningar.
* **Verdi, ikkje utspelingar.** Løv vert skorne av verdinettet, so eit 25-sims
  søk kostar ~25 forward i staden for 8 utspelingar à ~52 ply.
* **Bladbunta løv med PUCT** (`c_puct`, Dirichlet-støy ved rot), og forteiknet
  vert snutt når *sida* skiftar — ikkje for kvar ply, sidan same spelar gjer
  opptil tre trekk i ein tur.
* **Verdimålet er partiresultatet** (AlphaZero-konvensjonen), ikkje søkets eige
  verdianslag. Er partiet ikkje avgjort (ply-taket), vert søkeverdien brukt.

**Tre ting som kosta ein køyring kvar:**

1. **Å søkje på `reroll` er meiningslaust** — ho er ei null-handling som kastar
   terningane på nytt utanfor treet, og ho fekk 86 % av vitjingane då ho var
   med (partia sto på 300-ply-taket). Ho er no ute av søkjesettet, men framleis
   lovleg når ho er det einaste ein kan gjere.
2. **Kostnaden er Python per node**, ikkje forwarden: 366 sims/s ved 100 sims,
   og `--leaf-batch` 8/16/32 gjer ingen skilnad. `State.clone()` (0,09 ms) og
   `legal_ids` per utviding er det som dominerer.
3. **Sterkare spel gjev lange parti.** Med AIL0.1-nettet gav 25 sims korte,
   avgjorte parti (3 s/parti); med AIL0.2-nettet gjev same setting 182
   prøver/parti og 88 % av partia når taket. Partilengda styrast av
   nettstyrken, ikkje av sims-talet.

## 6. Publisere en ny motor til SnartSahkku

Motorene appen kan velge mellom står i `engines/engines.json` i SnartSahkku
(én rad per motor: `id`, `name`, `type`, `script`, `weights`). Å publisere en
ny trent modell er tre steg:

```sh
# 1. eksporter med motornavnet som skal vises i S-UCI-handshaken
.venv/bin/python -m sahkku_ai.export --net checkpoints/ail01/c7.pt \
    --out ../SnartSahkku/engines/ail0.1 --name "AIL0.1"

# 2. legg raden i engines/engines.json (eller lag en ny id, t.d. ail0.2)

# 3. deploy
cd ../SnartSahkku && ./sendsahkku.sh
```

Motoren lastes lazy — klassisk CPU er standard, og en NN-modell hentes først
når spilleren velger den. Flere NN-modeller kan ligge side om side; hver modul
holder vektene sine for seg selv, så de kan være lastet samtidig.

## 7. Koble motoren på et GUI

Motoren er en vanlig S-UCI 1.0-motor (nivå A + `info movepool` + `debug` +
`setoption`):

- **SnartSahkku sitt testkit** (kjører et heilt parti som et ekte GUI):

  ```sh
  cd ../SnartSahkku
  SAHKKU_ENGINE="node ../sahkku-ai/engines/sahkku-nn.js" node tests/suci.test.js
  # kast: 4 · trekk: 8 · pass: 0 · omkast: 0 · resign: 0 · info-linjer: 28
  # 31 pass, 0 fail
  ```

- **Motorveljar i SnartSahkku** (fra v1.0.4): menyen har `CPU: Klassisk /
  Nevral (NN)`. Nevral lastar `engines/sahkku-nn.js` + vektene lazy (først når
  du vel han), og spelar gjennom appens eige `SS.applyMove` — altså same
  dommar, same reglar. Tenkjer ~0,5 s per trekk. Klassisk CPU (1–3) er
  standard og fungerer offline utan ekstra nedlasting.

  Bundling = to filer inn i SnartSahkku (`engines/sahkku-nn.js` +
  `engines/sahkku-nn.weights.json`) pluss `js/nn-adapter.js`:

  ```sh
  .venv/bin/python -m sahkku_ai.export --net checkpoints/ail01/c9.pt \
      --out ../SnartSahkku/engines/sahkku-nn
  ```

- **I nettleseren** (ingen Node-globaler; samme fil, samme vekter):

  ```html
  <script src="engines/sahkku-nn.js"></script>
  <script>
    const w = await (await fetch('engines/sahkku-nn.weights.json')).json();
    NN.exported.loadWeights(w);                       // atob-basert, ingen deps
    const st  = NN.exported.SS.fromFEN(fen);
    const enc = NN.exported.encodeState(st);
    const r   = NN.exported.netForward(enc.planes, enc.glob);   // policy + value
    const ids = NN.exported.legalIds(st);                       // lovlege aksjons-ID-ar
  </script>
  ```

  `tests/test_browser_path.js` kjører nettopp denne stien i en VM-kontekst uten
  `require`/`module` og sammenligner med Node-stien.

## Testene

```sh
./tests/run_all.sh                      # alt, 40 fuzz-parti
FUZZ_GAMES=200 ./tests/run_all.sh       # full paritetskjøring
```

| test | hva den beviser |
|---|---|
| `tests.test_rules` | terningreglene som kjøyrbar spesifikasjon: X vekkjer, omkast berre når X er ubrukeleg, X > III > II, resten av terningane er tapt |
| `tests.test_features` | aksjons-ID-er entydige, `decode_id` runder, FEN runder, masker lekker ikke |
| `tests/fuzz_parity.py` | Python-motoren = JS-motoren: pool, FEN og trekkutførelse, 1:1 |
| `tests.test_net_parity` | PyTorch og rein JS regner samme nett (plan, policy, value) — mot checkpointet vektene kom fra |
| `tests/test_browser_path.js` | motoren laster uten Node-globaler og gir samme svar som i Node |
| `tests/suci.test.js` (i SnartSahkku) | protokollen: handshake, `diceroll`, `go`, lovlige `bestmove`, heile partiet |

## Datastrukturer

**Shard** (`.npz`, komprimert): `planes (N,720) f32` · `glob (N,10) f32` ·
`policy (N,8282) f32` · `value (N,) f32` (i `[-1,1]`, spilleren-som-skal-trekke
sitt syn).

**Aksjonsrom (8282, fast og tett):**

| område | ID-er | innhold |
|---|---|---|
| `REL_BASE` | 0–179 | løslating: side(2) × klasse(soldat/dronning) × målfelt(45) |
| `MOVE_BASE` | 180–8279 | trekk: klasse (M/K/Q/k) × fra(45) × til(45) |
| `SPEC_BASE` | 8280–8281 | `pass`, `reroll` |

En aksjon er identiteten `(bokstav, fra, til)` som paritetsproben bruker;
lovlighet maskeres ved beslutning. **16 plan** à 3×15: utløste soldater per side,
soldater på hjemrad, dronninger (utløst/hjemme), konge (nøytral/M/K), aktiv
terning, stakktelling, andre-pass-markør på midtrekka. De 10 globale tallene
inkluderer `g[6]/g[7]` = **dronninga er utløst** (ikke bare «finnes» — en
slått ut dronning avslutter partiet, så den gamle betydningen var konstant 1.0
og nettet kunne ikke lære noe om aktiveringstidspunkt).

## Ytelse og GPU-notater

### Kvar tida faktisk går (profilert, `cProfile`, 3 parti / 1 arbeidar)

| del | tid | merknad |
|---|---|---|
| nettet framover (conv/bn/relu) | **35 %** | 2,0 ms per forward — **same for batch 1, 8 og 64** |
| regelverket i Python (`occ`, `soldier_at`, `legal_moves`, `move_id`) | **~30 %** | `occ` åleine 11 % (70 000 kall), `soldier_at` 2,4 mill. kall |
| Python/torch-dispatch | ~15 % | 110 000 modulkall per 3 parti |
| treninga | ~4 % | 900 steg ≈ 25 s mot ~600 s per syklus |

**Hovudfunnet: GPU-en er ikkje berekningsbunden, han er *latens*bunden.**
Eit forward kostar 2,0 ms same om batchen er 1 eller 64 rad, fordi grafen er
~15 kjerner som kvar kostar ~0,13 ms i launch — P100-en sit og ventar.
Målt tak under produksjonslast: ~900–1000 forward/s, uansett kor mange
arbeidarar som spør.

Det forklarar òg at **flere arbeidarar gjer vondt verre**: 8 arbeidarar gav
63 parti/min, 12 gav 45 (kvar arbeidar vart 2× treigare av å stå i same
GPU-kø). `OMP_NUM_THREADS=1` endra ingenting — det er ikkje CPU-sult.

| grep | før | etter |
|---|---|---|
| `soldier_at` (memo-tabell i staden for `_mod` per kall) | 10 s / 3 parti | **7 s / 3 parti** |
| same, med 8 arbeidarar | 23 s / 24 parti | 23 s (GPU-køya er taket) |

### Vektorisert sjølvspel (`--k-games`, målt)

`sahkku_ai/fastplay.py` held K parti i lufta per arbeidar og sender alle
utspelingane deira i éin forward per ply, med CUDA-graph på fast batch:

```sh
.venv/bin/python -m sahkku_ai.selfplay --net checkpoints/ail01/c9.pt \
    --outdir runs/x --games 48 --workers 8 --k-games 8 --batch 64 --device cuda
```

| oppsett | parti/min |
|---|---|
| klassisk, 1 arbeidar | 20 |
| vektorisert k=8, 1 arbeidar | **51** (2,4×, `graph=ja`) |
| klassisk, 8 arbeidarar | 63 |
| vektorisert k=8, 8 arbeidarar | 64 |
| vektorisert k=8, 12 arbeidarar | 63 |

**Så: 1+2 gjer éin arbeidar 2,4× raskare, men produksjonskøyringa (8–12
arbeidarar) står på same ~64 parti/min.** Måling under køyring syner kvifor:
8 arbeidarar ligg på ~99 % CPU kvar (792 % totalt) medan GPU-en er 100 %
oppteken — når mange arbeidarar deler same GPU, er det Python-regelverket som
er veggen, ikkje forward-kostnaden. Skal produksjonsløkka verta raskare må
regelarbeidet ned (opptaksindeks, vektoriser `encode_state`, færre `legal_ids`
per ply), eller utspelingane kortast ned.

**Ikkje slått på som standard** (`--k-games 1` er klassisk løype). Grunn:
z-fordelinga i dei vektoriserte shardane er annleis — 20 % av avgjerdene har
|z| > 0,5 og 18 % er nøyaktig 0, mot 92 % / 2 % i den klassiske løypa. Det kan
vere rimeleg (meir blanda utfall når åtte born får kvar sine terningar), men
det er ikkje verifisert mot treningskvalitet enno, so løypa vert verken brukt
i pipeline-en eller påstått å vere ekvivalent før det er målt.

### Kva som faktisk ville gjort løkka raskare (målt/estimert)

1. **CUDA-graph per fast batch-storleik** — målt på denne P100-en:
   batch 8: 2,01 → **0,49 ms (4,1×)**; batch 64: 1,97 → 1,46 ms. Sidan nettet er
   65 % av tida, er dette den største einskildgevinsten (~2–2,5× på heile
   syklusen). Krev faste batchar (padding) og statiske innbuffera.
2. **Buntlegg fleire avgjerder per forward** (vektorisert sjølvspel): GPU-en tek
   same 2 ms for 8 som for 64 rader, so 8 avgjerder kan dele éin forward.
   ~2–3×, medium inngrep.
3. **Kutt utspelingane og bootstrap med verdinettet** (t.d. 25 ply i staden for
   heile partiet): kuttar både forwardar og regelarbeid ~3×. Endrar måla, so det
   må validerast på tap-kurva.
4. **Regel-mikrooptimalisering** (opptaksindeks per stilling, vektoriser
   `encode_state`): ~1,3×, låg risiko — `soldier_at`-memoet over er fyrste steg.

Med 1+2 (eller 3) er 9 syklusar ~2 timar → ~35–45 minutt. Med berre 8 arbeidarar
og dagens kode er taket rundt 600 s per syklus.

- P100 (16 GB, cc 6.0): 8 sjølvspel-arbeidarar gir ~63 parti/min; ~330 MB VRAM
  per arbeidar. Sjekk med
  `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.
- Treningssteget er billig (900 steg på ~25 s); flaskehalsen er sjølvspelet.
- Er GPU-en opptatt: `--device cpu` fungerer (bare langsommere).
- Minne: shards holdes i RAM som tett f32 (`8282 × 4 B ≈ 33 kB` per prøve), så
  ~30 000 prøver ≈ 1 GB. `--window` i pipelinen begrenser hvor mange sykluser
  treningen ser samtidig.

## Feilsøking

| symptom | årsak / fiks |
|---|---|
| `error go utan terningar` | GUI-en må sende `diceroll` (eller replay med `roll`-token) før `go` |
| `error diceroll treng 3 terningar` | `diceroll` tar **tre** ord (`diceroll X 2 _`); i FEN er de samme fjesene **ett** token (`X2_`) |
| `pool=[]` / alle trekk «ulovlege» i en egen DOM | `SS.roll()` setter **ikke** `st.phase = 'move'`; GUI-en gjør det (`js/main.js`) — samme gjelder `SS.rerollX()` |
| `RuntimeError: Input type ... should be the same` | `Net.load(..., map_location='cuda')` flytter bare tensorene; bruk `Net.load(p).to(device)` |
| `Cannot re-initialize CUDA in forked subprocess` | bruk spawn (gjøres i `selfplay.run_selfplay`) |
| `ulovleg trekk «...»` i egen motor | ikke dikt opp trekk: returner alltid noe fra `legalMoves`-poolen (eller `pass`/`reroll`) |
| motoren «henger» | motoren skriver nøyaktig én `bestmove` per `go` og ingenting ellers |
| treg motor | `setoption name Lookahead value 0` (rein policy, 5× raskare) |

## Begrensninger og veien videre

Dette er et *komplett, testet og dokumentert rammeverk* — ikke et sterkt
sáhkku-nett. Ærlig status:

1. **Ingen tresøk.** Rotvalget er N policy-sampla utspillinger (ingen MCTS), og
   policy-målet er derfor støyende: π har i snitt bare ~2.8 aksjoner i støtta og
   entropi 0.74 nats. Standard AlphaZero bruker hundrevis av simuleringer per
   stilling og verdinettet som løvverdi.
2. **Skala.** 3 × 500 parti ≈ 39 000 stillinger; AlphaZero-klasse spill bruker
   millioner. `--games`, `--cycles` og flere arbeidere skalerer rett opp.
3. **Verdinettet er bare delvis utnytta.** Motoren bruker det via `Lookahead`
   (85 % mot seg selv uten), men treningsløkka bruker det ikke som løvverdi i
   utspillingene.
4. **Taket er målstøyen, ikke optimeringa.** 4000 treningssteg gir marginalt
   bedre val-tap og *dårligere* value-MSE (overfitting på støy) — se
   `runs/<run>/training_log.jsonl` og `history` i checkpoint-metaen.
5. **Ingen sidespegling.** Nettet ser absolutte posisjoner. Det gir en målbar
   Menn/Giera-skjevhet (13 % vs 40 % mot L2); speiling som augmentasjon ville
   doblet mengden trening per side.

Neste steg, i rekkefølge: (a) MCTS med verdinettet som løvverdi og
`--rollouts` som simuleringsbudsjett, (b) sjølvspel som *spiller* med
lookahead/1-ply verdisøk, (c) 10–50× flere parti per syklus, (d) større nett
(96×6 = 238 ms/trekk uten lookahead).

## Lisens

AGPL-3.0-or-later. Reglene i `vendor/` er hentet ordrett fra
[SnartSáhkku](https://github.com/larsnygard/SnartSahkku) (AGPL-3.0), og
`engines/sahkku-nn.js` inneholder dem, altså gjelder AGPL også for den fila.
