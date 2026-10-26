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

**AIL0.1** (`checkpoints/ail01/c9.pt`, 64×6, 9 sykluser à 600 parti,
**120 093 prøver**, ~2 timar på P100), 30 parti per benk (seed 41):

| benk | poeng |
|---|---|
| mot ref L2 (grådig + støy) | **26.7 %** (8–22) |
| mot ref L3 (deterministisk) | 30.0 % (6–14) |
| `Lookahead=4` mot `Lookahead=0`, samme vekter | 85 % (17–3) |

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

- P100 (16 GB, cc 6.0): 4 sjølvspel-arbeidere gir ~50 parti/minutt og ~99 %
  GPU-utnyttelse; hver arbeider bruker ~330 MB VRAM. Sjekk med
  `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.
- Treningssteget er billig (600 steg på 15 s); flaskehalsen er regelarbeidet i
  Python under sjølvspel, ikke nettet.
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
