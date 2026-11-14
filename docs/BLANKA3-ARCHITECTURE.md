# BLANKA.3 — korleis nettet er bygd

*Dokumentert 2026-11-14 (nattkøyring). Kjelde: `sahkku_ai/`, `engines/blanka3-nn.*`, `checkpoints/blanka3/`, `runs/blanka3_*`.*

## 1. Kva BLANKA.3 er

AlphaZero-aktig nettmotor for sáhkku: eitt konvolutt-nett med policy- og
verdihovud, eksportert til rein JavaScript (same graf, bit-identiske tal) og
levert som S-UCI-motor `engines/blanka3-nn.js` med lagra vekter i
`blanka3-nn.weights.json` + PyTorch-ckpt `blanka3-nn.pt`.

**Målkøyring:** trenast mot *menneskestrategien* (Punisher) i **Nesseby**-varianten,
med handikapet `hold` på nettsida — for at straffaren skal ha noko å straffe.

## 2. Arkitektur (1 600 550 parametrar)

```
Inngang: 16 brettplan (3×15) + 27 globale tal
  inp:  Conv2d(16 → 64, 3×3) + BN + ReLU
  trunk: 6 × ResBlock(64)     [Conv3×3-BN-ReLU-Conv3×3-BN + residual, ReLU]
  Policy: Conv1×1(64→8)+BN+ReLU → flatten(360) ++ glob(27) → FC(387→128) → ReLU → FC(128→8282)
  Verdi:  Conv1×1(64→4)+BN+ReLU → flatten(180) ++ glob(27) → FC(207→128) → ReLU → FC(128→1) → tanh
```

Konfig `cfg = {ch:64, blocks:6, fc:128}` — standardkonfigurasjonen som held
~110 ms/trekk i rein JS (lesbar i nettlesaren). Alle operasjonar er
conv/matmul/BN/ReLU, så `export.py` kan skyggje nettet nøyaktig i JS
(paritetstest: `tests/test_net_parity.py`, S-UCI 41/41 + 25/25 paritetsstillingar
for BLANKA.3-eksporten, commit `fb8310d`).

### Action-rom (8282, fast og tett)
- RELEASE: side(2) × klasse(soldat|dronning) × målfelt(45) = 180
- MOVE: klasse(M|K|Q|k) × frå(45) × til(45) = 8100
- SPECIAL: pass, omkast = 2
Lovlegheitsmasking skjer ved trekkvals tid (maskerte logit = −inf).

### Brettplan (16)
Heime-rad u/løyste for begge sider, løyste soldatar, stabeltal (1/15 per
brikke), annankvars-gjennomgangsmarkør, dronning inne/ute, kongen (Menn/Giera/
nøytral), aktiv terningsflate i hjørne (X/I/II/III). Absolutt orientering —
sáhkku-syklusar er sidesymmetriske, så inga spegling.

### Globale tal (27) — doktrinen som inndata
Bygd lag på lag gjennom AIL/NAIS/KAISER-generasjonane:
- 0–9: side, drap, løyste soldatar, klokke, dronning utløyst (begge), X-teljing, vinnar
- 10–14: **kongedoktrinen** (kongen som buffer; 8 tvungne pass med konge mot 46
  utan), `queen_risk` (dronninga mi kan takast NO), P(motstandaren får eit
  nyttbart kast) frå 8-rads tabell over 64 kast
- 15: **regelmodus** (Nesseby=1) — eitt nett for begge variantar
- 16–17: **bonderisiko** — del av løyste soldatar som står på felt fienden kan
  slå på (det valet som gjer at menneske slår NAIS: nettet såg skilnad på
  «ute» og «utsett» først her)
- 18–26: **KAISER-utvidinga** — syklusposisjon (p.at/60, spelet sin klokke),
  kongen sin avstand frå slottet, dronninga sin avstand frå opningsfeltet,
  X-resursar, aktive terningverdiar, og NESSEBY-NØKLELEN: port for
  landingskap (kongen kan først kaprast når nokon har nådd motstandarens
  heimrad)

Når N_GLOBAL veks, blir gamle checkpoints *fora* med null-kolonner
(`_widen_if_needed`): nøyaktig samme spel, og treninga kan halde fram.

## 3. Treningsløypa (badass-trainer → pipeline → selfplay → train)

Kvar **syklus**:
1. **Sjølvspel/data**: 200 parti (blanka3-oppsettet). Motstandar =
   `punisher` (menneskestrategien: slå trygt, aktiver opp til 6, hald brikka
   attende), nettsida spelar med handikapet **`hold`** (policy-vektar:
   løysing ×2.0, landa på slagfelt ×0.15, landa trygt ×1.3). Alle parti i
   **Nesseby** (`nesseby_frac 1.0`).
2. **Mål**: π = vitjingstal frå MCTS (45 sim-ar, PUCT c=1.6, Dirichlet-rotstøy
   α=0.3) når `sims>0`, elles vekt policy; z = **partiemålet ±1**
   (AlphaZero-konvensjon — søkets verdi blir overstyrt av utfallet).
3. **Trening**: 450 steg, batch 256, lr 3e-4, warmup, **window 2** (berre
   siste 2 syklars data). Tapt policy-kryssentropi + MSE på z.
4. **Benk**: `classic3:40` = FLOKK L3, `EVAL_NESSBY=0`, `EVAL_SETOPT_B=Level=3`
   → **klassiske reglar**. (Dette er poenget seinare i analysen.)
5. **Port**: publisérst nok når benken når målet.

`blanka3` gjekk 53 syklusar, 15 420 prøver totalt, `best_val 0.798`,
eksportert frå `checkpoints/blanka3/c40.pt` (run-meta: `blanka3_c40`).
Init-vekt: `engines/blanka2-nn.pt` (BLANKA.2 var 0 % mot FLOKK L3).

## 4. Motorluket i JS (ruletten, ikkje nettet, som spelar partiet)

`runner_template.js` → `blanka3-nn.js`. Policy-argmax + enkle regel-lag:
- **LA=4**: 1-ply verdilookahead på 4+1 trekk — barna blir verdit av VERDIHOVUDET
- **DEEP=0**: motstandaren sine svar blir **ikkje** verdt (av)
- **CAP_MARGIN=0.12**: trygt slag innanfor 0.12 av beste trekk blir valt
- **REROLL_EVAL=0**: omkast berre når tvungen (gamal oppførsel)
- Materialvektar `MAT` (override-bar via `§MATERIAL§`)

## 5. Kjende kjenneteikn ved checkpointet

- Frå `weights.json`-historikken: `val_value_mse = 0.0` i fleire syklusar
  (alle z = ±1, nettverdiar som alle er ±1 → MSE 0, modellen ser perfekt ut
  mens han eigentleg berre er **matta**)
- Verdi-hovudet er målt i 2026-11-14-spelet: **55 % av midtspielstillingar
  har |v| > 0.9** — sjå `analysis/blanka3_overnight/FINDINGS.md`
- Policy-argmax avviker frå motorvalet i 40 % av trekka (lookahead +
  capture-margin avgjer), og rå policy hengar dronninga si i 0.0 % av trekka —
  likevel taper nettet dronninga si: svaret ligg i det matta verdi-hovudet.
