# BLANKA.3 — nattleging: kvifor Slår-blankas ikkje CPU-en

*Køyrd 2026-11-14 00:14–01:40. Alle tal er målte parti (SPG-filer + opptak i
`analysis/blanka3_overnight/`). Analyseverktøy: `tools/analysis_stats.py`,
`tools/analysis_agg.py`, `tools/queen_autopsy.py`, `tools/policy_audit.py`,
`tools/punish_rate.py` (alle ny-skrivne i natt).*

## 0. Oppsummert

**BLANKA.3 er ikkje svekka — ho er trenad på ein annan spelvariant.**
Mot FLOKK L3 i **Nesseby** (varianten ho trenar i) spelar ho 53 %.
Mot same CPU i **klassiske reglar** (varianten benke-porten måler i) spelar
ho 6–8 %. Tapet i benken er ein **rule-variant-mismatch**, og bak den
mismaten ligg to strukturelle svekkar: eit **matt verdihovud** (±1.000
overalt) som gjer lookahead-luka si blind, og **lita, skjev treningsfordeling**
(100 % Nesseby, éin straffar-motstandar, 200 parti/syklus, vindauge 2).

## 1. Simuleringane som vart køyrde

| # | Oppsett | Parti | Blanka-score |
|---|---|---|---|
| M1 | blanka3 (Nesseby) vs classic3-default (L2), Nesseby-modus | 30 | **53.3 %** |
| M2 | blanka3 spegel (Nesseby) | 24 | 41.7 % A — **Menn-sida 20-4, Giera-sida 4-20** |
| M3 | classic3-spegel (Menn-L2 vs Giera-L2) | 14 | 64.3 % A (CPUane seg imellom: 50/50-aktig) |
| M4 | blanka3 vs classic3 **med Level=3** i Nesseby-modus | 30 | **UGYLDIG: 30/30 feil** — CPU-referee sin Level-3-søk spelar trekk som er ulovlege under Nesseby-reglane («ulovleg O1-O2») |
| M5 | blanka3 vs classic3 **Level=3**, klassiske reglar (m5/gate-repro), seed 31337/2026 | 16/40 | **6.3 % / 7.5 %** |
| M6 | blanka3 vs classic3 **Level=2**, klassiske reglar | 16 | **68.8 %** |
| M7 | blanka3 vs classic3 Level=3, Nesseby | 16 | UGYLDIG (same CPU-bug som M4) |
| Ab1 | blanka3 **Lookahead=0** vs L3, klassisk | 16 | **0.0 %** |
| Ab2 | blanka3 **Deep=2** vs L3, klassisk | 16 | **0.0 %** (0-16) |
| Ab3 | blanka3 **Lookahead=8** vs L3, klassisk | 16 | **12.5 %** (berre +5 pp mot LA4) |
| Pun | blanka3 (MCTS 45 sim) vs **Punisher** (treningsmotstandaren), Nesseby | 16 | **75 % for nettet** |

Benkefila `runs/blanka3_bench.txt` (12.11.) melder: 10 % mot NAIS.7 (Nesseby),
10 % mot FLOKK L3 (klassisk), 23 % mot NAIS.9, 56.7 % mot BLANKA.2.

## 2. Kvifor blanka ikkje slår CPU-en — målte årsaker, rangerte

### 2.1 Rule-variant-mismatch (den dominerande effekten)
- `runs/blanka3_params.json`: `nesseby: true, nesseby_frac: 1.0`. **Alle**
  treningsparti er Nesseby.
- Benke-porten `classic3:40` køyrer `EVAL_NESSBY=0` — klassiske reglar
  (`tools/bench_list.py:55-62`).
- Same motor, same CPU, same reglar som treniga: **53 %** (M1). Bytt til
  treningsportens reglar: **7.5 %** (gate-repro, nøyaktig reprodusert i natt).
- Nettoeffekt: nettet spelar 200-syklusar med Nesseby-prestasjonar i hovudet:
  landingskap av kongen krev «porten» (nokon nådd motstandaren sin heimrad —
  g[25]/g[26]). I klassiske reglar er porten ALLT open. Nettet undervurderer
  kongefaren i klassisk spel — og FLOKK L3 kaprar kongen sin tidleg og
  gjer han om til raider (doktrinen ligg i AIL0.7-notata: 6/6 parti,
  4.2 drap/parti, 1.9 felt frå fienden).
- Boks-testen M6 (L2 i staden for L3, klassisk): **68.8 %** — svekka er
  spesifikt L3 sin taktikk + port-luking, ikkje klassiske reglar åleine.

### 2.2 Matt verdihovud → lookahead-luka er blind
- Midtspøling i spegel-parti: **55 % av stillingane har |v| > 0.9**, mange
  nøyaktig ±1.000 (målt i natt; `val_value_mse = 0.0` i 12+ syklusar av
  treningshistorikken — z er ALLT ±1 (partiemål), så MSE 0 = metning, ikkje
  perfeksjon).
- Queen-autopsi (tapte dronningeslag, parti for parti): i partiet (gate
  parti 0): policyen si rekkjefølgje: trygg dronning `QK2->I2` 0.450,
  dødeleg `QK2->M2` 0.441 — **policyen ser faren**, men LA4 ser barna si
  VERDI, og alle barna er metta (−1.000 vs −1.000) → valet blir kast mynt.
  Andre parti: nettverdi +1.000 idet ho heng dronninga (feilverdi), og
  «dødelege trekket» ligg på policy-topp 0.42–0.58 medan trygge svar ligg
  0.02–0.17 — metninga gjer at ein høg-policy trekk ikkje kan korrigerast
  av verdien.
- Abiaslar: LA0 → **0 %**. Heil styrke hennar i mot L3 ligg i LA4 +
  CAP_MARGIN (1. ply lookahead på verdien). Deep=2 → **0 %**: når
  motstandaren sine svar òg vert verdit av det metta hovudet, ser alle
  svar «±1-vinnande» ut for motspelarane, verdiskalaen kollapsar og
  trekkvalet blir degenerert. Utan avmetning av verdien er alle djupare
  søk verdi-lause (og skadelege). LA8 → 12.5 %: meir søk på same metta
  verdi hjelper nesten ikkje (+5 pp), og Deep gjer det berre verre.

### 2.3 Straffemønsteret: få brikker ute → dei vert plukka
- Punisher-probe: nettet held i snitt **2.48 soldatar ute** og taper
  **7.8 utløyste brikker per parti** — sjølve straffemekanismen
  (menneskestrategien frå `punisher.py`: aktiver opp til 6, plukk den eine
  aktive). Mot Punisher sjølv (treningsmotstandaren, Nesseby) vinn ho 75 %
  — straffaren straffer ho altså IKKJE i Nesseby; FLOKK L3 gjer det i klassisk.
- Spegel-parti (M2): vinnaren held **4.68 soldatar ute**, taparen 1.99
  (klassisk CPU-spegel: 4.7 — CPUane gjer det rett). `avg_exposed` taparlag
  0.31 mot vinnarlag 0.58: taparen gjer seg LITEN (ikkje trygg — passiv),
  vinnaren tek initiativet med fleire truga samstundes.
- Heile spelinga endar nesten alltid med **{dronning}** (30/30 og 39/40) —
  partiet døyr på dronninga, så dronning-tryggleik ER spelet.

### 2.4 Side-asymmetri i spegel (Menn 20-4 / Giera 4-20)
- Same nett, same opningskatalog, same spegel: Giera-sida taper 5:1.
- Årsak: **treningsdata-fordelinga**, ikkje brettet: 15 420 prøver er
  dominerte av Menn-first-mønster (first er 50/50, men planar er absolutt
  orientert — `features.py`: inga spegling, sidene er ALDRI bytte —
  klassen M = menn-soldat, K = giera-soldat).
- Følgjer: mot tilsynelatande «like» motorar (eiga spegel) er ho systematisk
  svak som Giera; i gate-repro: A som Giera 1-2, som Menn 0-0 —
  benke-tala blandar sidene, så 10 %-talet er òg delvis dette.

### 2.5 Treningsøktas storleik (underliggjande)
- 53 syklusar × 200 parti, 450 steg/syklus, **window 2** → effektivt
  treningsset ≈ siste 2 syklusar (≈ 600 parti, 15 420 prøver).
  Gamal data kastast; ny stil (Nesseby+hold) har aldri møtt L3, og L3-
  erfaringa frå earlier runs (NAIS/KAISER-linja) er borte for BLANKA-linja
  (init = blanka2, som er ei «few»-linje).
- `--hand-style hold` på nettsida: nettet trenar MED handikap (løysing ×2.0,
  landing på slagfelt ×0.15). Policy-tauprogrammet lærer den vekta —
  men audit viser at rå policy likevel berre 1.1 % av trekka landar på
  slagfelt: straff-vektene har lært forsiktig spel, og i klassisk spel er
  forsiktig-utan-port-viden = passiv = død (2.3).

## 3. Konklusjon

BLANKA.3 slår CPU-en **når ho spelar den varianten ho er trenad i** (Nesseby
53 %, mot Punisher 75 %, mot BLANKA.2 57 %). Ho slår han ikkje når benken
bytt til klassiske reglar mot L3, fordi:
1. Nesseby-port-prestasjonen er feil i klassisk spel (kongen for utsett),
2. verdihovudet er metta til ±1, slik at 1-ply lookahead og CAP_MARGIN
   samanliknar myntkast,
3. spelestilen er passiv (2.5 brikker ute) og straffbar av L3 sin
   fleirtruge-spel,
4. ho er systematisk svakare som Giera i spegla pga. treningsfordelinga.

## 4. Plan for å betre AI-en (prioritert)

**P0 — Benken må måle det vi vil ha (gjer port-målinga sant)**
1. Utvid `classic3`-porten til to delar: `classic3c` (klassisk, noverande)
   og `classic3n` (Nesseby). **Krev ≥55 % på BEGGE.** Med ein variant-eige
   benk forsvinn misforståinga som føregjekk her.
2. Fiks CPU-referee sin Nesseby-bug (M4/M7: Level-3-søket genererer trekk
   ulovlege under Nesseby-porten → 100 % FEIL-parti). Utan denne fiksens
   kan vi ikkje måle noko nett mot L3 i Nesseby i det heile.
3. Legg spegel-sidetal til rapporten (A som Menn / A som Giera finst frå
   før — vis Giera-talet som eige hovudtal, det var 0-20 heile veka).

**P1 — Avmetning av verdihovudet (gjer search meining)**
4. Bland z-mål: `z = α·utfall + (1−α)·søksverdi` (α≈0.75) eller label
   smoothing (klipp |z| til 0.9) i `selfplay.py` (MCS-bana og
   play_vs_opponent-bana har begge hard `z=±1`).
5. Mål `|v|>0.9`-andel som eigen trend (verktøyet finst i natt:
   value-scan i `analysis/`-mønstret) — port: **<30 %** metning før publisering.
6. Etter 4–5: køyr Deep=2/3- og LA-sweep på NYTT nett — same abiaslar som i
   natt; LA4/DEEP2-målinga no er ubrukeleg med metta verdi.

**P2 — Tren på begge reglar + mot L3**
7. `--nesseby-fraction 0.5` (blanda, akkurat som flagg-stilen gjev
   «eitt nett per variant», g[15]).
8. Bland inn `--train-opponent classic3` (L3):
   Punisher-straffen er målt uverknadsfull i Nesseby (75 % for nettet);
   L3 er rovdyret som faktisk slår ho. Klassisk-L3-vs-Nesseby-data gjer
   nettet port-viden i klassisk spel (L3 spelar berre klassisk; blanding
   0.5/0.5 gjev klassisk-L3 + Nesseby-sjølvspel).
9. Større økter for BLANKA-linja: `--games 400 --steps 900 --window 4`
   (Nais-linja brukte 400/900/2; BLANKA ligg på halve. 15 420 prøver
   totalt er litt for lite for 1.6M parametrar).

**P3 — Side-balansering**
10. Data-augmentering med sideskifte (byt M↔K-klassane i action-rommet og
    spegl planane 0↔1, 2↔3, 4↔5, 6↔7, 8↔9/10, 12↔13, 14↔15 + bytt forteikn
    på dei POV-relative globalane g[0]) — halver treningsdata-behovet og
    drep Giera-svekken. Eit sideskifta parti er matematisk gyldig sidan
    brettet er symmetrisk om midtraden (sjå `features.py`-design:
    «absolutt orientering, side-asymmetrisk syklus») — kontroller at
    syklus- og heimrad-tolkinga følgjer med i transformasjonen.

**P4 — Dronning-tryggleik som målt tal (verifiser at P1–P3 verkar)**
11. Køyr `tools/queen_autopsy.py` + `tools/punish_rate.py` (skrivne i natt)
    på port-benkane: mål «eiga dronning teken i N/16 parti» og
    «dronninga til motstandaren slagbar ply/parti». Noverande
    BLANKA.3-vs-L3: 14/16 tap med eiga dronning teken. Port: <50 %.
12. CAP_MARGIN bør samanlikne **DEEP-verdiar** (motspelar-svar) på
    barna, ikkje berre eigen verdi — med avmet verdien frå P1 blir dette
    ein reell test av 2-ply; om LA4/DEEP2-vegen fell, vurder MCTS i
    partispel (`sims≥25`, `mcts.py` finst allereie, 45 sim = 105 ms/trekk
    i Python; JS-side treng eigen port — ikkje gratis, så P4 sist).

**Rekkefølgje:** P0 + P1 same kveld (benk + avmetning = neste
`tools/overnight.sh --run blanka4`), P2 + P3 i blanka4-konfigurasjonen,
P4 som port på blanka4-eksporten. Målt suksess: classic3c ≥55 %,
classic3n ≥55 %, spegel ≥48 % på begge sider, eiga-dronning-talen <50 %.
