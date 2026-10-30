# NAIS.6 — taktikk-duellar: aktivering og kongedoktrine

Skrive før køyringa, so «godt nok» ikkje vert avgjort klokka 03:00.

## Spørsmåla

1. **Kor mange soldatar bør ein aktivere, og korleis bør dei fordelast?**
   Få, hardt framover (konsentrasjon) mot mange, spreidd over breidda.
2. **Kva gjer ein etter at kongen er kapra?** Bruke han offensivt inn mot
   fienden, eller verne han i bakre rekke og sende soldatane fremst?

## Dette er eit kontrollert forsøk, ikkje berre trening

Kvar kamp er ein **duell**: éi side spelar stil A, den andre stil B, og sidene
byter kvar kamp (`selfplay.style_pair_for_game`). Difor er vinstprosenten i
duellane eit svar på spørsmåla med stort n — 800 parti gjev ~400 per par — medan
benken mot NAIS.5 seier om nettet vart sterkare av å ha spelt begge sider.

| stil | kva han gjer | vekt |
|---|---|---|
| `few` | løyser FÅ soldatar (etter 2 alt ute), dyttar dei som er ute hardt fram | løysing ×0,30, soldatframrykk ×1,6 |
| `many` | løyser SÅ MANGE som mogleg | løysing ×2,5 |
| `kingfwd` | med kongen: trekk som kjem nærare fienden | ×2,5 (bort ×0,4) |
| `kingguard` | med kongen: hald han attende, send soldatar fram | bort ×2,5 (fram ×0,4), soldat ×1,6 |

Stilen vert vekta **før** policy-målet vert skrive, so nettet lærer stilen — ikkje
berre stillingane som oppstår. (Den gamle `--aggressive` vekta etter målet; det
vart retta i same omgang, og `tests/test_styles.py` pinnar rekkjefylgja.)

## Konfig

```bash
./badass-trainer.sh --run nais6 --init engines/nais5-nn.pt \
    --cycles 2 --hours 0 --games 400 --sims 45 \
    --nesseby --nesseby-fraction 1.0 --styles
```

- init: **NAIS.5** (`engines/nais5-nn.pt`, 76,7 % mot NAIS.4)
- 2 syklusar à 400 parti, 45 sims, `--max-plies 2500`, lr 3e-4, vindauga 2
- anslått ~110 min per syklus på P100/6 arbeidarar, ~57 min på 3070 Ti/11

## Måltal (før køyringa)

Åtferd — **stilane må bite**, elles måler køyringa ingenting. Frå ein probe med
NAIS.5, 24 parti, 12 sims, 4 arbeidarar:

| mål | før | krav etter køyringa |
|---|---|---|
| løyste soldatar ved trekk 40, `few` mot `many` | 0,0–1,2 mot 1,8–4,0 | `few` skal framleis liggje **≥1,0 under** `many` |
| kongens avstand til fienden sin heimrad, `kingfwd` mot `kingguard` | 0,0 mot 1,3–2,0 | `kingfwd` **≤0,5**, `kingguard` **≥1,0** |
| duellar avgjorde (ikkje avkorta) | få — 12 sims og 300-ply tak | **≥80 %** av partia med ein vinnar |

Styrke — det som avgjer publisering:

| bench | NAIS.5 (referanse) | krav |
|---|---|---|
| mot NAIS.5, Nesseby, n=30 | — | **≥55 %** for å bli standard |
| mot klassisk nivå 3, n=40 | 97,5 % (39–1) | **≥90 %** (ingen standardregel-regresjon) |

Duell-svar (frå treningspartia sjølv, ikkje benken):

- `few` mot `many`: kven vinn? Krav til rapportering: talet på avgjorde parti,
  og om skilnaden er større enn 2σ for eit tipp.
- `kingfwd` mot `kingguard`: same.

## Regel for fiasko

- Stilane **bit ikkje** (måltala over) → køyringa er ugyldig; fiks vektinga og
  køyr om att. Ikkje tolk benken.
- Stilane bit, men nettet vert **svakare** mot NAIS.5 (<55 %) → ikkje publiser.
  NAIS.5 står. Køyringa er framleis eit gyldig svar på kva taktikk som vinn, og
  det skal rapporterast med tala sine.
- Standardregel-benken under 90 % → ikkje standardmotor, sjølv om Nesseby er
  betre. (NAIS.3 er presedensen: 26,7 % vart ståande att i menyen som eksperiment.)

## Etterpå

1. `node tools/eval_match.js` (Nesseby 30 mot NAIS.5, L3 40) — som vanleg.
2. Duell-tala vert lesne frå arbeidaroppsummeringa: `stil: few 12-8 many, løyste@40: few 1,2 vs many 2,6, kongeavstand: fwd 0,2 vs guard 1,8`.
3. Publiser som `NAIS.6` valfri motor; standard berre om han slår NAIS.5.
