# AIL0.7 — plan for the 4-hour session (2026-10-27)

## Hypothesis

AIL0.6's weakness is not style, it is **what the net can perceive**. Measured
on today's engine:

| finding | measurement |
|---|---|
| search is the binding constraint | LA0 62.5 % → LA4 85 % → LA8 92.5 % vs classic L3 |
| the king is used as a raider | captured in 6/6 games by move ~12, then 63 moves and 4.2 kills per game, standing **1.9 squares** from the enemy |
| the buffer effect is real | forced passes: **8 with king vs 46 without** |
| the encoder is blind to all of it | 16 planes are piece placement; the 10 globals are kills/released/queen/dice/clock. No king term, no threat term, no risk term |

So: give the net the doctrine as input, and give it deeper targets.

## Changes in this run

Five new globals (N_GLOBAL 10 → 15), each from a measured effect:

| index | feature | what it encodes |
|---|---|---|
| `g[10]` | `king_mine` | I own the king |
| `g[11]` | `king_theirs` | they own the king |
| `g[12]` | `king_exposure` | 1 − distance/14 to the nearest enemy piece when I own him — "keep him back" |
| `g[13]` | `queen_risk` | my activated queen can be captured **right now** — "activation is risk" |
| `g[14]` | `opp_usable` | P(the opponent's next roll contains a die they can actually use), from an 8-entry table over the 64 rolls: 1 face 57.8 %, 2 faces 87.5 %, 3 faces 98.4 % |

**Checkpoint compatibility.** The globals are the last columns of `pol_fc` and
`val_fc1`, so AIL0.6's weights were *widened* with zero columns
(`tools/expand_net.py`): the new net is mathematically identical to AIL0.6 at
step 0 and can only improve by learning to use the new inputs. Verified:
old columns bit-identical, new columns all zero, all 98 other tensors unchanged.

## Run configuration

```
--run ail07 --init checkpoints/ail06/c6_g15.pt
--games 140 --workers 8 --sims 45 --leaf-batch 8 --c-puct 1.6
--steps 900 --batch 256 --lr 3.0e-4 --warmup 100 --window 2
--ch 64 --blocks 6 --fc 128 --max-plies 2500 --max-seconds 14400
```

Plain self-play — **no handicap**. AIL0.6's queen-delay handicap was confounded
by the MCTS fix, so this run changes exactly one thing (the features) plus
deeper search targets (sims 25 → 45).

## Acceptance tests (must pass before publishing)

1. `king_exposure` doctrine metric rises well above today's **1.9** — the engine
   actually keeps the king back.
2. Forced passes without the king fall below today's **46** (per 6 games).
3. Bench vs **AIL0.6** (40 games, head-to-head): no regression; target > 60 %.
4. Bench vs **classic L3** (40 games): no regression from **85 %**.
5. Technical: net parity 25 positions, S-UCI conformance, engine/rules/archive
   suites, Python↔JS fuzz parity.

If 1–2 move but 3–4 regress, the features are teaching the doctrine at the cost
of strength — report it, do not publish. If nothing moves, the feature or the
targets failed, and that is the finding.

## Morning checklist

- [ ] read `runs/ail07/training_log.jsonl` — cycles, samples, val loss trend
- [ ] export latest ck to `engines/sahkku-nn`, run `tests/test_net_parity.py`
- [ ] bench: vs AIL0.6, vs classic L3 (40 games each)
- [ ] doctrine metrics (king exposure, forced passes, queen activations)
- [ ] if accepted: registry entry `ail0.7`, manifest `default`, sw bump, deploy
- [ ] push both repos; report numbers with the benches that justify them
