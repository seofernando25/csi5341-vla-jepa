# Program v1: clean-room compression evidence

Started 2026-11-15. This namespace replaces the deleted pre-cleanroom compression and Dream-RSI
evidence. The old tree remains in git history at tag `pre-cleanroom-20261115`. Nothing under this
directory may be copied from that tree.

## Question

How far can post-training compression of VLA-JEPA go on LIBERO-Spatial without retraining, and
where does the loss come from: weight precision, execution, or the search procedure?

## Scope

- Model: `lerobot/VLA-JEPA-LIBERO` at `735d9f692981e286ade093b5046627eda876e5d0`.
- Suite: LIBERO-Spatial, tasks 0-9.
- No Qwen training. Calibration is unlabeled and uses development observations only.
- Reserved-500 states are never used for selection.
- Old evidence is not reused, pooled, or cited. Every number in the report comes from a run under
  this namespace.

## Phases

| Phase | Trials | Purpose |
|---|---:|---|
| Screen | 80 fixed frames, no simulator | Reject runtime failures and gross action changes |
| Smoke | 10 | Reject gross failures; promotion gate |
| Development | 100 | Selection and curve estimation |
| Reserved | 500 | Final claim only, after selection is frozen |

Gates:

- Smoke promotion: at least 8/10.
- Development promotion: at least 77/100.
- Final non-inferiority: paired 95% interval above -5 pp against B16.
- A gate is written before the phase starts and does not move after outcomes.

## Experiments

1. **Precision surface.** Language, vision and action separately at INT8/NF4/HQQ3/HQQ2, plus a
   joint vision+action 2-bit cell. 100 development trials per point. Plot success against effective
   bits, not nominal bits.
2. **Format curve.** Decoder-only INT8, NF4, HQQ4, HQQ3, HQQ2, GPTQ and AWQ-style diagonal.
   100 development trials per point.
3. **Joint recipes.** Fixed compositions selected from development; 500 reserved trials only after
   freeze.
4. **Execution ladder.** Stock versus CUDA graph, dense versus packed NF4, one versus four flow
   steps. Timing only, except the final arm.
5. **Dream-RSI clean loop.** Paper-faithful replay and policy rewriting. Static `GridPlan` fixed
   by the protocol; no adaptive `plan_grid` heuristic; no `gate_retry`; no sequential gate in the
   clean path. The selected policy is recorded by code hash.

## Required artifacts

Every run writes:

- `run.json`: command, code revision, checkpoint revision, recipe hash, environment, GPU, protocol,
  timestamps, exit status.
- `episodes.csv`: task, trial, initial-state hash, seed, success, steps, wall time, status.
- `timings.csv` for benchmark runs.
- `summary.json`: successes, intervals, paired differences, peak memory, storage inventory.
- `recipe.json` and transformation events.
- Dream-RSI cycles: agent logs, sandbox logs, `tree.json`, `live_cycle_manifest.json`, selected
  policy path and code hash.

Runtime artifacts stay in `outputs/`. Compact records stay here.

## Commands

```sh
uv run --no-sync vla-compress calibrate
uv run --no-sync vla-compress run --recipe studies/program-v1/recipes/<id>.json --phase screen
uv run --no-sync vla-compress run --recipe studies/program-v1/recipes/<id>.json --phase smoke
uv run --no-sync vla-compress run --recipe studies/program-v1/recipes/<id>.json --phase development
uv run --no-sync vla-compress run --recipe studies/program-v1/recipes/<id>.json --phase final
uv run --no-sync vla-compress-analyze
```

`--phase final` is the reserved-500 phase; the queue runner refuses it without `--allow-reserved`.

## Measurement queue

`vla-compress-queue` runs ordered (recipe, phase) cells from [queue.json](queue.json). It skips a
cell whose `run.json` says `completed`, stops on the first failure that is not a measured outcome
(a nonfinite-action screen is one), and refuses reserved cells unless `--allow-reserved` is passed.

```sh
uv run --no-sync vla-compress-queue --dry-run
uv run --no-sync vla-compress-queue
uv run --no-sync vla-compress-queue --queue studies/program-v1/queue-reserved.json --allow-reserved
```

`queue.json` is the default: B16 screen/smoke/development, the precision ladder, the joint cell,
the format curve, then the execution-ladder benchmarks. Benchmark runs already serialize the packed
archive, so no separate `size` cell is queued. [queue-reserved.json](queue-reserved.json) holds the
reserved-500 cells and is never run by default.

## Recipes

22 files under `recipes/`, all validated by `evaluation.compression.recipes.validate`, which
`vla-compress run` calls before loading a model.

| Group | Files |
|---|---|
| Baseline | `B16` (no changes) |
| Precision ladder | `precision-{lang,vision,action}-{int8,nf4,hqq3,hqq2}` |
| Joint cell | `precision-joint-va-hqq2` (vision HQQ2 + action HQQ2, language BF16) |
| Format curve | `format-lang-{hqq4,gptq,awq}` (decoder only) |
| Execution ladder | `ladder-b16-graph-{4step,1step}`, `ladder-nf4-graph-{4step,1step-dense,1step-packed}` |

Readings taken where the plan was ambiguous:

- The joint cell is the factorial cell the H4 interaction needs: vision HQQ2 plus action HQQ2 only.
  The earlier file also quantized action attention and interfaces at 4 bits, which no single-module
  arm shares.
- Q4 and Q8 are the decoder-only NF4 and INT8 arms, so the reserved anchors and the ladder rows
  reuse `precision-lang-nf4` and `precision-lang-int8` instead of duplicating them.
- The format curve holds HQQ axis at 0, the pre-registered precision-ladder axis, so format is the
  only variable. The plan names no axis for HQQ4.
- The ladder NF4 arms use the deployment shape the 1.5 GB rungs measure: `world_remove`,
  `int8_embedding`, NF4 on text, vision blocks, merger and action. `world_remove` stays out of the
  module curves.
- C1/C2 are not authored: the plan defines them by development outcomes. Append their cells to the
  reserved queue after selection, then freeze. The packed NF4 one-step ladder arm is the
  pre-registered stand-in for the deployment claim.

## Status

- Protocol frozen.
- Precision-surface pre-registration: [precision_surface_plan.json](precision_surface_plan.json).
- Protocol additions dated 2026-11-15 (agent tools, timeouts, GPU slots) are recorded in
  [protocol.json](protocol.json) under `dream_rsi`.
- Recipes: 22 arms under `recipes/`; queue: 59 cells, 431 estimated minutes.
- Evidence: none yet. No smoke or development run has been launched.
- Report: stale. Do not quote pre-cleanroom numbers.
- Old namespace: deleted from the working tree; recoverable at `pre-cleanroom-20261115`.

## Harness trials (synthetic, not evidence)

- Calibration `haiku-low-v1` (`vla-rsi calibrate`, synthetic worlds, no GPU; file `outputs/rsi-calibration/haiku-low-v1/calibration.json`). Agent: Claude Code CLI, Haiku, low effort, tools `read,write,edit`.
- Verdict `improved_and_generalised`. Steps: 3 (2 accepted, 1 not better). Train mean V 0.4853 to 0.5267 (delta +0.0414). Held-out mean V 0.5245 to 0.5394 (delta +0.0149).
- Cost $0.0615 over 3 calls and 52 turns. Wall time 140.7 s.
- This run used the pre-fix harness, before `PrefixView`.
