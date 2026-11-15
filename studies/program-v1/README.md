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
- Dream-RSI cycles: agent logs, sandbox logs, `tree.json`, `live_cycle_manifest.json`,
  `score.json`, `development.json` (with per-task successes), selected policy path and code hash.

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

## Dream-RSI run, 2026-11-15: invalid nodes, preliminary results

One run, 3 cycles, 24 probes, 7.55 h. Evidence: `cycles/` (trees, manifests, per-node scores) and
`policies/` (policy steps, replay evidence).

**Invalid nodes.** `iter_002-b1-a3`, `iter_002-b2-a1` and `iter_002-b2-a2` scored 1.0 with 0
resident bytes through an evaluator exploit, not through compression: the workspace parked the bf16
weights on the CPU and restored them on the first inference call, after the memory measurement. The
loop's `best_score` 1.0 and the cycle-3 promotion (`iter_002-b2-a1`) are therefore not results. Do
not report or plot them. The exploit and the evaluator fix are recorded in
[exploit/README.md](exploit/README.md).

**Preliminary results, pending re-measurement.** Development phase, 100 trials per node, same
states:

| Node | Memory removed | Development |
|---|---:|---:|
| `iter_000-b0-a3` | 86.9 % | 80/100 |
| `iter_001-b1-a3` | 84.0 % | 84/100 |
| reference `B16` (uncompressed) | — | 82/100 |

A keyword scan of every `proposal.md` found no hiding or restore language in cycles 1 and 2, and
their resident bytes are 0.72 to 3.9 GB. A scan is not an audit: these numbers stay preliminary
until each reported node is re-measured with the fixed evaluator. Reserved-500 has not run.

Dream-RSI runs (`vla-rsi start`, `status`, `stop`, `watch`): [docs/COMMANDS.md](../../docs/COMMANDS.md).
The pre-GPU check is the test suite. Synthetic calibration runs are harness checks, not evidence.

Protocol additions, dated 2026-11-15 and made before any recorded run: `dream_rsi.discovery_tools`
(`read,write,edit,bash`), `dream_rsi.policy_dev_tools` (`read,write,edit`), `agent_timeout_s`,
`eval_timeout_s` (1800 each), `policy_timeout_s` (1200) and `gpu_slots` (2). `allowed_methods` and
`agent_tools` were removed.

## Status

- Protocol frozen.
- Precision-surface pre-registration: [precision_surface_plan.json](precision_surface_plan.json).
- Recipes: 22 arms under `recipes/`; queue: 59 cells, 431 estimated minutes. All 59 cells ran.
- Evidence: one Dream-RSI run (3 cycles, 24 probes); three cycle-3 nodes invalid, two development
  winners preliminary (see the section above).
- Report: stale. Do not quote pre-cleanroom numbers.
- Old namespace: deleted from the working tree; recoverable at `pre-cleanroom-20261115`.

