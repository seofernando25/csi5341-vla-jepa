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

Dream-RSI:

```sh
uv run --no-sync vla-rsi --help
```

The pre-GPU check is the test suite. The first live pilot was a harness check in a scratch
directory, not evidence (2026-11-15): 4 nodes, 2 rounds, 6 min 14 s wall, agent cost $0.03. All four
solutions were agent-written INT8 variants scoring 8/10 on smoke, removing 25.4 %, 35.5 %, 38.6 %
and 44.2 % of memory. Do not cite these numbers as results.

Protocol additions, dated 2026-11-15 and made before any recorded run: `dream_rsi.discovery_tools`
(`read,write,edit,bash`), `dream_rsi.policy_dev_tools` (`read,write,edit`), `agent_timeout_s`,
`eval_timeout_s` (1800 each) and `gpu_slots` (2). `allowed_methods` and `agent_tools` were removed.

## Status

- Protocol frozen.
- Precision-surface pre-registration: [precision_surface_plan.json](precision_surface_plan.json).
- Recipes: 13 precision arms under `recipes/`.
- Evidence: none yet. No smoke or development run has been launched.
- Report: stale. Do not quote pre-cleanroom numbers.
- Old namespace: deleted from the working tree; recoverable at `pre-cleanroom-20261115`.

## Harness trials (synthetic, not evidence)

- Calibration `haiku-low-v1` (`vla-rsi calibrate`, synthetic worlds, no GPU; file `outputs/rsi-calibration/haiku-low-v1/calibration.json`). Agent: Claude Code CLI, Haiku, low effort, tools `read,write,edit`.
- Verdict `improved_and_generalised`. Steps: 3 (2 accepted, 1 not better). Train mean V 0.4853 to 0.5267 (delta +0.0414). Held-out mean V 0.5245 to 0.5394 (delta +0.0149).
- Cost $0.0615 over 3 calls and 52 turns. Wall time 140.7 s.
- This run used the pre-fix harness, before `PrefixView`.
