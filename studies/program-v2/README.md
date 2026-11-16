# Program v2: Dream-RSI clean-room search under the open protocol

Started 2026-11-15. A fresh Dream-RSI run from the baseline policy with no history, under the
protocol revisions that program-v1's three cycles did not run with. The frozen rules are in
[protocol.json](protocol.json); `supersedes` there states what this namespace replaces.

## Question

Under the open policy-development protocol (shell, GPU and network for the policy agent, anytime
steps, live `plan_grid`, and the fixed memory rule), can the search remove more of the
VLA-JEPA-LIBERO policy on LIBERO-Spatial than program-v1's best node at equal or better
development success?

## Scope

- Model: `lerobot/VLA-JEPA-LIBERO` at `735d9f692981e286ade093b5046627eda876e5d0`.
- Suite: LIBERO-Spatial, tasks 0-9. Same seeds, phases, gates, statistics and budgets as
  program-v1.
- The search starts with no trees, no beta sweeps, no scores and the default policy
  (`rsi/policies/optimal.py`). Nothing under `studies/program-v1/` is read by this run and no v1
  node is pooled with a v2 node.
- Reserved-500 states are never used for selection; the lock is on.
- Calibration is the frozen unlabeled artifact `outputs/compression/calibration.pt` (runtime, not
  in Git), the same one program-v1 used. [calibration.json](calibration.json) is its frozen
  description, copied here so this namespace states its own calibration. It carries no robot
  outcomes. `methods.json` is not copied: it pins vendor sources for the measurement recipes,
  which this search does not read.

## Protocol differences from program-v1

| Field | program-v1 (cycles iter_000..iter_002) | program-v2 (starting rule) |
|---|---|---|
| Policy-development tools and environment | `read,write,edit`; no shell, no GPU, no network, five `rsi/` files at `/tmp/rsi_lib` | `read,write,edit,bash`; repo source read-only at `/tmp/repo` without `rsi/synth.py`, GPU visible, network, writable `/tmp/pkgs` on `PYTHONPATH`, runnable `replay.py` |
| Policy step that exceeds its timeout | recorded as `timeout`, workspace discarded | anytime: the file left behind is audited and replayed, recorded `timeout_salvaged`, accepted only if it strictly beats the incumbent |
| Grid for the next cycle | static pre-registered grid; `plan_grid` never called | the selected policy's `plan_grid` decides, clamped by the protocol grid values and the run arguments; the protocol grid is the fallback |
| Memory rule | `max(tensor walk, live CUDA bytes)`, measured after `compress` and before the first forward pass | `max(tensor walk, every payload kept alive anywhere in the process, live CUDA bytes, peak CUDA bytes during the episodes)`, measured after `compress` and again after inference; workspace files over 1 MiB count in full |
| Starting history | cycle 2 and 3 resumed from the trees and policies of the previous cycle | no history: fresh `cycles/`, fresh `policies/`, default policy, cycles 1-3 of v1 not pooled or cited |

Unchanged from program-v1: checkpoint and revision, suite and task ids, seeds, phase sizes
(screen 80, smoke 10, development 100, reserved 500), gates (smoke 8/10, development 77/100,
final non-inferiority 5 pp), statistics (Wilson 95 %, task-stratified paired bootstrap 10,000
resamples), agent (pi on `mistral/deepseek-v4-1-flash-1017`, max thinking), agent and evaluation
timeouts (1800 s), policy timeout (1200 s), GPU slots (1), `beta_grid`, `lambda_parallel`,
`gate_retry` and `sequential_gate` off, reserved lock on.

## Reproduction

From a checkout of the run branch, with `outputs/compression/` and `outputs/evaluation/` populated
as `rsi/sandbox.py` binds them:

```sh
uv run --no-sync vla-rsi start --study-dir studies/program-v2 -- --cycles 3 --probe-budget 8 \
  --max-parallelism 2 --promote-top-k 1 --max-hours 8 --max-agent-usd 20 --policy-timeout 1200
```

Re-running the same command resumes from `loop_state.json`. `vla-rsi status`, `vla-rsi stop` and
`vla-rsi watch --port 8788` take the same `--study-dir`. The pre-GPU check is the test suite; a
synthetic calibration run is a harness check, not evidence.

`--max-parallelism 1` is the fallback when the GPU is shared: one discovery agent and one
evaluation at a time.

## Results

Three cycles, 12 nodes, 5.4 h, $2.32 of the $20 agent budget. No node scored above 0.95, no agent
timed out, no harness-failure streak.

| cycle | best node | removed | smoke | development |
|---|---|---|---|---|
| 1 | `iter_000-b2-a2` | 83.4 % | 8/10 | timeout at 59/100 |
| 2 | `iter_001-b2-a1` | 79.6 % | 8/10 | 82/100 |
| 3 | `iter_002-b2-a2` | 79.1 % | 8/10 | 83/100 |

**Answer to the question: no.** Under the same memory rule, program-v1's two qualifying nodes
removed 83.0 % at 80/100 and 81.6 % at 84/100. v2's best node with a complete development record
removed 79.6 % at 82/100. Opening the policy agent did not buy compression.

The highest-compression node, `iter_000-b2-a2` at 83.4 %, has no development number: its evaluator
run hit the 1800 s limit at 59/100 episodes. The node is 2.7× slower than its parent (1001 ms
against 366 ms median step), so 100 episodes do not fit the budget. It is not selectable.

**Policy development.** Every cycle's accepted steps raised in-sample V (0.703→0.804, 0.511→0.618,
0.552→0.580) and none raised held-out V: 0.524→0.382 in cycle 1, then flat at 0.382. The selected
steps did not generalise.

**Limitation.** All 12 nodes tie themselves to the 53 calibration prompt token ids: the text
embedding is reduced to the rows those prompts use. That is a real compression for a fixed task
suite, not a general one. `iter_002-b2-a0` does it silently, mapping unseen ids to a zero row.

**Reserved.** On 500 held-out episodes `iter_002-b2-a2` scores 481/500 (96.2 %, paired
$+1.2$ pp $[-0.2,+2.8]$ against the reference) and `iter_001-b2-a1` 475/500 (95.0 %, $0.0$ pp
$[-1.2,+1.2]$). The reference is 475/500. Both match it; neither is a better compression than the
constrained search's 83.0 % at 80/100.
