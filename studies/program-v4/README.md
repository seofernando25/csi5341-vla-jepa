# Program v4: Dream-RSI search for a faster policy

Prepared 2026-11-16. Not launched. The frozen rules are in [protocol.json](protocol.json), the
discovery prompt in [discovery.md](discovery.md). Checkpoint, suite, task ids, seeds, phases,
gates, statistics, agent and budgets are program-v2's; the score is new.

## Question

Can the Dream-RSI search make the VLA-JEPA-LIBERO policy faster on LIBERO-Spatial while it stays
as successful as the reference?

## Score

    speedup = reference median observation-to-action latency / candidate median
    score   = speedup, if the smoke gate passes and the candidate's peak CUDA bytes
              are not above the reference's
    score   = 0 otherwise

The reference is the untouched B16 policy, four flow steps, stock execution path, measured in the
same process on the same frames. The identity scores 1.0; that is the protocol's baseline. Peak
CUDA is compared with a 1 % measurement allowance, because the reference and candidate peaks are
separate runs.

`evaluation/compression/latency.py` measures it. The evaluator runs reference episodes, keeps the
observation batches, and times the untouched policy on them. After `compress` it times the
candidate on the same batches. Each timed call is a full `select_action` with
`torch.cuda.synchronize()` before and after, after 20 warm-up calls, on 60 observations that were
never timed before. Median and p95 are reported. Fewer flow steps, a smaller model or any other
change is allowed if the episodes still pass.

## Guards

Every guard is a failing condition (`guard_violation`, score 0), not a warning.

| Guard | Catches | Test |
|---|---|---|
| Captured clock | patching `time` or `torch.cuda` readers | `test_clock_rebinds_are_detected`, `test_a_patched_clock_does_not_change_the_measurement` |
| Disjoint warm-up and timed frames | a lookup table keyed on the observation | `test_timed_calls_use_frames_never_seen_in_warmup`, `test_a_cache_of_warmup_frames_gains_nothing_on_new_frames` |
| Novel-frame cost | a cache with a slow fallback | `test_a_cached_solution_is_caught` |
| Equivalence | a constant answer, a stale answer, a special-cased `select_action` | `test_a_constant_answer_fails_the_equivalence_check`, `test_a_stale_answer_fails_order_invariance`, `test_a_special_cased_select_action_fails_path_equivalence` |
| Reset cost | work moved out of the timed call into `reset` | `test_work_moved_into_reset_is_charged` |
| Episode overhead | background or asynchronous work | `test_episode_overhead_above_the_reference_fails` |
| CPU and child CPU | work moved to the CPU or another process; new threads | `test_cpu_above_the_cap_fails`, `test_child_cpu_above_the_cap_fails` |
| Peak CUDA | more memory than the reference (1 % allowance) | `test_peak_memory_above_the_reference_fails` |
| Score | the formula and its two zero conditions | `test_score_is_the_speedup_when_the_gate_and_memory_pass`, `test_score_is_zero_below_the_gate_or_above_the_memory` |

`tests/test_latency.py` runs all of them on the CPU, without the simulator or a GPU.

## Open holes

1. A nearest-neighbour cache of actions precomputed from the visible observation bank can be fast
   and may pass the gate. The equivalence check cannot prove the action is the policy's, because
   a compressed policy has no reference action.
2. A detached process that is reparented is not counted by `RUSAGE_CHILDREN` and not seen as a
   child pid. Only children started during the timed calls are checked.
3. GPU busy time is CUDA-event elapsed time on the default stream, not CUPTI counters. A kernel
   on a side stream still lands inside the device-wide synchronize, so it is charged to the wall
   time, but per-kernel attribution is not measured.
4. The reference latency is measured once per evaluation in the same process, before
   `compress`. The GPU is shared: four identity check runs scored 0.77, 1.02, 1.03 and 1.04, and
   the absolute median moved from 80 to 150 ms between runs. A co-tenant burst during one pass
   can bias a node by about 30 %; a node within that band is not distinguishable from the
   identity. The search rewards larger wins, and the reserved phase re-measures selected nodes.
5. Peak CUDA is `torch.cuda.max_memory_allocated`, which sees the caching allocator only.
6. The episode-overhead guard compares against the reference's overhead on different
   trajectories. A candidate whose actions make the simulator slower is charged for it.
7. A candidate that makes the sampler deterministic is not failed; its seed sensitivity is
   recorded, not judged.

## Reproduction

```sh
uv run --no-sync vla-rsi start --study-dir studies/program-v4 -- --cycles 3 --probe-budget 8 \
  --max-parallelism 2 --promote-top-k 1 --max-hours 8 --max-agent-usd 20 --policy-timeout 1200
```

Re-running the same command resumes from `loop_state.json`. `vla-rsi status`, `vla-rsi stop` and
`vla-rsi watch --port 8788` take the same `--study-dir`. `--max-parallelism 1` is the fallback
when the GPU is shared: one discovery agent and one evaluation at a time.

`python tools/check.py` inside a node workspace runs the check phase: one reference episode, 32
frames, 16 timed calls and one candidate episode, about 3 minutes on a free GPU. It prints the
speedup and any guard violation; the evaluator's smoke run decides the node.

The reserved phase runs outside the search loop with a longer evaluator timeout than the
protocol's 1800 s, as in program-v2; the search never reads reserved states.

## Status

No node has been evaluated and no reserved trial exists. The smoke gate is 8/10 and the
development gate 77/100, both unchanged from program-v2.
