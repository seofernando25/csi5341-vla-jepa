# Program v4: Dream-RSI search for a faster policy

Prepared 2026-11-16, revised and launched the same day. The frozen
rules are in [protocol.json](protocol.json), the discovery prompt in [discovery.md](discovery.md).
Checkpoint, suite, task ids, seeds, phases, gates, statistics, agent and budgets are program-v2's;
the score is new.

## Question

Can the Dream-RSI search make the VLA-JEPA-LIBERO policy faster on LIBERO-Spatial while it stays
as successful as the reference?

## Score

    speedup = geometric mean over every task of the suite of (reference median
              observation-to-action latency / candidate median on that task)
    score   = the lower bound of the 95 % interval of the speedup, if the smoke gate passes,
              the candidate's peak CUDA bytes are not above the reference's, and no guard fails
    score   = 0 otherwise

The reference is the untouched B16 policy, four flow steps, stock execution path, measured in the
same process on the same frames. The identity scores below 1.0 by construction: its point estimate
is 1.0 and the score is the lower bound of the interval around it. Peak CUDA is compared with a
1 % measurement allowance, because the reference and candidate peaks are separate runs.

`evaluation/compression/latency.py` measures it, with the instruments in
`latency_timing.py`, the frame plan and the stratified bootstrap in `latency_plan.py`, and the
failing conditions in `latency_guards.py`. The evaluator runs reference episodes, keeps the
observation batches, and times the untouched policy on them. After `compress` it times the
candidate on the same batches.

Every count is per task, and all ten tasks get the same count in every role. The evaluator
collects frames from one reference episode per task (stride 2, at most 36 frames each) and uses
each task's frames for 3 equivalence frames, 8 arm-check frames, 6 warm-up calls and 18 timed calls
(2 and 12 in the check phase, which also covers all ten tasks). The speedup is the median
reference time over the median candidate time on each task's calls, aggregated as the geometric
mean, so a gain on one task moves the score by a tenth of its log and no task carries the median.
The equivalence, arm-action and perturbation guards are judged on each task and the worst task
decides, so degrading one task cannot be averaged away.

Timing is block-based. Each side is timed in six blocks; each block holds the same number of calls
from every task (three each, two in the check phase), in alternation. Every timed call is a full
`select_action` with `torch.cuda.synchronize()` before and after, after warm-up, on frames that
were never timed before. When a second copy of the reference fits beside the candidate, the blocks
alternate reference, candidate, reference, candidate, and the order inside a pair alternates too,
so neither a co-tenant burst nor a clock ramp at the start of a block favours one side. When it
does not fit, the reference is timed in six blocks before `compress` and six after the candidate
episodes and the two pools are combined (sandwich); the node fails if the two pools differ by more
than 10 %, because a moving reference cannot be pooled.

The speedup is reported with a 95 % percentile cluster bootstrap over blocks (10 000 resamples,
seed 20261116). In the interleaved design a resample draws block indices and takes both sides of
each drawn block; in the sandwich the two sides are resampled independently. The score is the
2.5th percentile. Fewer flow steps, a smaller model or any other change is allowed if the episodes
still pass.

## Guards

Every guard is a failing condition (`guard_violation`, score 0), not a warning.

| Guard | Catches | Test |
|---|---|---|
| Captured clock | patching `time` or `torch.cuda` readers | `test_clock_rebinds_are_detected`, `test_a_patched_clock_does_not_change_the_measurement` |
| Disjoint warm-up and timed frames | a lookup table keyed on the observation | `test_timed_calls_use_frames_never_seen_in_warmup`, `test_a_cache_of_warmup_frames_gains_nothing_on_new_frames` |
| Novel-frame cost | a cache with a slow fallback | `test_a_cached_solution_is_caught` |
| Equivalence | a constant answer, a stale answer, a special-cased `select_action` | `test_a_constant_answer_fails_the_equivalence_check`, `test_a_stale_answer_fails_order_invariance`, `test_a_special_cased_select_action_fails_path_equivalence` |
| Arm-action error | a lookup table whose action is not the policy's | `test_a_nearest_neighbour_cache_fails_the_arm_check`, `test_a_quantized_policy_passes_the_arm_check`, `test_the_identity_passes_the_arm_check` |
| Perturbation response | a lookup table that returns a stored answer | `test_a_lookup_table_fails_the_perturbation_check`, `test_a_quantized_policy_passes_the_perturbation_check`, `test_the_identity_passes_the_perturbation_check` |
| Reset cost | work moved out of the timed call into `reset` | `test_work_moved_into_reset_is_charged` |
| Episode overhead | background or asynchronous work | `test_episode_overhead_above_the_reference_fails` |
| CPU and child CPU | work moved to the CPU or another process; new threads | `test_cpu_above_the_cap_fails`, `test_child_cpu_above_the_cap_fails` |
| Shared GPU | a co-tenant above 50 % of the SM utilisation during the timing blocks | `test_another_process_above_the_level_fails_the_node`, `test_a_quiet_gpu_passes_the_watch`, `test_the_gpu_watch_counts_only_other_processes` |
| Peak CUDA | more memory than the reference (1 % allowance) | `test_peak_memory_above_the_reference_fails` |
| Block plan and interval | timing noise creating a win | `test_the_frame_plan_has_disjoint_blocks_and_warmup`, `test_a_burst_in_one_block_of_both_sides_does_not_create_a_win`, `test_a_burst_in_one_candidate_block_lowers_the_lower_bound` |
| Sandwich | an incomplete or drifting reference pool | `test_the_sandwich_adds_the_after_blocks`, `test_the_sandwich_without_a_second_reference_fails_the_node`, `test_the_sandwich_fails_when_the_reference_drifts` |
| Score | the formula and its zero conditions | `test_score_is_the_lower_bound_when_the_gate_and_memory_pass`, `test_score_is_zero_below_the_gate_or_above_the_memory` |

`tests/test_latency.py` runs all of them on the CPU, without the simulator or a GPU.

## The arm-action threshold

`arm_action_mse_max = 0.01` is the mean over 32 fresh frames of the squared physical arm-action
difference against B16, after the post-processor, on the first six action dimensions. The evidence
is the program-v1 screen, which measured the same quantity over 80 held-out bank frames for 17
precision cells (`studies/program-v1/measurements/*/screen/run.json`,
`summary.mean_physical_arm_change_mse`). One cell is above 1e-2: `precision-lang-hqq2` at 3.57e-2,
which scored 20/100 in development. Every cell at or below 3.1e-3 scored 79-86/100. The threshold
sits in that gap, 3.2x above the worst passing cell and 3.6x below the failing one. The screen's
`maximum_normalized_action_difference` is not used: `precision-action-nf4` sits at 1.17e-2 there
while its arm error is 2.9e-6.

The perturbation check perturbs the visible bank frames by at most 8 grey levels per pixel, times
the same eight frames before and after `compress`, and requires a Pearson correlation of at least
0.5 between the candidate's arm changes and B16's. A lookup table returns the same action for the
original and the perturbed frame, so its changes are zero.

## Harness checks (not evidence)

Run on the shared RTX 3090 on 2026-11-16 with the check phase, six blocks of two calls per task.

- Identity under the revised score: score 0.987, point 0.998, 95 % interval [0.987, 1.003],
  per-task speedups 0.988 to 1.004. The identity is below 1.0 by construction.
- Identity under the earlier three-task score, 16 runs: 14 accepted, scores 0.944 to 0.995; two
  were zeroed by the 25 % GPU level before it was raised. Point estimates 0.990 to 1.053, mean
  1.017. The level is 50 % because the alternating blocks already absorb moderate co-tenancy: at
  25 % the guard lost nodes without improving the estimate.
- `precision-action-nf4` as a solution: arm error 2.3e-6, perturbation correlation 0.989, no
  violation, score 0.879 (point 0.973) under the three-task score.
- A nearest-neighbour cache of B16 actions on the visible bank, keyed on `observation.state`: it
  passes the equivalence check and measures a 412x point speedup, and fails with arm error 0.178
  and perturbation correlation 0.000.
- Forced sandwich, three runs: point estimates 1.203, 1.247 and 0.929; the reference moved 20 %,
  39 % and 3 % between the before and after pools. The drift guard fails the first two.
- The stopped first cycle is the specification-gaming check: its task-0 node scored 3.40 under the
  three-task score and is archived in [stopped/](stopped/README.md). Under the revised score it
  would keep only its share of the ten tasks.

## Open holes

1. A detached process that is reparented is not counted by `RUSAGE_CHILDREN` and not seen as a
   child pid. Only children started during the timed calls are checked.
2. GPU busy time is CUDA-event elapsed time on the default stream, not CUPTI counters. A kernel on
   a side stream still lands inside the device-wide synchronize, so it is charged to the wall time,
   but per-kernel attribution is not measured.
3. Peak CUDA is `torch.cuda.max_memory_allocated`, which sees the caching allocator only.
4. The episode-overhead guard compares against the reference's overhead on different trajectories.
   A candidate whose actions make the simulator slower is charged for it.
5. A candidate that makes the sampler deterministic is not failed; its seed sensitivity is
   recorded, not judged.
6. The reference arm actions and perturbation deltas are held in the evaluator's process while
   `compress` runs. A solution that walks the heap could read them; it would still have to
   interpolate them to pass the perturbation correlation and the observation-sensitivity check.
   Not observed.
7. The interval is a cluster bootstrap over six blocks. With six clusters the percentile interval
   is coarse, and the lower bound is conservative: the identity scores 0.987 under the revised
   score, so a real win must exceed roughly 2 % to score above 1.0. The check phase uses two calls
   per task per block and is noisier than the 18-call protocol.
8. The interleave needs a second B16 (5.66 GB) beside the candidate (at most 5.71 GB). On the
   shared 24 GB card with a 12 GB co-tenant it fits only while the other search is idle. The
   sandwich fallback needs the candidate freed and the reference stable to within 10 %; when
   either fails the node scores 0.
9. The GPU watch reads `nvidia-smi pmon`, which reports per-process SM utilisation at the block
   boundaries, not continuously. A burst shorter than a block that starts and ends between samples
   is seen only through its effect on the timings. When `pmon` is unavailable the guard is not
   applied and the record carries the error.
10. The perturbation probe uses eight visible bank frames and one amplitude. A cache whose key
    changes under the perturbation returns a different stored action; the correlation is then
    near zero only if the stored actions are unrelated to the perturbation response. A cache of
    neighbouring frames could in principle correlate; it would still have to pass the arm check.
11. Specialising to the timed tasks is bounded, not impossible: a candidate that speeds up one
    task and leaves nine stock gains one tenth of its log in the geometric mean, and the per-task
    guards limit how far its actions may move on that task. A candidate that handles all ten tasks
    is a real deployment win; a candidate that handles one is a one-task win and scores like one.

## Reproduction

```sh
uv run --no-sync vla-rsi start --study-dir studies/program-v4 -- --cycles 3 --probe-budget 8 \
  --max-parallelism 2 --promote-top-k 1 --max-hours 8 --max-agent-usd 20 --policy-timeout 1200
```

Re-running the same command resumes from `loop_state.json`. `vla-rsi status`, `vla-rsi stop` and
`vla-rsi watch --port 8788` take the same `--study-dir`. `--max-parallelism 1` is the fallback
when the GPU is shared: one discovery agent and one evaluation at a time.

`python tools/check.py` inside a node workspace runs the check phase: ten reference episodes, one
per task, 28 frames per task, 12 timed calls per task in six blocks, and one candidate episode,
about 3 minutes on a free GPU. It prints the score, the point estimate and its interval, the
per-task speedups, the arm error, the perturbation correlation, the co-tenant GPU use and any
guard violation; the evaluator's smoke run decides the node.

The reserved phase runs outside the search loop with a longer evaluator timeout than the
protocol's 1800 s, as in program-v2; the search never reads reserved states.

## Status

The first cycle was stopped and archived in [stopped/](stopped/README.md): its score rewarded the
timed tasks, not the suite. The run restarted under the revised score with every task timed. No
reserved trial exists. The smoke gate is 8/10 and the development gate 77/100, both unchanged from
program-v2.
