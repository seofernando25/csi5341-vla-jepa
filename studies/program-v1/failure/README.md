# Failure analysis: where language-decoder quantization changes the policy

Answers the proposal's promise of "diagnostic failure analyses on identical observation sequences"
for the four language-decoder cells. The proposal named three mechanisms: multimodal representations,
spatial attention and downstream action trajectories. Two are measured here, one under a caveat.

## Question

On the same 80 observation-bank frames the screen measures, how far does each language-decoder cell
move the decoder's representations, the action head's conditioning input, the image-token attention
and the predicted action chunk away from B16 — and which tasks lose success in the rollout records?

## Result

80 frames, one seed per frame, four cells. CKA is linear CKA against B16 at the decoder's last layer;
the action error is the mean physical arm MSE against B16 on the same frames.

| Cell | Bits | CKA, all tokens | CKA, image tokens | CKA, conditioning | Arm MSE | Gripper flips |
|---|---:|---:|---:|---:|---:|---:|
| INT8 | 8 | 1.0000 | 0.9692 | 0.9968 | 1.67e-5 | 0 / 80 |
| NF4 | 4 | 0.9999 | 0.9099 | 0.9899 | 2.00e-4 | 0 / 80 |
| HQQ 3-bit | 3 | 0.9997 | 0.7124 | 0.9762 | 7.84e-4 | 0 / 80 |
| HQQ 2-bit | 2 | 0.9977 | 0.1761 | 0.7071 | 3.57e-2 | 3 / 80 |

- **The image-token representation is where the drift is.** CKA over image-token rows falls 0.97,
  0.91, 0.71, 0.18 as the decoder drops to 8, 4, 3, 2 bits, and the action head's conditioning input
  follows it down (0.997, 0.990, 0.976, 0.707). The arm action error rises four decades over the same
  cells (1.7e-5 to 3.6e-2).
- **The all-token CKA cannot see it.** It stays at or above 0.9977 because a few text-token channels
  carry activations of 3.3e4 to 3.8e4 while image-token rows stay below 1.9e3; those channels
  dominate the Gram matrix. The all-token column is a warning about the metric, not evidence that
  the decoder is unchanged.
- **Attention keeps its budget and loses its pattern.** The embodied-action queries put 4.7 % of
  their attention mass on image tokens in B16 and 4.7 %, 4.6 %, 4.8 %, 5.6 % in INT8, NF4, HQQ 3-bit and
  HQQ 2-bit:
  the amount barely moves. The pattern does: the median per-layer Spearman correlation of the mass
  vector against B16 falls 0.997, 0.990, 0.958, 0.781, and the worst HQQ 2-bit layer reaches 0.005.
- **The gripper is the last thing to break.** No cell flips a gripper command until HQQ 2-bit, which
  flips 3 of 560 (frame, step) commands, all in 3 of the 80 frames.
- **Task level.** Development, 100 trials, 10 per task: HQQ 2-bit loses all ten trials on tasks 1, 4
  and 9, eight on 6 and 7, six on 3, and keeps only task 2 (8/10 against 10/10). INT8, NF4 and HQQ
  3-bit stay within 2 trials of B16 on every task. Reserved, 500 trials, 50 per task: INT8 loses 2 on
  task 5 (31/50 against 33/50) and gains 1 on task 3; NF4 gains 5 on task 5 (38/50) and loses 1 each
  on tasks 2 and 4. Task 5 is the task B16 itself fails most (33/50).
- **Reproduction.** Per-frame arm MSE matches the committed screen record exactly on all 80 frames of
  all four cells (largest absolute difference 0.0), so these are the screen's observations, not a
  re-measurement of something else.

## Method

- Frames: the 80 observation-bank indices outside the 20 calibration states, in bank order, seeded
  `protocol.seed + bank_index` after `policy.reset()` — the screen's own frame set and seeding.
- CKA: `evaluation/compression/cka.py`'s `linear_cka` on unit-Frobenius features (CKA is invariant to
  that rescaling, and the raw float32 Gram squares overflow on the activations above). 16 evenly
  spaced token rows per frame per layer for each of three row sets: all tokens, image tokens, text
  tokens. 28 decoder layers, plus the DiT's conditioning input (32 embodied-action tokens per frame).
- Attention: the deployed sdpa kernel returns no weights, so the encoder pass is repeated with
  `_attn_implementation="eager"`; a guard rejects a non-causal mask. Mass of the 32 embodied-action
  query tokens on the 128 image-token columns, mean over 16 heads, per layer, then Spearman rank
  correlation with B16 over the 80 × 32 masses.
- Actions: physical arm MSE over the six arm dimensions and gripper disagreement over the seventh,
  per frame, against B16's chunk on the same frame.
- Tasks: `measurements/<cell>/development/episodes.csv` for the development phase and
  `reserved/summary.json` `per_task` for the reserved phase. Phases are reported apart and never
  pooled.

## Evidence

- [failure.json](failure.json): every per-frame arm MSE, per-layer CKA, attention mass, per-task count
  and the environment record. Written by `vla-compress-failure`.
- [../../report/figures/failure.pdf](../../report/figures/failure.pdf): CKA and arm action error
  against weight bits, from `vla-report-plots`.
- Screen records compared against: `measurements/<cell>/screen/observations.json`.

## What it does not show

- **Not LIBERO success.** CKA and offline action error are not task outcomes. The task counts come
  from development and reserved rollouts on other states, not from these 80 frames.
- **Not the deployed attention kernel.** Eager attention is numerically different from sdpa. B16 and
  each cell are measured the same way, so the comparison holds, but the absolute mass is not the
  deployed computation.
- **Not a cause.** The conditioning input drifts with the image representation and the actions drift
  with both. Nothing here separates the decoder's contribution from the action head's.
- **Not the whole decoder.** The all-token column is saturated by massive activations; a drift that
  leaves those channels intact would be invisible in it.
- One frame set (LIBERO-Spatial, 80 held-out development frames), one seed per frame, and a 16-row
  sample per frame per layer. Splitting the frames in half moves the image-token CKA by at most 0.04
  (HQQ 3-bit 0.694 against 0.731), so the sample is stable at this resolution but cannot resolve
  differences below about 0.01.

## Reproduction

```sh
uv run --no-sync vla-compress-failure     # GPU, ~7 min, writes failure.json
uv run --no-sync vla-report-plots         # CPU, writes report/figures/failure.pdf
```
