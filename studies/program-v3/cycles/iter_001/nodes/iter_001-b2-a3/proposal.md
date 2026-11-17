# iter_001-b2-a3 — activation-aware pruning of the language MLP

## Change relative to the parent (`iter_001-b2-a2`)

The parent deleted the training-only JEPA world model, dropped the unused tied `lm_head`,
compacted the token embedding to the rows the fixed LIBERO-Spatial prompts can address,
truncated the action-head position table, and quantized the inference stack: int8
language-decoder attention in blocks 0–2, NF4 attention in blocks 3–27, NF4 language MLP,
NF4 visual tower and NF4 action head (79.5% removed, 8/10 episodes, failures on tasks 0
and 8). This node keeps that recipe and adds one change:

**The language-decoder MLP is physically pruned before quantization.** For each layer
3–26, the intermediate neurons are ranked by their estimated contribution to the residual
update and the lowest 10% are removed. The layer's `gate_proj`/`up_proj` output rows and
`down_proj` input columns are sliced to the kept set, so the smaller matrices are what
bitsandbytes packs to NF4; the original bf16 modules are released as each layer is
replaced. Layers 0–2 and 27 are left intact because their calibration `block_influence`
(1.25, 0.155, 1575 and 0.081) is orders of magnitude above the middle stack (≤0.012).

The ranking uses the calibration's `linear_inputs` only:
for neuron `i` of layer `l`, the residual update is
`down[:, i] * silu(gate[i] . x) * (up[i] . x)`, so with
`E[x_j^2]` for the gate/up input and `E[h_i^2]` for the down input
`h = silu(gate(x)) * up(x)`, the expected squared contribution is proportional to

```
||gate[i] * sqrt(E[x^2])||^2 * ||up[i] * sqrt(E[x^2])||^2 * ||down[:, i] * sqrt(E[h^2])||^2
```

Keeping the top 90% by that score removes neurons that barely move the residual stream on
the calibration distribution. The score is computed in float32 one layer at a time; the
temporary tensors are freed before the next layer. No copy of the original weights is kept.

`compress` mutates the evaluator's policy object in place and returns it, then calls
`gc.collect()` and `torch.cuda.empty_cache()`. The token-id set is computed with the
already-loaded tokenizer only; no file is opened and no image processor is invoked inside
`compress`.

## Why

After the parent, the language MLP was the largest remaining term (about 0.53 GB), and
the calibration exposes exactly the per-channel input energy needed to rank its neurons.
A 10% prune removes about 53 MB and keeps the verified 8/10 success set. Larger ratios
were tried and rejected on the actual development episodes:

- language MLP keep 0.85 (15% prune, layers 3–26): **7/10**, task 4 additionally failed.
- language MLP keep 0.75 (25% prune, all layers): **7/10**, task 4 additionally failed.
- language MLP keep 0.75 (25% prune, layers 3–26): **7/10**, task 4 additionally failed.
- language MLP keep 0.50 for layers 20–26, 0.90 elsewhere: **7/10**, task 4 failed.

The 10% node's action error is 0.09375, essentially the parent's 0.0898, and its episode
success set is identical to the untouched policy and the parent. The 15% variant's action
error was 0.110 and it already lost task 4, so 10% is the largest verified-safe ratio.

## Last check output

```
status: ok
memory removed: 80.0% (resident 1119652352 of baseline 5604584960 bytes)
action error (max normalized difference): 0.09375
latency ms: median 140.96, p95 144.00
compress: 6.1 s
nonfinite predictions: 0
```

The score implied by the check is `1 - 1119652352 / 5604584960 = 0.800226`, versus the
parent's 0.795223.

## Episode verification (evaluator-style, development trial 0, tasks 0–9)

- Parent recipe (`iter_001-b2-a2`): **8/10**, failures tasks 0 and 8; 79.5% removed.
- This node (10% language-MLP prune, layers 3–26): **8/10**, failures tasks 0 and 8;
  80.0% removed. The final `solution.py` was re-run on the same ten development episodes
  after the constants were hard-coded; result: **8/10**, same failures.
- 15% prune (keep 0.85): **7/10**, task 4 additionally failed. Rejected.
- 25% prune (keep 0.75, all layers or only layers 3–26): **7/10**, task 4 failed. Rejected.
- 50% prune of layers 20–26 plus 10% elsewhere: **7/10**, task 4 failed. Rejected.

## What I would try next

- Compute the neuron importance per task from the observation bank and use a minimum (or
  low quantile) across tasks instead of the calibration's pooled mean. The pooled mean can
  dilute a neuron that matters for one task; a per-task minimum might allow a 20–25% prune
  while keeping task 4. This needs an episode run before it is kept.
- The visual tower's `pos_embed` is an `nn.Embedding` with 2304 rows but the fixed 256×256
  input uses a 16×16 patch grid. Precomputing the interpolated 256-row table and setting
  `num_grid_per_side=16` is lossless for the protocol and saves about 4.2 MB.
- The compact embedding's full-vocabulary int32 mapping is about 0.6 MB; a sorted 87-entry
  id table plus `torch.searchsorted` would remove it losslessly.
- Converting one of the remaining int8 attention blocks (block 0 or 1, not block 2) to NF4
  saves about 6 MB each and is worth an episode check; the parent found all-NF4 attention
  loses task 5, so do not convert block 2 or all three.
- Do not raise the MLP prune above 10% without an episode run: 15% already loses task 4.
  Do not prune the trailing decoder blocks or their MLPs aggressively; every attempt above
  lost task 4 even though `block_influence` for those layers is small.
