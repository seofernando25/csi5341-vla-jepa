# Node iter_000-b0-a3

## Change relative to parent (iter_000-b0-a2)

The parent scored 0.855983 with text/vision/action all at groupwise int3, after dropping
the training-only world model, compacting the token embedding and dropping the unused LM
head.  This node keeps all of those changes and adds two new ones:

1. **Structured pruning of text-MLP intermediate channels.**  Every Qwen3 text MLP has a
   6144-wide intermediate activation.  The frozen calibration stores the mean squared
   input of `down_proj`, i.e. the activation energy of each intermediate channel.  For
   each layer I score each channel by
   `E[a_j^2] * ||down_proj[:, j]||^2` (its expected squared contribution to the layer
   output), keep the best 5376 channels and physically slice the `gate_proj`/`up_proj`
   output rows and the `down_proj` input columns.  That removes 12.5% of the text-MLP
   weights (about 132 M parameters) and their quantizer side data.  The calibration
   vector for `down_proj` is sliced to the same channels so the AWQ scale search stays
   consistent.
2. **Compact int3 quantizer metadata.**  The parent stores one bf16 scale and one uint8
   zero per 128-weight group (3 bytes/group, ~46 MB in total).  This node stores the
   per-row scales as an 8-bit affine code with one fp16 min and step per row, and packs
   the 3-bit zero points eight per three bytes.  Metadata drops to 1.375 bytes/group
   (~21 MB), a lossless-to-within-0.5% change in the dequantized weights.

The deployed bit widths are unchanged: text int3, vision int3, action int3.  The
structural removal of the world model, the compact 53-row embedding and the unused LM
head are inherited from the parent.

All replacements happen in place, so the original storages are freed; no compressed copy
is kept.  `compress` performs no file I/O and writes no files.

## Last check output

`python tools/check.py` on this workspace:

```
status: ok
memory removed: 86.9% (resident 724667303 of baseline 5532276502 bytes)
action error (max normalized difference): 0.240234
latency ms: median 873.80, p95 1019.57
```

Size reduction: `1 - 724667303 / 5532276502 = 0.869011`.

## Smoke rollout (the evaluator's 10-episode phase)

```
status: ok, successes 8/10
resident 724,667,303 bytes, size_reduction 0.8690110115
per task: 0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:0, 9:1
```

The two failures are tasks 0 and 8, exactly the two the untouched policy and every
previous node also fail; every task the parent passed still passes.  This is the
deployed recipe.

## Experiments tried and rejected

- **25% text-MLP pruning (keep 4608):** 87.8% removed, action error 0.260, but the smoke
  rollout failed task 5 (7/10: tasks 0, 5 and 8 failed).  Reverted to 12.5%.
- The 12.5% prune is the largest tested width that kept the parent's exact 8/10 per-task
  pattern.  The parent's int2 experiments (text down_proj int2, vision int2, all-text
  int2) were not repeated; they were already rejected for action error and smoke
  failures.

## What I would try next

- Try an intermediate text-MLP width (18.75%, keep 4992) or 25% only in the late layers
  whose calibration energy removal is smallest; the 25% all-layer variant was just over
  the edge at task 5.
- Add 12.5% vision-MLP pruning (about 10 MB, +0.18 points) if a smoke rollout confirms
  it does not flip a task.
- The remaining large gain is still 2-bit text/vision with a per-group codebook or
  GPTQ-style error compensation; the affine int2 paths in the parent were too lossy.
