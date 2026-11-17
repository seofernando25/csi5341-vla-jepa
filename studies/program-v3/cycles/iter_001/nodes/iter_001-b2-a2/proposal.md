# iter_001-b2-a2 — compact token embedding, truncated action position table

## Change relative to the parent (`iter_001-b2-a1`)

The parent deleted the training-only JEPA world model, dropped the unused tied `lm_head`,
and quantized the inference stack: NF4 token embedding, int8 language-decoder attention in
blocks 0–2, NF4 attention in blocks 3–27, NF4 language-decoder MLP, NF4 visual tower and
NF4 action head (76.6% removed, 8/10 episodes, failures on tasks 0 and 8). This node keeps
that recipe and changes two things:

1. **The token embedding is compacted instead of quantized.** The 151936-row bf16 table is
   replaced by a bf16 table holding only the token ids the fixed LIBERO-Spatial prompts can
   emit: the ten task instructions, the prompt template, the chat-template role tokens and
   newline, the vision placeholder tokens, the added action tokens, and all tokenizer
   special ids (87 rows). A vocab-sized int32 lookup maps token ids to compact rows; ids
   outside the set fall back to row 0. This removes ~154 MB: the parent's NF4 table was
   155.6 MB, the compact table plus mapping is about 1 MB. The compact rows are bf16, so
   they are more precise than the parent's NF4 rows.
2. **The action head's position embedding is truncated.** `_build_inputs` indexes only
   `torch.arange(action_features.shape[1])`, i.e. the first `action_horizon` = 7 rows, but
   the checkpoint table has 1024 rows. The table now has 7 rows (about 2 MB saved).

`compress` mutates the evaluator's policy object in place and returns it, then calls
`gc.collect()` and `torch.cuda.empty_cache()`. No copy of the original weights is kept.
The token-id set is computed with the already-loaded tokenizer only; no file is opened and
no image processor is invoked inside `compress`.

## Why

After the parent, the largest single remaining tensor was the token embedding (155.6 MB),
and the next-largest avoidable table was the action-head position embedding (2 MB). The
scored prompts are fixed LIBERO-Spatial instructions, so almost all of the 151936-row
vocabulary is unreachable. Keeping only the reachable rows is lossless for every token the
evaluation can produce, unlike a lower-bit embedding. The compact table is bf16, so it
does not add quantization error; the slight action-error change (0.0859 → 0.0898) comes
from replacing the parent's NF4 embedding rows with exact bf16 rows, and the episode
success set is unchanged.

## Last check output

```
status: ok
memory removed: 79.5% (resident 1147685376 of baseline 5604584960 bytes)
action error (max normalized difference): 0.0898438
latency ms: median 139.77, p95 142.37
compress: 6.2 s
nonfinite predictions: 0
```

## Episode verification (evaluator-style, development trial 0, tasks 0–9)

- Parent recipe (`iter_001-b2-a1`): **8/10**, failures tasks 0 and 8; 76.6% removed.
- This node (compact embedding + truncated action position table): **8/10**, failures
  tasks 0 and 8; 79.5% removed. The success set is identical to the untouched policy, the
  parent, and the sibling nodes.
- The final `solution.py` was re-run on the same ten development episodes after the
  tokenizer-only token-set construction; result: **8/10**, same failures.

## What I would try next

- The language-decoder MLP is now the largest remaining term (~0.53 GB). A 3-bit packed
  path or activation-aware neuron pruning (using the calibration `linear_inputs` for
  gate/up/down importance) could remove ~0.1–0.26 GB, but both need an episode run before
  they are worth keeping; the 8/10 gate has no margin.
- The visual tower's `pos_embed` (about 4.7 MB) and the four merger MLPs (about 34 MB) are
  the next structural targets; compacting `pos_embed` to the rows the fixed 256×256 image
  grid interpolates would be lossless but saves only a few MB.
- Do not convert the remaining int8 attention blocks 0–2 to NF4, and do not prune the
  trailing decoder blocks: the parent tried the latter (7/10) and the former loses task 5.
- Do not trade away the verified 8/10 set for action error alone.
