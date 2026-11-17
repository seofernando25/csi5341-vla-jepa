# iter_001-b2-a1 — NF4 attention after the high-influence blocks, NF4 embedding

## Change relative to the parent (`iter_001-b2-a0`)

The parent deleted the training-only JEPA world model, dropped the unused tied `lm_head`,
and quantized the inference stack: int8 token embedding, int8 language-decoder attention,
NF4 language-decoder MLP, NF4 visual tower and NF4 action head (71.1% removed). This node
keeps the world-model deletion, the `lm_head` deletion, and the NF4 MLP/visual/action
recipe, and changes two things:

1. **The token embedding is NF4, not int8.** `embed_tokens` becomes
   `bitsandbytes.nn.Embedding4bit` (NF4, one scale per 64-value block, bf16 output).
2. **The language-decoder attention is NF4 for blocks 3–27, int8 for blocks 0–2.**
   The first three blocks keep the parent's `Linear8bitLt` (`has_fp16_weights=False`,
   threshold 6.0); every later `q/k/v/o_proj` becomes `Linear4bit` (NF4,
   double-quantized, bf16 compute). The MLP stays NF4 everywhere.

`compress` mutates the evaluator's policy object in place and returns it, then calls
`gc.collect()` and `torch.cuda.empty_cache()`. No copy of the original weights is kept.

## Why

The parent's remaining resident bytes were dominated by the int8 attention projections
(~0.47 GB) and the int8 token table (~0.31 GB); NF4 halves both. The sibling
`iter_001-b1-a0` found that plain NF4 on the whole language decoder cost an extra episode,
and this node's first attempt confirmed it: all-NF4 attention plus AWQ-style column
scaling on every Linear scored 76.9% on the offline check but only **7/10** on the
development episodes (task 5 additionally failed).

The calibration's `block_influence` explains where the sensitivity is: blocks 0–2 carry
the largest residual influence (1.25, 0.155 and 1575), while blocks 3–27 are all ≤0.081.
Keeping the first three attention projections int8 and converting only the later blocks
preserves the parent's exact success set (8/10, failures on tasks 0 and 8) while removing
most of the attention bytes. NF4 embedding alone was also verified: 8/10, same failures,
73.7% removed. The combination is 76.5% removed and still 8/10.

## Last check output

```
status: ok
memory removed: 76.5% (resident 1318520634 of baseline 5604584960 bytes)
action error (max normalized difference): 0.0859375
latency ms: median 139.23, p95 148.76
compress: 10.6 s
nonfinite predictions: 0
```

## Episode verification (evaluator-style, development trial 0, tasks 0–9)

- Parent recipe (`iter_001-b2-a0`): **8/10**, failures tasks 0 and 8.
- Parent + NF4 embedding only: **8/10**, failures tasks 0 and 8; 73.7% removed.
- Parent + NF4 embedding + all-NF4 attention + AWQ column scaling: **7/10**, failures
  tasks 0, 5 and 8; 76.9% removed. Rejected.
- This node (NF4 embedding, attention NF4 for blocks 3–27 and int8 for 0–2): **8/10**,
  failures tasks 0 and 8; 76.5% removed. The success set is identical to the untouched
  policy and to the parent.
- This node plus dropping the four trailing decoder blocks: **7/10**, failures tasks 0,
  4 and 8. Rejected; the final `solution.py` keeps all 28 blocks.

## What I would try next

- Dropping the four weakest trailing decoder blocks (calibration `block_influence` for
  layers 24–27 is 0.006, 0.007, 0.012 and 0.081 against 1575 for layer 2) was tried and
  rejected: **7/10**, with task 4 additionally failing. The trailing blocks are not as
  inert as their residual influence suggests; do not prune them without a new check.
- Do not use plain NF4 on the first three attention blocks: the all-NF4 decoder loses
  task 5 even with activation-aware scaling.
- The language MLP is the largest remaining term (~0.53 GB); a 3-bit packed path would
  need an episode run and a custom kernel before it is worth the risk. A safer next step
  is to convert one of the remaining int8 attention blocks (block 1 or block 0) to NF4
  and verify the 8/10 set; block 2's influence is far too large to touch.
