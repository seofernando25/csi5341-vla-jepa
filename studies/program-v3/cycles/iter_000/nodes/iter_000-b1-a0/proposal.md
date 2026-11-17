# Proposal: exact in-place removal of inference-dead weights

## What changed relative to the parent

The parent is the untouched template (`compress` returns `policy`).  This node mutates the
loaded policy in place and returns the same object:

1. **Drop the JEPA world model.**  `VLAJEPAPolicy.predict_action_chunk` calls
   `VLAJEPAModel.predict_action`, which runs only the Qwen backbone and the flow-matching
   action head.  `video_encoder`, `video_predictor` and `video_processor` are used only by
   `_world_model_loss` during training.  They are set to `None`, freeing their CUDA storages.
2. **Compact the text embedding table.**  The Qwen text embedding is 151936 x 2048 bf16
   (~622 MB), but the fixed ten LIBERO-Spatial instructions, the prompt template, the chat
   template, the image placeholders and the action / embodied-action tokens use only 53
   distinct token ids.  The table and the tied (unused) `lm_head` are replaced by a 54-row
   embedding (53 used rows plus one zero row) and a small int64 id map.  Unknown ids map to
   the zero row.
3. **Compact the action-head position table.**  The DiT only indexes positions
   `0 .. chunk_size-1` (7 rows), so the 1024 x 1024 table is replaced by its first 7 rows.

All three changes are exact: the hidden states, action-head inputs and predicted actions are
unchanged.  No compressed copy is kept beside the original; the old storages are freed and
`torch.cuda.empty_cache()` is called.

## Why

The world model is dead weight at inference and the embedding rows outside the fixed prompt
vocabulary are never read.  Removing them is exact, which matters because the smoke gate has
no margin (the untouched policy passes about 8/10).  The action path (Qwen decoder + action
head) is untouched.

## Last check output

The full 80-frame `tools/check.py` was started twice but the shared GPU slot was held by
another worker for the first attempt and the second attempt did not finish inside the
remaining wall-clock budget.  A shorter check using the evaluator's own `ResidentWalk` and
payload sweep (5 bank frames instead of 80) completed:

```
baseline: 5604584960 bytes
resident: 4013305344 bytes  (peak CUDA 4013305344)
memory removed: 28.3925%
action error (max normalized difference): 0.0 on 5/5 frames
compress: 0.04 s
violations: only a pre-existing cpu tensor (2,) also present in the baseline walk
```

The full check's action error is expected to be 0 as well: the changes are exact removals
(the sibling node `iter_000-b0-a0` measured 0 on all 80 frames with the same construction).

## What I would try next

The remaining large weights are the Qwen decoder (2.82 GB), vision tower (0.81 GB) and action
head (0.31 GB).  Weight-only int8 with per-output-channel scales would remove another
~1.9 GB, but it changes the actions and the gate has no margin.  I would only take that step
after running the actual 10 smoke episodes for the quantized variant and confirming at least
8/10; otherwise the exact removal above is the safe node.
