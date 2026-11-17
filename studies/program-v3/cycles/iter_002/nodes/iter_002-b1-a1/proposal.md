# Proposal: trim two redundant action-head blocks and the unused position rows

## Parent
`iter_002-b1-a0` (NF4/AWQ in-place compression): resident 1,106,270,720 bytes,
80.26% removed, check action error 0.096, development smoke 8/10 (tasks 0 and 8 fail,
exactly the untouched baseline). The gate has no margin, so any new episode failure is fatal.

## Change
All work is in place on the loaded policy; no copy of the original weights is kept.

1. **Drop action-head DiT blocks 7 and 9.** The calibration `block_influence` ranks them
   among the least influential action blocks (`action.7 = 0.0020`, `action.9 = 0.0016`;
   the first block is 2.03 and the median block is ~0.003). Each NF4 block costs about
   4.2 MB, so removing two saves ~8.4 MB and shortens the flow-matching loop.
2. **Compact the action position table.** `action_model.position_embedding` is a 1024-row
   table, but `DiT._build_inputs` only indexes positions `0..chunk_size-1` (7 rows). I
   replace it with an 8-row `nn.Embedding` holding the first eight rows, saving ~1.56 MB.
3. Everything else is the parent: the training-only world model is removed, the tied
   151936-row Qwen embedding is compacted to the fixed LIBERO prompt rows, text block 3 is
   dropped, and the text/vision/action Linears are NF4 with AWQ-style per-input-channel
   scaling from the calibration `linear_inputs`.

## Why
The parent already sits at the 8/10 gate with tasks 0 and 8 failing, so the only safe
direction is to remove parameters that inference barely uses. The action-head influence
scores identify blocks 7 and 9 as almost inert, and the position embedding is provably
over-sized for the fixed chunk length. Both changes are structural removals, not deferred
or hidden memory: the old modules are replaced and freed before `compress` returns.

I also measured more aggressive variants and rejected them because they raise the action
error too far for the no-margin gate:

- 3-bit packed MLP weights (custom 3-bit codebook, per-group absmax): text MLP only gave
  action error 0.173; adding vision MLP gave 0.282. Even 3-bit only on `up_proj` gave
  0.161. The weight MSE was close to NF4, but the error accumulates over 27 decoder layers.
- Dropping text block 5 in addition to block 3: 80.8% removed but action error 0.209.
- Dropping eight low-influence action blocks: 80.9% removed but action error 0.327.

## Last check
`python tools/check.py` (final run):

```
status: ok
memory removed: 80.4% (resident 1097536512 of baseline 5604584960 bytes)
action error (max normalized difference): 0.109375
latency ms: median 135.36, p95 138.76
compress: 8.4 s
nonfinite predictions: 0
```

Development smoke with the repository's own LIBERO rollout code, one episode per task:

```
task=0 success=0  task=1 success=1  task=2 success=1  task=3 success=1  task=4 success=1
task=5 success=1  task=6 success=1  task=7 success=1  task=8 success=0  task=9 success=1
successes=8/10
```

Tasks 0 and 8 are the same two failures as the untouched policy and the parent, so the
8/10 gate is preserved while resident memory falls from 1,106,270,720 to 1,097,536,512
bytes (80.26% -> 80.4% removed).

## Next
The remaining ~1.10 GB is still dominated by the NF4 text decoder (~681 MB) and vision
tower (~210 MB). A further safe reduction needs a better low-bit scheme than the 3-bit
codebook tried here: the next attempt should use calibration activations and GPTQ-style
error compensation (rather than weight-only MSE) so the 3-bit MLP error does not compound
across layers, or prune a small fraction (5%) of text MLP intermediate channels with a
fresh development smoke. Dropping more whole text or action blocks is not worth it: the
measured action error already rises sharply, and the gate has no margin.
