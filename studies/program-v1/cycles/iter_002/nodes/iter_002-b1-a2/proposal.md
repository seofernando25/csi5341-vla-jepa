# iter_002-b1-a2: bit-packed 4-bit decoder (denser storage of the parent's quantized weights)

Parent: `iter_002-b1-a1` (score 0.813786, 8/10 episodes, failures on tasks 0 and 8).

## Change relative to the parent

The parent's dead-state removal, prompt-row embedding compaction, NF4 vision/action
storage and 4-bit per-group decoder quantization are kept exactly. The single change is
how the decoder's quantized weights are stored:

- The parent stored each 4-bit value in one int8 byte and zlib-compressed that stream
  (measured ratio 0.503, i.e. ~4.0 bits/value).
- This node packs two 4-bit values into each byte first, then zlib-compresses the packed
  stream (measured ratio 0.433 on the same weights). The BF16 per-group scales are
  compressed separately. On first forward the packed stream is decompressed, unpacked to
  int8 and restored on CUDA, exactly as before.

The quantized values themselves are unchanged, so the restored weights and all actions are
bit-identical to the parent. This is a pure storage-format change with no accuracy risk.

## Why

The decoder blobs dominate the resident charge (729.9 MB of the parent's 1030.2 MB).
Packing removes the constant high nibble that zlib was already encoding inefficiently;
the measured compression ratio improves from 0.503 to 0.433, saving ~86 MB with zero
change to the model.

## Last check output

`cd /tmp/work && /tmp/venv/bin/python tools/check.py`:

```
status: ok
memory removed: 82.9% (resident 943888474 of baseline 5532276502 bytes)
action error (max normalized difference): 0.126953
latency ms: median 146.90, p95 187.78
```

The action error is exactly the parent's 0.126953: the 4-bit quantization is unchanged.

Smoke rollouts run during this node (same 10 development episodes as the gate):

| decoder storage | memory removed | smoke |
|---|---|---|
| 3-bit uniform, all decoder linears | 85.3% | 6/10 (tasks 0,4,6,8 fail) |
| 3-bit uniform MLP, 4-bit attention | 84.7% | 7/10 (tasks 0,4,8 fail) |
| 3-bit Lloyd codebook MLP | 84.3% | rejected: action error 1.82 |
| **4-bit packed, all decoder linears** | **82.9%** | expected 8/10 (bit-identical to parent) |

The 3-bit variants were rejected because they lose at least one episode the baseline
passes; the gate has no margin (baseline is 8/10 with tasks 0 and 8 failing).

## What I would try next

- The remaining 3-bit accuracy loss is concentrated in task 4. A mixed scheme that keeps
  the layers most responsible for that failure at 4 bits (e.g. selected by calibration
  `block_influence` or by a per-layer sensitivity sweep) could recover the gate at ~84%.
- Compressing the NF4 vision/action packed tensors and their absmax metadata with the
  same lazy-blob mechanism (small additional saving, ~5-10 MB).
- The 4-bit ceiling is close: 4-bit packed + zlib is ~0.433 bytes/parameter; a custom
  arithmetic coder could reach the ~3.4-bit entropy bound for another ~10-15 MB.
