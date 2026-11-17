# iter_000-b0-a3, reserved 500

`final/` is the single reserved run that counts: 475/500 on the corrected environment.

`attempt-1-cache-error/` is the first launch. It stopped in the evaluator's baseline stage with
`size_violation` (`opened outside workspace (mapped): ~/.triton/cache/...cuda_utils...so`) before
any episode ran: `torch.compile` put its Triton cache under the home directory, outside the
workspace and temp roots. No reserved trial was consumed, so the cell was rerun once with
`TRITON_CACHE_DIR` and `TMPDIR` pointing at the same scratch directory (inside the allowed temp
root; the sandbox already gets this through `HOME=/tmp`). Code, workspace and protocol were
unchanged.
