# Studies

## Clean-room reset (2026-11-15)

The pre-cleanroom compression and Dream-RSI evidence was deleted from the working tree. It remains
in git history at tag `pre-cleanroom-20261115`. Nothing from that tree is reused, pooled, or cited.

The new evidence base is [program-v1/](program-v1/README.md). It holds the frozen protocol, the
pre-registered precision surface with its development measurements, and the Dream-RSI cycles. Its
README states what is measured and what is still running.

## Closed studies kept outside the reset

These are separate from the compression clean room. They are not used to select compression
recipes.

| Study | Question | Status | Folder |
|---|---|---|---|
| SmolVLM backbone (S500) | Can SmolVLM2-500M replace Qwen with limited fine-tuning? | Negative. 2.2% after 10k steps | [evaluation/](evaluation/README.md) |
| SmolVLM recovery | Do corrected inputs and wider adaptation recover S500? | Closed. 31/100 development vs B16 82/100 | [recovery/](recovery/README.md) |
| n0008 confirmation | Does 10k-step training of Dream-RSI's best SmolVLM adapter succeed on LIBERO? | Closed. 0/10 development | [confirmation/](confirmation/README.md) |
| SmolVLA check | How does the complete pretrained SmolVLA policy do? | Quick check: 7/10, one trial per task | [smolvla/](smolvla/README.md) |

Hardware for local runs: [HARDWARE.md](HARDWARE.md).

## Protocol

The clean-room protocol is frozen in [program-v1/protocol.json](program-v1/protocol.json). Core
points:

- LIBERO-Spatial, ten tasks. Development uses 10 states per task; reserved uses the 50 official
  states per task.
- Fixed per-episode seeds, 280-step limit, seven-action chunks, four flow steps, BF16, batch size 1.
- Success: Wilson 95% interval. Differences: task-stratified paired bootstrap, 10,000 resamples.
- Reserved trials never guide selection. No phase is pooled with another.
- Every run records code revision, checkpoint revision, recipe hash, environment, GPU, seeds and
  exit status. Failed runs stay recorded.
- The precision surface runs through `vla-compress-queue`; `vla-compress run` measures one arm and
  phase.

## Evidence rules

- Write new results to a new path; do not overwrite another participant's records.
- Keep weights, datasets, raw logs and runtime state outside Git (`outputs/`).
- Export compact results with provenance before deleting runtime artifacts.
- A number quoted in the report must trace to a committed record under `studies/program-v1/`.
