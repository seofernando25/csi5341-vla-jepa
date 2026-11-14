# Studies

Compact evidence for every experiment. The [final report](../report/main.pdf) summarizes them;
`report/plots/` reads its numbers from here.

| Study | Question | Status and headline | Folder |
|---|---|---|---|
| Quantization (A) | Does INT8/NF4 decoder quantization keep success? | Done. Q4 −37% memory, +0.8 pp; Q8 −23%, −0.2 pp (500 trials each) | [evaluation/](evaluation/README.md) |
| Compression (B) | How far can post-training compression go, and what breaks first? | Done. 201 recipes; two joint recipes keep 96.8–98.0% success at 1.38–1.51 GB (73–75% less) | [compression/](compression/README.md) |
| SmolVLM backbone (S500) | Can SmolVLM2-500M replace Qwen with limited fine-tuning? | Negative. 2.2% after 10k steps | [evaluation/](evaluation/README.md) |
| SmolVLM recovery | Do corrected inputs and wider adaptation recover S500? | Closed. 31/100 development vs B16 82/100 | [recovery/](recovery/README.md) |
| n0008 confirmation | Does 10k-step training of Dream-RSI's best SmolVLM adapter (`n0008`) succeed on LIBERO? | Closed. Held-out loss 0.2653, 0/10 dev (double `÷255` bug) | [confirmation/](confirmation/README.md) |
| Dream-RSI search | Can an automated search find better compression or conditioning policies? | Reopened (SmolVLM: 0/10 dev stands). Replay fidelity repair: held-out V beats incumbent on all folds; Qwen Stage 2 (`s2-c2-t10-v25`: 85/100 dev, 904.8 MB) and live rounds R1–R5 | [dream-rsi/](dream-rsi/README.md) |
| SmolVLA check | How does the complete pretrained SmolVLA policy do? | Quick check: 7/10, one trial per task | [smolvla/](smolvla/README.md) |

Hardware for all local runs: [HARDWARE.md](HARDWARE.md).

## Protocol shared by all rollout studies

- LIBERO-Spatial, ten tasks. Reserved trials use the 50 official initial states per task; development
  trials use 10 other states per task, disjoint by hash ([initial_states.json](evaluation/initial_states.json)).
- Fixed per-episode seeds, 280-step limit, seven-action chunks, four flow steps, BF16, batch size 1.
- Success: Wilson 95% interval. Differences: task-stratified paired bootstrap, 10,000 resamples.
- Latency: synchronized observation-to-action time, 20 warm-ups, 3 × 500 predictions, simulator excluded.

## Evidence rules

- Every run has a unique ID, checkpoint revision, code revision, configuration, hardware and
  seeds. Failed and interrupted runs stay recorded and separate from task failures.
- Keep weights, datasets, raw logs and runtime state outside Git (`outputs/`).
- Write new results to a new path; do not overwrite another participant's records.
