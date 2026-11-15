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

## Table I decision rules (full)

The report's Table I abbreviates these; the question column is the plain-language scaffold.

- **E1 (H1)** — decoder linear precision BF16/INT8/NF4; 500 reserved rollouts per arm. Paired
  bootstrap; pass if memory reduction ≥25% and Δ ≥ −5 pp.
- **E2 (H2)** — module × method × strength, 201 recipes; 80 fixed frames, 10-trial smoke (163),
  100-trial development (24). Development gate ≥77/100 (B16 control 82/100 minus 5); selection
  frozen before E3.
- **E3** — two frozen joint recipes; 500 reserved rollouts each. Wilson and paired intervals;
  report regardless of outcome.
- **E4 (H3)** — all E1/E3 arms; 3×500 timed predictions. Median and p95 latency vs peak memory.
- **E5 (H4)** — per-episode outcomes; 80-frame action change. Paired transitions per task; Spearman ρ.
- **E6** — SmolVLM2-500M + adapter, 10k fine-tuning steps; 500 reserved rollouts. Same rule as E1.
- **E7 (H3)** — execution engine (stock/CUDA graph), flow steps (4/1), dense/packed NF4; 3×500
  timed predictions per variant, 500 reserved for C2. Same as E1; the one-step setting comes from
  the E2 smoke screen.
