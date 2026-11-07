# Studies

Compact evidence for every experiment. The [final report](../report/main.pdf) summarizes them;
`report/figures.py` reads its numbers from here.

| Study | Question | Status and headline | Folder |
|---|---|---|---|
| Quantization (A) | Does INT8/NF4 decoder quantization keep success? | Done. Q4 −37% memory, +0.8 pp; Q8 −23%, −0.2 pp (500 trials each) | [evaluation/](evaluation/README.md) |
| Compression (B) | How far can post-training compression go, and what breaks first? | Done. 201 recipes; C1–C3 keep 93–98% success at 0.94–1.38 GB | [compression/](compression/README.md) |
| SmolVLM backbone (S500) | Can SmolVLM2-500M replace Qwen with limited fine-tuning? | Negative. 2.2% after 10k steps | [evaluation/](evaluation/README.md) |
| SmolVLM recovery | Do corrected inputs and wider adaptation recover S500? | Closed. 31/100 development vs B16 82/100 | [recovery/](recovery/README.md) |
| Dream-RSI search | Can an automated search find better SmolVLM conditioning? | Closed. Lower loss, 0/10 development successes | [dream-rsi/](dream-rsi/README.md), [confirmation/](confirmation/n0008-20261001/README.md) |
| SmolVLA check | How does the complete pretrained SmolVLA policy do? | Quick check: 7/10, one trial per task | [smolvla/](smolvla/README.md) |
| Review | What went wrong in the SmolVLM work? | Three AI reviews, synthesized 2026-10-02 | [audit/](audit/20261002/README.md) |

Hardware for all local runs: [HARDWARE.md](HARDWARE.md).

## Protocol shared by all rollout studies

- LIBERO-Spatial, ten tasks. Reserved trials use the 50 official initial states per task; development
  trials use 10 other states per task, disjoint by hash ([initial_states.json](evaluation/initial_states.json)).
- Fixed per-episode seeds, 280-step limit, seven-action chunks, four flow steps, BF16, batch size 1.
- Success: Wilson 95% interval. Differences: task-stratified paired bootstrap, 10,000 resamples.
- Latency: synchronized observation-to-action time, 20 warm-ups, 3 × 500 predictions, simulator excluded.

## Evidence rules

- Every run has a unique ID, checkpoint revision/hash, code revision, configuration, hardware and
  seeds. Failed and interrupted runs stay recorded and separate from task failures.
- Keep weights, datasets, raw logs and runtime state outside Git (`outputs/`, `.rsi/`).
  `scripts/evaluation_cleanup.py` previews safe checkpoint cleanup; `--apply` writes a removal
  manifest first.
- Write new results to a new path; do not overwrite another participant's records.
