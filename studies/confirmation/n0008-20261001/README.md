# n0008 confirmation (closed)

Longer training of Dream-RSI's best candidate, n0008: a trainable residual adapter before the final
SmolVLM decoder layer. Lower held-out loss did not produce successful control.

| Measure | Result |
|---|---|
| Held-out loss, 200 samples | 0.2653 at 10k steps (0.3056 at 5.5k) |
| Development success | 0/10, one trial per task |
| Pipeline latency, RTX 5090 | 302.8 / 407.7 ms median / p95 |
| Peak allocated inference memory | 2.97 GiB |
| Cloud cost | about USD 2.17 |

Plan: [plan.json](plan.json): seed 42, batch 2, fixed 10k-step schedule, existing trajectory split,
selection by held-out loss at 1.5k/5k/10k steps. Training ran locally to 5k and on an RTX 5090
from 5k to 10k. The matched SmolVLM baseline arm and full development/final evaluation were not
completed.

The frozen GPU pipeline rescaled images twice (see [recovery](../../recovery/README.md)), so these
results describe a degraded visual input and cannot rank architectures.

Evidence: [cloud-results/summary.json](cloud-results/summary.json), [validation.csv](cloud-results/validation.csv),
[figure](cloud-results/figures/F6_cloud.pdf). Rebuild the figure with
`python -m evaluation.cloud_analysis --from-summary`.
