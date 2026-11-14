# Study: n0008 confirmation (closed)

**Question.** Does training Dream-RSI's lowest-loss SmolVLM conditioning adapter (`n0008`) for 10,000 steps yield successful closed-loop control on LIBERO-Spatial?

**Result.** No. Held-out loss dropped to `0.2653` at 10k steps (`0.3056` at 5.5k), but closed-loop development evaluation scored `0/10` (one trial per task; RTX 5090 pipeline latency `302.8 ms` median, `2.97 GiB` peak allocated). A double `÷255` image-rescaling bug in the frozen GPU processor crushed visual contrast by `255×`, so these numbers describe degraded inputs rather than backbone capacity.

**Evidence.**

- [n0008-20261001/README.md](n0008-20261001/README.md): run summary and caveats
- [n0008-20261001/plan.json](n0008-20261001/plan.json): frozen training and selection plan
- [n0008-20261001/cloud-results/summary.json](n0008-20261001/cloud-results/summary.json): cloud confirmation record
- [n0008-20261001/cloud-results/validation.csv](n0008-20261001/cloud-results/validation.csv): held-out validation curve
- [n0008-20261001/cloud-results/figures/F6_cloud.pdf](n0008-20261001/cloud-results/figures/F6_cloud.pdf): training/validation figure

**Archive.** Closed negative result. The SmolVLM plugin and training scripts (`evaluation/train.py`, `evaluation/adaptation.py`, `src/lerobot_policy_vla_jepa_smolvlm/`) were removed from the live tree and are preserved in Git history at commit `82df05d`.
