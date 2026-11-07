# SmolVLM recovery (closed)

Could the failed S500 policy (2.2% success) be recovered with corrected inputs and wider
adaptation? Partly: the best recovered policy scores 31/100 development trials against 82/100 for
B16 on the same states. No recovery model met the project target. Research stopped on 2026-10-02.

## Results

| Step | Change | Evidence | Outcome |
|---|---|---|---|
| Input fixes | GPU processor rescaled 0–1 images twice (255× lower contrast); legacy loader rounded rotary frequencies to BF16 | [image range](diagnostics/image_range.json) | Both fixed; neither explains the whole gap |
| RGB recovery | Four final decoder layers adapted for 20k updates from the 10k S500 weights | [expanded development pair](diagnostics/expanded_development_pair.json) | 31/100 vs B16 82/100 (RTX 3090) |
| Query/decoder follow-up | All 32 decoder layers + four input-query residuals, 10k updates; held-out arm error selected 9.5k | [RTX 5090 check](diagnostics/query_parent5090_development_pair.json) | 5/10 vs parent 4/10 (one gain, no losses) |
| Learning-rate pair | Original vs 10× lower rate, 500 updates each | [comparison](diagnostics/local_lr_pair_comparison.json) | Neither beats the parent on arm error |
| Precision audit | BF16 parameter casting | [audit](diagnostics/query_deployment_precision.json) | +0.43% arm MSE; closed-loop effect untested |

The selected query policy measures 108.6 ms median pipeline latency and 2.94 GiB peak allocation
on an RTX 5090 ([benchmark](diagnostics/query_final_benchmark.json)); this is not comparable with
RTX 3090 results. Ten repeated development states cannot establish reliable improvement.
Training histories and numerical loaders differ from B16, so the gap does not isolate backbone
capacity.

Cloud spending for this study and the n0008 confirmation: USD 11.27 from provider credit
differences ([receipt](diagnostics/cloud_closeout.json)). Final native checkpoints (26 files) were
hash-verified locally before the rental was deleted.

Figures: [RGB recovery](figures/F8_recovery_progress.pdf), [query recovery](figures/F13_query_recovery_progress.pdf),
[expanded development](figures/F16_expanded_development.pdf), [learning-rate pair](figures/F17_local_learning_rate_pair.pdf).

The recovery code was removed after the study closed; it is in Git history at commit `82df05d`.
