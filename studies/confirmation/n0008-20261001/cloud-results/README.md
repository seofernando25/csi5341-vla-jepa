# n0008 cloud confirmation

RTX 5090 continuation from native step 5,000 to 10,000; frozen SmolVLM backbone, trainable adapters/action head/latent predictor. Held-out loss: 0.2653 at 10k. Development: 0/10 valid trials (one per task), not final LIBERO. Full-pipeline latency: 302.8/407.7 ms median/p95; allocated inference VRAM: 2.97 GiB. Full native checkpoint verified before rental deletion; total spend about $2.17.

[Summary and provenance](summary.json), [validation measurements](validation.csv), [vector figure](figures/F6_cloud.pdf). Published timing/episode CSVs are linked through the summary's evidence paths. Raw training diagnostics and weights stay local.

Reproduce the figure from committed inputs:

```bash
python -m evaluation.cloud_analysis --from-summary
python -m evaluation.report
```

Lower prediction loss did not establish useful closed-loop control. Training exposure, frozen representations and pipeline correctness are not isolated here. A matched base confirmation and full development/final evaluation remain pending. RTX 5090 results are separate from the earlier RTX 3090/PIL campaign. The first benchmark hit its operational timeout; only the completed retry supplies reported timings.

Follow-up: the [recovery audit](../../../recovery/README.md) verified double image rescaling in the frozen GPU pipeline. This is a major confound for interpreting these architecture results. The original measurements remain unchanged; corrected-pipeline training/evaluation require a separate study.
