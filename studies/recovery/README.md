# SmolVLM recovery

Research stopped for the requested October 2 closeout. Acceptance requires **at least 25% lower peak inference memory and a success drop of at most five percentage points against B16**, on matched hardware and final trials. No recovery model has passed this target; the original RSI history is preserved.

**Completed:** corrected-input recovery trained for 20k updates with four trainable decoder layers. Its selected endpoint achieved [31/100 development successes versus B16's 82/100](diagnostics/expanded_development_pair.json) on the same RTX3090 cohort. Training histories and numerical loaders differ, so this does not isolate backbone capacity. The earlier cloud result was 6/10 under its recorded legacy precision; these cohorts are not pooled.

**Final follow-up:** the separate [query/decoder study](query_recovery_registration.json) completed its fixed 10k updates, adapting all 32 decoder layers and four input-query residuals. Held-out arm error selected **9.5k**, independently of robot outcomes; latest 10k state is retained for resume. The [matched RTX5090 check](diagnostics/query_parent5090_development_pair.json) gave **5/10 versus the parent's 4/10**, with one gain and no losses. Ten repeated diagnostic states do not establish reliable improvement. [Curves](figures/F13_query_recovery_progress.pdf) contain all twenty full validations.

The selected model measures **108.6 ms median pipeline latency and 2.94 GiB peak inference allocation** on RTX5090 ([benchmark](diagnostics/query_final_benchmark.json)). This is not a matched comparison with earlier RTX3090/PIL results. A [paired precision audit](diagnostics/query_deployment_precision.json) found 0.43% higher arm MSE after BF16 parameter casting; it did not trigger the registered follow-up or test closed-loop equivalence.

The [local learning-rate comparison](diagnostics/local_lr_pair_comparison.json) completed 500 optimizer updates per branch. Neither the original rate nor a tenfold lower rate beat the matched parent on arm error; the parent remains selected. [F17](figures/F17_local_learning_rate_pair.pdf) retains the actual curves and checkpoint provenance.

Two verified defects informed recovery: GPU processing rescaled 0–1 images again, reducing contrast by 255×; the legacy loader rounded rotary frequencies to BF16. Corrected processing and native rotary buffers are separately recorded. These findings have not explained the whole control gap. Historical results remain available with their original provenance.

**Closeout:** both final native checkpoints and resume state were hash-verified locally (26 files), then the rental was deleted. Recorded project spending is **USD11.27**, based on provider account credit differences ([receipt](diagnostics/cloud_closeout.json)). No further paid training is planned. The investigation found useful defects and partial recovery, but did not deliver a SmolVLM policy comparable to B16.

- [Methodology and diagnostic history](METHODOLOGY.md)
- [Compact evidence and hashes](diagnostics/)
- [Original recovery curves](figures/F8_recovery_progress.pdf)
- [Concise LaTeX report](../evaluation/report/report.pdf)

Rebuild query curves with `python -m evaluation.recovery_progress --study query`. `python -m evaluation.control_trace --help` documents development-only command/state traces; raw traces are excluded from timing benchmarks and final trials.
