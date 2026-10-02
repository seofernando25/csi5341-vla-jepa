# SmolVLM recovery

Active goal: obtain a useful SmolVLM LIBERO policy while preserving the original RSI history. Acceptance requires **at least 25% lower peak inference memory and a success drop of at most five percentage points against B16**, on matched hardware and final trials. No recovery model has passed this target.

**Completed:** corrected-input recovery trained for 20k updates with four trainable decoder layers. Its selected endpoint achieved [31/100 development successes versus B16's 82/100](diagnostics/expanded_development_pair.json) on the same RTX3090 cohort. Training histories and numerical loaders differ, so this does not isolate backbone capacity. The earlier cloud result was 6/10 under its recorded legacy precision; these cohorts are not pooled.

**Current:** the separate [query/decoder study](query_recovery_registration.json) starts from that exact parent and trains all 32 decoder layers plus four input-query residuals for a fixed 10k updates. Native-precision RTX5090 development results are 2/10, 4/10 and 2/10 at 500, 2k and 5k updates. [Measured curves](figures/F13_query_recovery_progress.pdf) include complete validation through 9.5k; final control evaluation is pending. Held-out physical arm error selects checkpoints, never robot outcomes.

The [local learning-rate comparison](diagnostics/local_lr_pair_comparison.json) completed 500 optimizer updates per branch. Neither the original rate nor a tenfold lower rate beat the matched parent on arm error; the parent remains selected. [F17](figures/F17_local_learning_rate_pair.pdf) retains the actual curves and checkpoint provenance.

Two verified defects informed recovery: GPU processing rescaled 0–1 images again, reducing contrast by 255×; the legacy loader rounded rotary frequencies to BF16. Corrected processing and native rotary buffers are separately recorded. These findings have not explained the whole control gap. Historical results remain available with their original provenance.

Next: complete the fixed cloud run, verify native/selected exports, run the [registered same-GPU parent comparison](parent5090_native_registration.json), and perform the [local deployment-precision audit](deployment_precision_registration.json). Ten-episode development checks remain diagnostic; final acceptance is still reserved.

The RTX5090 rental costs USD0.526/hour. **USD14 is a ceiling, not a spending target:** release it promptly after required exports and the bounded comparison are verified, without waiting for local analysis or report writing. No additional paid training is planned. Provider cleanup is October 2 at 20:00 UTC.

- [Methodology and diagnostic history](METHODOLOGY.md)
- [Compact evidence and hashes](diagnostics/)
- [Original recovery curves](figures/F8_recovery_progress.pdf)
- [Concise LaTeX report](../evaluation/report/report.pdf)

Rebuild query curves with `python -m evaluation.recovery_progress --study query`. `python -m evaluation.control_trace --help` documents development-only command/state traces; raw traces are excluded from timing benchmarks and final trials.
