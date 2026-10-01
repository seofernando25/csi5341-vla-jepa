# Evaluation campaign

[Report](report/report.pdf) · [Plan and deliverables](../EVALUATION_PLAN.md)

B16/Q8/Q4 use the same trained Qwen policy; S500 requires adaptation. Dream-RSI is deferred. CUDA smoke checks establish execution only. Raw CSVs and JSON provenance live in `runs/`; failed and interrupted runs remain recorded.

## Reproduce

Run from the repository root. GPU jobs run sequentially for comparable timing; building the PDF also requires `pdflatex`.

```bash
uv sync --frozen --extra dev --extra eval --extra report
source .venv/bin/activate
python -m evaluation.prepare baseline qwen smolvlm world_model pretrain simulator_assets
python -m evaluation.libero_setup
python -m evaluation.run collect
python -m evaluation.smoke B16
python -m evaluation.run benchmark --variant B16
python -m evaluation.core_benchmark --variant B16
python -m evaluation.run rollout --variant B16 --phase development --episodes 10
python -m evaluation.train --dataset-root "$CSI_DATASET_ROOT" --prepare-only
python -m evaluation.train --dataset-root "$CSI_DATASET_ROOT" --steps 500
python -m evaluation.train --dataset-root "$CSI_DATASET_ROOT" --resume "$CSI_CHECKPOINT" --steps 1500
python -m evaluation.run rollout --variant S500 --checkpoint "$CSI_CHECKPOINT" --phase final --episodes 50
python -m evaluation.analysis --runs "$CSI_B16_RUN" "$CSI_Q8_RUN" "$CSI_Q4_RUN"
python -m evaluation.training_analysis --runs "$CSI_TRAIN_RUN"
python -m evaluation.select_checkpoint --training-runs "$CSI_INITIAL_RUN" "$CSI_TRAIN_RUN"
python -m evaluation.reproducibility --runs "$CSI_B16_RUN" "$CSI_Q8_RUN" "$CSI_Q4_RUN"
python -m evaluation.report
```

`CSI_DATASET_ROOT` identifies the local LIBERO-Spatial LeRobot dataset. Checkpoints and observations stay under ignored `outputs/evaluation/`; the environment variables above are user-supplied paths, not committed machine settings. Repeat inference commands for Q8/Q4 and, after selection, S500. `--resume` takes a native checkpoint's `pretrained_model` directory and restores optimizer, scheduler, RNG and sampler state. `--steps` is the cumulative target.

Select run directories explicitly for analysis. The analyzer rejects mismatched controls, unequal timing protocols, duplicate trials and missing records. Main figures use matched complete runs; timing pilots are separate. Save vector PDF and 300-dpi PNG figures with hashed input records. Missing values stay missing.

The neural-only benchmark excludes tokenization and image processing, and must first match native-policy outputs on all 100 bank observations under identical noise seeds. Include its completed run directories in `evaluation.analysis` to export neural median/p95 alongside full pipeline timing. Exact execution-source archives are indexed in [`sources/index.json`](sources/index.json); their contents are checked against each run's recorded source hashes.

Execution-source archives are overlays, not standalone packages. To restore a measured implementation, use an isolated checkout at its `environment.git_revision`, overlay the indexed ZIP, and retain this study's frozen JSON manifests. Verify the archive and source hashes before running. Dataset hashes identify the exact local conversion; an upstream conversion revision has not been established.

## Frozen controls

- Revisions: [`sources.json`](../../evaluation/sources.json); inference: [`protocol.json`](../../evaluation/protocol.json).
- Robot states: [`initial_states.json`](initial_states.json); development states are disjoint from the 50 official final states per task. Keep development diagnostics separate from final benchmark scores.
- Demonstrations: [`training_split.json`](training_split.json); 384 training / 48 held-out trajectories. Hashes describe the local conversion, not an asserted upstream dataset revision.
- Adaptation: [`training_config.json`](../../evaluation/training_config.json); frozen SmolVLM and target encoder; train adapter, action head and latent predictor. Baseline normalization stays fixed. Validation is held out from this adaptation run, not necessarily from the published models' prior training.
- Timing includes backbone tokenization and image processing in both modes. `policy` excludes outer observation preparation and action postprocessing; `pipeline` includes them. Neither includes simulator stepping or rendering. Q8 internally casts decoder inputs to FP16; duplicate backend warnings are logged once.
- The [SmolVLM processor probe](checks/processor_probe.json) verifies correct float-image scaling and records 17 native 512-pixel tiles for one 224-pixel input. Camera/outer policy preprocessing is shared; backbone-native processing differs. Interpret S500 as a deployment comparison, not an isolated parameter-count experiment.

The [500-step pilot](training/20260923T235657557094Z-S500-train/run.json) completed; [F5](adaptation/figures/F5_adaptation.pdf) contains its measured validation loss. The [frozen budget](budget.json) retains 1,500/5,000/10,000-step checkpoints, seed 42, with approximately ten projected GPU-hours. The [development comparison](development/summary.json) and [F1–F4 figures](development/figures/) cover B16/Q8/Q4 at 100 episodes each. Final success and corresponding figures, further S500 adaptation/development points, and the complete T1 remain outstanding.

The [LIBERO maintainer’s explanation](https://github.com/Lifelong-Robot-Learning/LIBERO/issues/34) describes fixed states as seeded environment initializations and pruning as evaluation subsampling. Counts and exact membership here are verified against the installed files.
