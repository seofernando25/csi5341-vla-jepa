# Study A: decoder quantization and the SmolVLM backbone

Matched comparison of four configurations on LIBERO-Spatial (500 reserved trials each, RTX 3090):

| ID | Configuration | Success | Peak allocated | Median latency |
|---|---|---:|---:|---:|
| B16 | `lerobot/VLA-JEPA-LIBERO`, BF16 | 475/500 | 5.22 GiB | 79.1 ms |
| Q8 | Qwen decoder `nn.Linear` → LLM.int8 (threshold 6) | 474/500 | 4.00 GiB | 157.4 ms |
| Q4 | Same scope → NF4, double quantization, BF16 compute | 479/500 | 3.29 GiB | 93.6 ms |
| S500 | SmolVLM2-500M + adapter, 10k fine-tuning steps | 11/500 | 3.07 GiB | 1269.7 ms |

Machine-readable summary: [analysis/summary.json](analysis/summary.json), [T1_summary.csv](analysis/T1_summary.csv).
Per-episode outcomes: `runs/<run-id>/episodes.csv`. Vision tower, embeddings, action head and
world model stay BF16 in Q8/Q4; quantization runs after checkpoint restoration.

## Baseline provenance

B16 loads `lerobot/VLA-JEPA-LIBERO` at revision `735d9f692981e286ade093b5046627eda876e5d0`,
LeRobot's [conversion](https://huggingface.co/docs/lerobot/vla_jepa#pretrained-checkpoints) of the
authors' [checkpoints](https://huggingface.co/ginwind/VLA-JEPA). Qwen3-VL-2B, seven-action chunks,
four denoising steps and BF16 match the paper's Appendix A and the authors' evaluation script.
We measure 95.0% on Spatial; the paper reports 96.2%. We use the LeRobot port and our evaluator,
not the authors' evaluator, so numerical equivalence is not established.

## S500

S500 replaces Qwen with `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, projects its decoder output
into the existing conditioning interface, and trains the adapter, action head and latent predictor
with SmolVLM and the V-JEPA 2 encoder frozen ([training_config.json](../../evaluation/training_config.json)).
Demonstrations are split by whole trajectories: 384 training, 48 held-out
([training_split.json](training_split.json)). Held-out loss selected step 10,000 from checkpoints at
1,500/5,000/10,000 ([selection.json](selection.json)); simulator scores were not used. SmolVLM
expands each image into 17 tiles of 512 pixels, which explains most of its latency.
S500 is a deployment comparison under limited adaptation, not a controlled test of backbone size.

## Reproduce

Run from the repository root on a Linux NVIDIA host. GPU jobs run one at a time for comparable timing.

```bash
uv sync --frozen --extra dev --extra eval --extra report && source .venv/bin/activate
vla-sources baseline qwen world_model simulator_assets
vla-eval collect                                                   # observation bank
vla-eval benchmark --variant B16                                   # repeat for Q8, Q4
vla-core-benchmark --variant B16                                   # neural-only timing
vla-eval rollout --variant B16 --phase final --episodes 50         # repeat for Q8, Q4
vla-eval-analyze --runs "$CSI_B16_RUN" "$CSI_Q8_RUN" "$CSI_Q4_RUN"
```

S500 training and checkpoint selection (`evaluation/train.py`, `evaluation/adaptation.py`, and
`src/lerobot_policy_vla_jepa_smolvlm/`) belong to a closed negative arm and are preserved in Git
history at commit `82df05d`. The analyzer (`vla-eval-analyze`) rejects mismatched hardware or
software, duplicate trials and missing records.

## Frozen controls

- Revisions: [sources.json](../../evaluation/sources.json); inference protocol: [protocol.json](../../evaluation/protocol.json).
- Initial states: [initial_states.json](initial_states.json); observation bank: [observation_bank.json](observation_bank.json).
- Each run record stores the Git revision it ran from (`environment.git_revision`).
- Timing modes: `policy` excludes outer observation preparation and action postprocessing;
  `pipeline` includes them; neural-only timing excludes tokenization and image processing and
  first matches native outputs on all 100 bank observations.
- Hardware: [hardware/local_rtx3090.json](hardware/local_rtx3090.json).
