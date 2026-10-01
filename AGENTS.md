# Project scope

CSI 5341 efficiency comparison: pretrained VLA-JEPA, quantized inference, and a smaller SmolVLM backbone. Dream-RSI is optional. Keep documentation concise and results attributable to their actual model and protocol.

## Coordination

`F` = Fernando's agent; `N` = Noah's agent. Follow `AGENT_BOARD.md`; read the latest remote board before shared work. Append CLM then DONE/BLK locally. Publishing board messages follows the user's communication authorization; never rewrite historical records.

## Research boundaries

- The active plugin is `src/lerobot_policy_vla_jepa_smolvlm/`, using `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`.
- Read `rsi/README.md` and `rsi/config.json` before editing the optional search protocol.
- Dream-RSI starts fresh and maintains one journal across stop/resume/budget extensions. Use only current-study results for discovery and novelty checks.
- Once initialized, freeze scientific configuration, scoring, evaluator, and base source. User-authorized discovery-context changes use `python -m rsi update-discovery-context --reason TEXT` while stopped, recording a harness amendment in the same journal. Use CLI commands for lifecycle/budget changes; never hand-edit runtime state.
- Candidate changes are limited to backbone representation, adapters, fusion/conditioning, and explicit trainability. Preserve data/split, preprocessing, scoring, Qwen baseline, action/world-model architecture and initialization, inherited losses, and prediction semantics.
- Discovery workers receive isolated source, observed current-study history, and read-only per-batch diagnostic logs/metrics. They may use Python dependencies for CPU analysis and bounded code checks, but may not train, benchmark, or run rollouts. Do not expose Git history, archived studies, agent board, or unrelated runtime state.
- Qwen is evaluation-only. LeRobot stays pinned at `30074f7f1358b3c015ae1750017200e86e9c4eb6`.

## Evidence and hygiene

Use `studies/README.md` for comparison metrics. Screen loss is not LIBERO success; runtime failures are not architecture evidence. Export compact new results with provenance before deleting runtime artifacts.

Never commit runtime state, weights, datasets, checkpoints, videos, credentials, virtualenvs, or machine-specific paths. Do not launch training/search as part of routine code validation.
