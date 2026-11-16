# Compression workspace

## Contract
`solution.py` must define a top-level `compress(policy, calibration)` that returns a policy
object exposing `select_action` and `predict_action_chunk`. `proposal.md` describes the
change. Both files are required and checked before a node is scored.

## Where things are
- `/tmp/work`: this workspace, writable. Your files, packages and tools.
- `/tmp/repo`: project source, read-only. `evaluation/models.py` loads the policy;
  `evaluation/pipeline.py` builds its inputs from an observation.
- `/tmp/evidence`: read-only. `TASK.md`, `tree.json`, and for each earlier node
  `nodes/<cell_id>/proposal.md`, `score.json`, `error.txt`.
- `/tmp/out`: evaluator output directory for the agent run.

## Calibration (`calibration` argument)
- `linear_inputs`: dict of `"<root>.<module path>"` to float32 tensor of shape `(in_features,)`,
  the mean of the squared input per channel over the calibration observations of selected
  Linear layers.
- `block_influence`: dict of `"<root>.<index>"` to float, mean residual influence of that block.
- `bank_indices`: list of 20 ints, two observations per task.

Roots are `text`, `vision` and `action`. The calibration holds no robot outcomes and no
per-prompt information.

## Model parts
- text decoder: `policy.model.qwen.model.model.language_model` (blocks in `.layers`)
- vision encoder: `policy.model.qwen.model.model.visual` (blocks in `.blocks`)
- action head: `policy.model.action_model` (blocks in `.model.transformer_blocks`)

## Check
`python tools/check.py` loads the policy, runs `compress` and prints memory removed, action
error against the uncompressed policy on 80 bank frames, and latency, or the error. It
measures with the evaluator's own resident walk and payload sweep, in this workspace's
process; the evaluator is still the judge. It takes about a minute when the GPU is free.
Exit code 0 only if the workspace ran.

## Scoring and memory
- Memory: the returned object and everything reachable from it or from this workspace's
  modules, each storage counted once, or live CUDA bytes if larger. Memory removed is
  `1 - resident / baseline`, where baseline is the untouched policy measured the same way.
- Violations fail the node: meta or off-device tensors, memory maps, files opened outside
  the workspace during `compress`, and files created or changed in the workspace, the
  output directory or the temp directory during `compress`.
- `compress` has a wall-clock limit of 900 s.
- Action error is not task success. A large error predicts failure.
