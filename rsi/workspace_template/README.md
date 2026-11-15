# Compression workspace

## Contract
`solution.py` must define a top-level `compress(policy, calibration)` that returns a policy
object exposing `select_action` and `predict_action_chunk`. `proposal.md` describes the
change. Both files are required and checked before a node is scored.

## Where things are
- `/tmp/work`: this workspace, writable. Your files, packages and tools.
- `/tmp/repo`: project source, read-only. `evaluation/models.py` loads the policy;
  `evaluation/compression/` holds existing code you may import or copy.
- `/tmp/evidence`: read-only. `TASK.md`, `tree.json`, and for each earlier node
  `nodes/<cell_id>/proposal.md`, `score.json`, `error.txt`.
- `/tmp/out`: evaluator output directory for the agent run.

## Calibration (`calibration` argument, loaded from `outputs/compression/calibration.pt`)
- `linear_inputs`: dict of `"<root>.<module path>"` to float32 tensor of shape `(in_features,)`,
  the mean of the squared input per channel over the calibration tokens of selected Linear layers.
- `block_influence`: dict of `"<root>.<index>"` to float, mean residual influence of that block.
- `token_ids`: 1-D integer tensor of the unique prompt token ids seen during calibration.
- `bank_indices`: list of 20 ints, two observations per task.

Roots are `text`, `vision` and `action`. The calibration holds no robot outcomes.

## Model parts
- text decoder: `policy.model.qwen.model.model.language_model` (blocks in `.layers`)
- vision encoder: `policy.model.qwen.model.model.visual` (blocks in `.blocks`)
- action head: `policy.model.action_model` (blocks in `.model.transformer_blocks`)

## Check
`python tools/check.py` runs the evaluator's offline check on this workspace (GPU slot
held) and prints memory removed, action error against the uncompressed policy, latency,
or the error. Exit code 0 only if status is `ok`. Results go to `.check/<timestamp>/`.

## Scoring and memory
- Memory: the returned object and everything reachable from it or from this workspace's
  modules, each storage counted once, or live CUDA bytes if larger. Memory removed is
  `1 - resident / baseline`, where baseline is the untouched policy measured the same way.
- Violations fail the node: meta or off-device tensors, memory maps, files opened outside
  the workspace during `compress`, and files created or changed in the workspace, the
  output directory or the temp directory during `compress`.
- `compress` has a wall-clock limit of 900 s.
- Action error is not task success. A large error predicts failure.
