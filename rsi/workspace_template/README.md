# Node workspace

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
`python tools/check.py` runs the evaluator's offline check on this workspace (GPU slot held) and
prints the score, the figures behind it, any guard violation, or the error. Exit code 0 only if
status is `ok`. Results go to `.check/<timestamp>/`.
Where a study hides the compression code, the check measures with the evaluator's own resident walk
and payload sweep in this workspace's process instead; the evaluator is still the judge.

## Scoring
The score is a single number defined by the study's `protocol.json`; `tools/check.py` prints it
and the measurement it came from. The evaluator also records resident bytes, peak CUDA bytes and
the action change against the untouched policy. A node that hides work from the measurement, or
that fails a guard, fails the node whatever its score would have been. `compress` has a
wall-clock limit of 900 s.
