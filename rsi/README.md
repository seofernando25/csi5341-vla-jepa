# rsi

Dream-RSI over compression solutions. Paper-to-code map: [docs/DREAM_RSI.md](../docs/DREAM_RSI.md).
Commands: [docs/COMMANDS.md](../docs/COMMANDS.md) (`vla-rsi`, entry point `rsi.__main__:main`).

## Layout

Each subpackage's `__init__` re-exports what callers need; import from the package, not a
submodule, unless a test patches one.

| Path | Holds |
|---|---|
| `__main__.py` | The `vla-rsi` CLI: thin dispatch only |
| `api.py`, `tree.py`, `objective.py`, `dream.py`, `synth.py` | Paper interfaces: policy API, replay trees, objective $V$, beta schedule, synthetic worlds |
| `policies/` | Baseline and incumbent replay policies |
| `policy_audit.py`, `replay_worker.py`, `plan_worker.py`, `gpu.py` | Run inside the sandbox as `python -m rsi.<name>`; their module paths are fixed |
| `paths.py` | Repo paths and sandbox mount points (stdlib only) |
| `runner.py`, `gate.py` | Protocol loader, phase lock, gates |
| `agents/` | Agent backends (`pi`, `claude`), dispatch (`command`), one sandboxed run (`run`), transcript parsing (`output`), run records (`result`) |
| `sandbox/` | `bwrap` mounts (`bwrap`), process runner (`process`), fixed evaluator run (`evaluator`) |
| `expansion/` | One node: expand and promote (`node`), inputs (`context`), verdict (`verdict`), agent-output checks and path redaction (`outputs`), lifecycle (`states`) |
| `online_cycle.py`, `online_question.py`, `online_evidence.py`, `grid_plan.py`, `workspace.py` | One live cycle: driver, live question, evidence and prompts, grid plan, snapshot store |
| `policy_dev/` | Policy development: cycle and selection (`develop`), one step (`step`), the agent's view (`agent_env`), replay (`replay`), step status (`states`) |
| `loop/` | Outer loop (`run`, `records`), state files and lock (`store`), detached launch (`launcher`, `status`), status machines (`states`) |
| `watch/` | Watch page: state payload (`state`, `nodes`, `policy`), sources (`source`), transcripts (`transcript`, `events`, `pi_dialect`), panels (`context`, `dream`), server (`server`, `stream`, `assets`), page (`static/`) |
| `provenance.py`, `redact.py`, `calibrate.py`, `export_nodes.py` | Run provenance, secret redaction, instruction calibration, latency-run export |
| `prompts/`, `workspace_template/` | Agent prompts and the root workspace every node forks |

Dependencies point one way: `paths` <- `gpu` <- `sandbox` <- `agents` <- `expansion`,
`policy_dev` <- `online_cycle` <- `loop`; `watch` only reads files and the lock.

## State machines

Statuses on disk are plain strings. Each is a `StrEnum` with one transition table
(`evaluation/fsm.py`); a status change goes through the table and an illegal one raises
`IllegalTransition` naming both states and the legal moves. Old records load unchanged.

| Machine | Module | Values |
|---|---|---|
| Loop | `loop/states.py` `LOOP` | `running`, `stopped`, `done`, `error` |
| Cycle and policy-dev stages | `loop/states.py` `CYCLE_STAGE`, `POLICY_DEV_STAGE` | `pending`, `running`, `done`, `skipped` |
| Loop as viewed | `loop/states.py` `observed_status` | the loop values plus `idle` |
| Launch mode and stop | `loop/states.py` `LaunchMode`, `STOP` | `systemd`, `detached`; `requested` -> `waiting` -> `clean`, `unit-stop`, `signal`, `unknown-holder` |
| Node | `expansion/states.py` `NODE` | `pending`, `agent`, `evaluating`, `done`, `stopped` |
| Evaluator status, fail class | `expansion/states.py` `EvalStatus`, `FailClass` | `ok`, `solution_error`, ...; the `rsi.api` fail classes |
| Policy step | `policy_dev/states.py` `STEP` | `incumbent`, `rejected`, `not_better`, `accepted`, `timeout`, `timeout_salvaged` |
| Measurement-queue cell | `evaluation/compression/queue.py` `CELL`, `next_action` | `pending`, `failed`, `done` |

`tests/test_rsi_states.py` and `tests/test_queue.py` walk each machine, reject an illegal move and
load every value recorded under `studies/`.
