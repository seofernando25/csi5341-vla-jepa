# Reserved-500 results

Six cells, 500 trials each, one serial pass in the order frozen in [../freeze.json](../freeze.json).
Recipe cells ran through the measurement queue; the two Dream-RSI workspaces ran through the fixed
evaluator's reserved phase. Every cell ran once; none was rerun.

| Cell | Runner | Start (UTC) | Wall | Code revision |
|---|---|---|---:|---|
| `B16` | queue | 01:29:12 | 66.9 min | `f663d34` |
| `iter_001-b1-a3` | solution | 02:36:21 | 81.5 min | `e326730` |
| `iter_000-b0-a3` | solution | 03:57:58 | 111.6 min | `93e77cb` |
| `precision-lang-int8` | queue | 05:49:42 | 56.2 min | `64206a6` |
| `precision-lang-nf4` | queue | 06:45:56 | 46.8 min | `e5bc91d` |
| `ladder-nf4-graph-1step-packed` | queue | 07:32:49 | 36.6 min | `e5bc91d` |

[summary.json](summary.json) holds the successes, Wilson 95 % intervals, per-task counts and the
task-stratified paired differences against B16. `summarize.py` regenerates it from the committed
`episodes.csv` files.

## Commands

```sh
uv run --no-sync vla-compress-queue --queue studies/program-v1/queue-reserved-reference.json --allow-reserved
TMPDIR=<private dir> uv run --no-sync vla-compress solution --workspace <export> --phase final \
  --allow-reserved --out studies/program-v1/reserved/iter_001-b1-a3/final
TMPDIR=<private dir> uv run --no-sync vla-compress solution --workspace <export> --phase final \
  --allow-reserved --out studies/program-v1/reserved/iter_000-b0-a3/final
uv run --no-sync vla-compress-queue --queue studies/program-v1/queue-reserved-surface.json --allow-reserved
uv run --no-sync python studies/program-v1/reserved/summarize.py
```

The two queue files are run-order slices of the pre-registered `queue-reserved.json`, which is
unchanged. The node workspaces were exported read-only from `cycles/<cycle>/snapshots.git` at the
commits in [../exploit/remeasure.json](../exploit/remeasure.json); their `workspace_sha256` matches
the node `score.json`.

## Provenance

Each record names its own code revision, environment and GPU. `evaluation/` did not change between
the first and the last cell; the commits in between touch `HANDOFF.md`, `web/` and
`rsi/watch_static/`. The GPU was shared with the `program-v2` Dream-RSI run, so wall times are
inflated and vary.

## Limits

- Reserved states are the official 50 LIBERO-Spatial init states per task. Development uses 10
  other states, so absolute rates differ: B16 scores 82/100 in development and 475/500 in reserved.
  The paired comparison uses identical states and seeds; the two phases are never pooled.
- Recipe memory is resident tensor storage, not the fixed evaluator rule the Dream-RSI nodes use.
  The two figures are not comparable.
- `ladder-nf4-graph-1step-packed` had no smoke or development rollout before this run; it enters
  reserved as the pre-registered deployment arm, not as a gate-passing candidate.
- Task 5 drives most of the spread: 31 to 44 of 50 across cells, against 45 to 50 elsewhere. The
  deployment arm's +2.8 pp is +11 trials on task 5 and +3 on task 8, with the other eight tasks
  within 2 trials of B16.
