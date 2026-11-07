# Dream-RSI (optional, closed)

An evolving search over SmolVLM backbone conditioning, after
[Dream-RSI](https://arxiv.org/abs/2609.14858). One study, one append-only journal across stops,
resumes and budget extensions. Results and their caveats: [studies/dream-rsi](../studies/dream-rsi/README.md).

## Boundaries

- The scientific configuration, scoring, evaluator and base source are frozen once a study is
  initialized. Use the CLI for lifecycle and budget changes; never hand-edit `.rsi/` state.
- Candidates may change only backbone representation, adapters, fusion/conditioning and explicit
  trainability. Data, split, preprocessing, scoring, action/world-model architecture and
  initialization, inherited losses and prediction semantics stay fixed.
- Discovery workers see isolated source, this study's observed history and read-only per-batch
  diagnostics. They may run CPU analysis and bounded code checks; they may not train, benchmark
  or run rollouts, and never see Git history, archived studies, the agent board or credentials.
- Discovery-context changes, when authorized: `python -m rsi update-discovery-context --reason TEXT`
  while stopped. The CLI records the amendment in the same journal.
- Search loss is a proxy. Claims need matched confirmation training and LIBERO evaluation.

## Commands

Run from the repository root with `PYTHONPATH=src:.` and `RSI_DATASET_ROOT` set to the local
LIBERO-Spatial LeRobot dataset.

```bash
python -m rsi dry-run                         # validate the loop without models, GPU or agents
python -m rsi init
python -m rsi run                             # or: run --resume [--extend-cycles 1]
python -m rsi status
python -m rsi stop
python -m rsi export                          # compact results into studies/dream-rsi/
python -m rsi confirm-plan
```

Each cycle allows eight research probes and 100 policy-replay trajectories. Batches propose four
candidates (two refinements, two new families). Every candidate gets a 500-step screen; qualifying
candidates get a 1,500-step promotion against a matched baseline. Initial limits: three cycles,
24 probes, 32 reservations, four runtime failures, six promotions. Budget exhaustion pauses the
study. Resume closes interrupted reservations and assigns new IDs; it does not resume training
checkpoints.

`scripts/rsi_overnight.py --until TIMESTAMP` supervises the journal through the same CLI and
exports results at closeout. `scripts/rsi_dashboard.py` serves the read-only
[dashboard](../web/rsi-dashboard/README.md).
