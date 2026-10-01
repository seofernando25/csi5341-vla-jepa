# Dream-RSI (optional)

One evolving SmolVLM architecture study, starting from a clean baseline. Discovery and novelty checks use only this study’s observations. Attempt IDs continue across stops, resumes, and budget extensions.

Architecture proposals and search-policy revisions use `gpt-6.1-sol` with `high` reasoning effort.

The active search uses on-device Torchvision image processing, preserving image size and tiling but using bicubic instead of PIL Lanczos. This pipeline was rebaselined after the September 30 CPU study; prior journals and compact results remain archived separately. Historical paper evaluations retain their original PIL protocol.

```bash
python -m rsi init
python -m rsi run
python -m rsi status
python -m rsi stop
python -m rsi run --resume
# When the allocated budget is exhausted, add another cycle:
python -m rsi run --resume --extend-cycles 1
python -m rsi export
python -m rsi confirm-plan
```

Run from the project root with the environment activated and `PYTHONPATH=src:.`. Set `RSI_DATASET_ROOT` to the local LIBERO-Spatial LeRobot dataset. `python -m rsi dry-run` validates the loop without models, GPU, or agent calls.

Each cycle allows eight research probes and 100 policy replay trajectories. Batches propose four candidates, preferring two refinements and two new families; without eligible parents, slots start at the clean root. Evaluation is sequential on one GPU. Existing candidates remain available as parents across cycles.

Every candidate receives a 500-step screen; qualifying candidates may receive a separate 1,500-step probe against a matched baseline. Initial limits are three cycles, 24 research probes, 32 reservations, four runtime failures, and six promotions. One added cycle grants eight probes, twelve reservations, four runtime failures, and two promotions. Budget exhaustion pauses the study.

State lives in the append-only `.rsi/events/` journal. Resume closes interrupted reservations and assigns new IDs to replacement work; it does not resume individual training checkpoints. Preserve `.rsi/` to continue the study. Export compact results before cleanup. Frozen source/configuration keep comparisons consistent; budget changes are recorded separately.

Candidates may change the SmolVLM representation/adapter architecture, not data, preprocessing, evaluator, inherited losses, or VLA-JEPA action/world-model architecture. Search loss is a proxy; final claims require matched confirmation training and LIBERO evaluation.

## Overnight operation

`scripts/rsi_overnight.py --until TIMESTAMP` supervises the same journal through CLI stop/resume commands. Run under a persistent user systemd service with `Restart=on-failure`, `KillMode=control-group`, and a fixed timezone-aware deadline. It waits for GPU recovery, requests a stop after two failed GPU checks or 40 minutes without log/journal progress, and exports compact results at closeout. Completed work survives reboot; interrupted probes restart as new attempts. Resource-cap extensions use the CLI; runtime-failure caps and policy stops end the run. Account quota resets require the app.

For the installed overnight service: `systemctl --user status csi5341-rsi-overnight`, `journalctl --user -u csi5341-rsi-overnight -f`. To cancel automatic resumption: `systemctl --user disable --now csi5341-rsi-overnight`. Keep the PC powered and awake. User lingering enables startup after reboot without login; it does not reboot or repair a failed driver.
