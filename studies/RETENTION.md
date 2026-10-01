# Experiment retention

Dream-RSI screening/promotion probes disable checkpoint saving. Keep its journal, candidate source, proposals, metrics, telemetry and logs so the same study can resume and use every observed attempt. Cached public models and the dataset are shared, not copied per attempt.

For completed main evaluations, `python scripts/evaluation_cleanup.py` previews cleanup; add `--apply` to export a compact removal manifest before deleting intermediate checkpoints. Keep milestone inference weights (1,500/5,000 steps) and the selected 10,000-step checkpoint with optimizer state. Removed intermediate steps cannot be resumed; their measured metrics and recorded weight hashes remain attributable. The selected checkpoint remains usable for final evaluation and cloud transfer.

The overnight disk timer tries this safe cleanup before stopping for insufficient capacity. It never deletes the current study or arbitrary files. If nothing safe remains to reclaim, it stops rather than filling the filesystem. Check free space with `df -h .` and inspect `studies/evaluation/retention/` for cleanup provenance.
