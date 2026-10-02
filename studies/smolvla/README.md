# SmolVLA: quick LIBERO-Spatial check

**7/10 successes**, one episode on each of the ten tasks; all ten ran in one GPU batch. Evaluation wall time: **138.7 seconds** including environment startup, excluding policy loading and worker teardown. Approximate 95% Wilson interval: **39.7–89.2%**; one trial per task gives only preliminary evidence.

Successful task IDs: **0, 1, 4, 5, 6, 8, 9**. Failed task IDs: **2, 3, 7**. [Exact outcomes and provenance](summary.json).

We evaluated the complete pretrained `HuggingFaceVLA/smolvla_libero` policy, not our SmolVLM–VLA-JEPA transplant. The whole policy has 604,934,176 parameters, including its action expert. [Policy/backbone revisions](../../evaluation/smolvla_sources.json) and [execution source](execution-source.zip) are pinned and hashed. LeRobot remains at the project pin.

The run used the first official pruned state on each LIBERO-Spatial task, episode seeds `1000 + task_id`, two 256-pixel cameras, 20 Hz relative control, ten simulator processes and one batched policy. Native normalization, mixed parameter precision, FP32 buffers and disabled AMP were retained. The policy predicts 50 actions, executes one, and uses ten flow steps. TF32 was disabled. No training, video recording or cloud spending.

A separate native timing pilot measured **304.49 ms median / 306.97 ms p95** per replan and **1.193 GiB peak allocated** on RTX 3090. This used only ten measured predictions per mode after twenty warmups; it is not a full latency benchmark. The parallel rollout's roughly 10 GiB total GPU usage also includes simulator rendering and driver allocations.

The larger campaign is stopped. Its completed task 0 batch scored 8/10 over ten starting states and is retained separately; it is not ten-task success. Noah's reported 90% has unverified episode/state/seed details. Our quick 10 uses different batched RNG/seeds from the earlier 500-episode study and stays out of the Final LIBERO frontier.

## Reproduce

Cache the two pinned revisions in `evaluation/smolvla_sources.json`, then run from the repository root with the project environment:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u -m evaluation.smolvla_quick
```

Raw outcomes: [20261002T225820842709Z-SmolVLA-Spatial10-rollout](../evaluation/runs/20261002T225820842709Z-SmolVLA-Spatial10-rollout/episodes.csv). The code starts exactly one episode per task and stops after all ten. Existing failed/interrupted records remain separate; no additional campaign is scheduled.
