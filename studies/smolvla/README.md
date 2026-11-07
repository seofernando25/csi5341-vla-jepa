# SmolVLA quick check

Reference point for a small VLA trained end-to-end: the complete pretrained
`HuggingFaceVLA/smolvla_libero` policy (604,934,176 parameters), not our SmolVLM transplant.

- **7/10 successes** on LIBERO-Spatial, one trial per task (Wilson 95%: 39.7–89.2%). Failed
  tasks: 2, 3, 7. One trial per task is preliminary evidence only.
- Timing pilot, RTX 3090: 304.5 ms median per replan, 1.19 GiB peak allocated (ten predictions
  after twenty warm-ups; not a full benchmark).
- Settings: first official state per task, seeds `1000 + task_id`, two 256-pixel cameras, native
  normalization and precision, 50 predicted actions with one executed, ten flow steps.

Outcomes and settings: [summary.json](summary.json); episodes:
[run record](../evaluation/runs/20261002T225820842709Z-SmolVLA-Spatial10-rollout/episodes.csv).
The evaluation code was removed after the check; it is in Git history at commit `82df05d`.
