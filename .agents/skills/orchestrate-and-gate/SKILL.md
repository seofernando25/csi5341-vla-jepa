---
name: orchestrate-and-gate
description: Orchestrator trajectory discipline, log-first adversarial diagnosis, and sim-to-real release gates. Use when running multi-step debugging, spawning T3 subagents, diagnosing policy/control failures, or deploying RL policies to real hardware.
---

# Orchestrate and gate

Prevent linear-thread bloat, false root-cause retrains, and unverified hardware deployments.

## 1. T3 Orchestration & Delegation

- **Split early, never run linear marathons.** Do not chain >3–4 diagnostic or hardware-debug turns in one agent. Delegate file/log/thread scouting to `gemini-4-hydrogen` (`thinking_level: low`) in parallel, and keep a separate **adversarial reviewer** (`thinking_level: medium/high`) running against every diagnosis before acting.
- **Check workspace access before delegating.** Verify the target repo matches the active project/sandbox (`csi5341-vla-jepa` vs `ugv`) before spawning workers; use the right `projectId` or provider so pre-tool hooks do not reject edits.
- **Never poll or block on `t3_thread_wait`.** Launch `delegate_task(..., mode="async")` and end the turn. Completion wakes the thread.
- **Keep `t3_thread_read` bounded.** Always pass `limit=20, maxCharsPerItem=2000` and page with `afterPosition`; never dump `maxCharsPerItem=6000+` into context.
- **Verify environment & parity once up front.** In one command, check `git rev-parse HEAD`, `git status -s`, Python venv packages (`rl/.venv` vs root `.venv`), and OS flags (`macOS nc -G`, `OpenWrt uname -n`) before running or patching remote hosts. Create `scratch/` with `mkdir -p` before `scp`, use `ssh -M -S <sock>` control sockets instead of `pkill -f`, and never run broad `grep` across `~/Documents`.
- **Never patch remotes ad hoc while another agent owns hardware.** Keep one source of truth on host git, and reconcile remote working trees cleanly instead of editing individual UI/runtime files mid-run.

### Multi-Agent Efficiency & Shared-KB Discipline

- **Split into micro-agents (`<= 5` tool calls each).** Never bundle 3–5 unrelated files or tasks into one 40–60 call scout; spawn 4–8 parallel micro-agents scoped to one file, diff, or edit set with an explicit command budget.
- **Query the shared KB and pass targeted slices.** Before scanning `ugv`, run `/home/fer/Documents/ugv/.agent_kb/kb.py` (`facts`, `query <term> --slice N`, `slice <file>:<start>-<end>`) and embed exact file paths, line ranges, SSH/venv facts, and prior conclusions in each child's Task Card so subagents skip `find`, `wc -l`, `grep -rn`, and 500-line `sed` dumps.
- **Refresh the KB on milestones.** After behavior/layout changes or real hardware runs, re-run `kb.py index` and update `FACTS.md` before spawning downstream agents.

## 2. Single-Owner Resource & Interrupt Discipline

- **One GPU job and one robot owner at a time.** Check `nvidia-smi` and `ps` before launching training or hardware scripts; concurrent edits to `sim.py`/trainers or overlapping `l7_run.py` sessions invalidate runs.
- **Make long runs resumable.** Launch GPU training detached with `--ckpt-every <= 25` and never keep milestone weights only in `/tmp`.
- **Recover cleanly after any interrupt.** If a hardware `run.sh` or SSH session is interrupted, `rsync` remote logs immediately, verify the robot is braked and check its true physical rest pose before starting the next step.

## 3. Adversarial Log-First Diagnosis (Before Any Retrain)

Never blame a policy failure on "sensor noise" or "FOV loss" from replay alone, and never launch a GPU retrain until all four checks pass:

1. **Stationary sensor floor:** Measure `std` on stationary/idle ticks (`st=True`), not during closed-loop motion. *(Example: real gyro noise was `0.00077 rad/s`, 2.6x lower than sim's `0.002`; the `0.63 rad/s` signal was a `1.4 Hz` limit cycle.)*
2. **Loop lag & cross-correlation:** Check PSD peak, lag between commanded duty and sensor rate (`~240 ms` HTTP+onset lag), and whether an EMA filter worsens phase margin.
3. **Raw vs masked observation cadence:** Separate true sensor detections (`src == "tag"`) from dead-reckoned/held `valid` ticks (`13.5%` raw vs `46.9%` DR-masked; `40%` `valid=0` ticks zeroing small translation commands).
4. **Plant & adapter parity:** Verify training dynamics (`mean` vs `p0`), axis orientation, deadzones, velocity caps, `stale` timestamp propagation, and `swept_clear` yaw-sign conventions match between `sim.py`, `policy.py`, and the hardware adapter. Run a single-variable discriminating test (e.g., raw sensor + open-loop feed-forward) before retraining.

## 4. Pre-Hardware & Post-Run Gates

- **Live-frame & config parity check:** Before moving wheels, run tag detection + relaxed-param sweep on one live camera frame to verify physical tag IDs match `drone_tag_ids`, confirm `*_confirmed` flags, check camera shutter/gain drop-ins (`<= 4000 us` for moving robots), and ensure probes wait for fresh timestamped `/ws` or `/api/state` deltas (never pre-pulse cached state).
- **Embedded harness hygiene:** Pin `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1`, isolate OpenCV in a `spawn` worker started before I/O threads (`imu`), guard `cornerSubPix` near frame edges, and exclude bulk `frames/` from routine WireGuard `rsync` (`end.jpg` + `jsonl` only).
- **Sim-to-sim SHA gate:** Require a passing (`> 0` success) 3D/Gazebo evaluation matching the **exact checkpoint SHA-256** of the exported `.npz` before moving to hardware.
- **Offline observation-contract gate:** Audit logged `ticks.jsonl` + axis probes against the sim reference distribution (gyro std/span, raw tag rate, inter-tag dt, action saturation/sign-flips, plant deadzones/gains/cross-coupling) and parity-test PyTorch vs NumPy exports (`< 1e-5`).
- **Online runtime monitor (<= 2 s brake):** Brake automatically if 1 s rolling `std(imu_w)`, action saturation (`|a| > 0.8`), z-score vs sim, or 2 s sign-flip / raw-tag-drop thresholds trip.
- **Never trust dead-reckoned `"arrived"`:** On every hardware run, reject PnP flips, compare `end_est` against `end_pose_tag`, confirm near-zero rest velocity (`|imu_w|`), and visually inspect `end.jpg` before advancing to blind `< 0.4 m` final-approach steps.
