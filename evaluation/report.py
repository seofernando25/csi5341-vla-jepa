"""Build a concise typeset progress report from recorded campaign evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from evaluation.common import ROOT, read_json
from evaluation.figures import architecture


def tex(value):
    result = str(value)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ]:
        result = result.replace(old, new)
    return result


def final_measurements_ready(analysis, selection, adaptation):
    """Gate final-paper wording; the separate deliverable audit is still required."""
    if analysis.get("rollout_phase") != "final" or not analysis.get("full_timing_protocol"):
        return False
    if not selection or selection.get("simulator_scores_used") is not False:
        return False
    milestones = selection.get("eligible_steps", [])
    if milestones != [1500, 5000, 10000] or selection.get("selected_step") not in milestones:
        return False
    validation = {r["step"] for r in adaptation.get("validation", []) if r["samples"] == 200}
    development = {r["step"] for r in adaptation.get("development", []) if r["episodes"] == 100}
    if not set(milestones) <= validation or not {500, *milestones} <= development:
        return False
    results = analysis.get("results", {})
    for variant in ["B16", "Q8", "Q4", "S500"]:
        result = results.get(variant, {})
        success, timing, neural = (result.get(k, {}) for k in ["success", "timing", "neural"])
        tasks = success.get("per_task", {})
        if (
            success.get("episodes") != 500
            or set(tasks) != {str(i) for i in range(10)}
            or any(t["episodes"] != 50 for t in tasks.values())
            or timing.get("predictions_per_mode") != 1500
            or timing.get("repetitions") != 3
            or neural.get("predictions") != 1500
            or neural.get("equivalence", {}).get("observations") != 100
            or not neural.get("equivalence", {}).get("identical_flow_noise_seeds")
        ):
            return False
    return (
        results["S500"].get("study_training", {}).get("optimizer_steps")
        == selection["selected_step"]
    )


def render(output):
    output.mkdir(parents=True, exist_ok=True)
    architecture(output / "figures")
    checks = {}
    for path in (ROOT / "studies/evaluation/checks").glob("*.json"):
        evidence = read_json(path)
        checks[path.stem] = evidence
    rows = []
    analysis_path = ROOT / "studies/evaluation/analysis/summary.json"
    analysis = read_json(analysis_path) if analysis_path.exists() else {}
    selection_path = ROOT / "studies/evaluation/selection.json"
    adaptation_path = ROOT / "studies/evaluation/adaptation/summary.json"
    selection = read_json(selection_path) if selection_path.exists() else {}
    adaptation = read_json(adaptation_path) if adaptation_path.exists() else {}
    cloud_path = ROOT / 'studies/confirmation/n0008-20261001/cloud-results/summary.json'
    cloud = read_json(cloud_path) if cloud_path.exists() else {}
    cloud_section = ''
    if cloud:
        measured = cloud['benchmark']
        curve = cloud['validation']
        figure = cloud_path.parent / 'figures/F6_cloud.pdf'
        shutil.copyfile(figure, output / 'figures' / figure.name)
        cloud_section = (
            r'\FloatBarrier\section{Optional architecture confirmation}' + '\n'
            r'The optional discovery study selected candidate n0008, which adds a trainable '
            r'residual adapter near the end of the frozen SmolVLM decoder. A separate confirmation '
            r'continued its native 5,000-step checkpoint to 10,000 steps on an RTX 5090, preserving '
            r'the optimizer, seed, trajectory split, normalization and training objectives. '
            r'The action head, latent predictor and conditioning adapters train; the SmolVLM '
            r'and V-JEPA encoders remain frozen. This continuation is distinct from S500 in T1.' + '\n\n'
            f"Held-out loss decreased from {curve[0]['loss']:.4f} at {curve[0]['step']:,} "
            f"steps to {curve[-1]['loss']:.4f} at 10,000 steps (200 fixed samples). "
            r'Closed-loop development evaluation completed ten valid trials, one per task, '
            r'with zero successes. This small diagnostic does not establish a final success rate '
            r'or performance retention; lower prediction loss did not yield successful manipulation.' + '\n\n'
            f"Full-pipeline median/p95 latency was {measured['pipeline']['median_ms']:.1f}/"
            f"{measured['pipeline']['p95_ms']:.1f} ms; policy-native latency was "
            f"{measured['policy']['median_ms']:.1f}/{measured['policy']['p95_ms']:.1f} ms. "
            f"Peak allocated inference memory was {max(p['allocated_bytes'] for p in measured['memory_peaks'])/2**30:.2f} GiB. "
            r'Each mode used 500 predictions across three repetitions. Policy-native timing '
            r'includes backbone preprocessing and is not neural-only timing. These RTX 5090 '
            r'measurements cannot isolate a pipeline or architecture speedup against the earlier '
            r'RTX 3090/PIL measurements.' + '\n\n'
            r'The full resumable checkpoint was hash-verified before rental deletion. '
            r'The matched SmolVLM baseline confirmation, full development/final trials, and '
            r'fixed-checkpoint cross-hardware comparisons remain pending. Training exposure and '
            r'backbone trainability differ substantially from the original paper; this experiment '
            r'does not isolate undertraining, representation mismatch or a control-pipeline issue.' + '\n'
            r'A subsequent processor audit identified a concrete input defect in this frozen GPU '
            r'pipeline: structured image-device arguments suppressed the flat no-rescale argument. '
            r'Unit-range images were scaled again, reducing pixel contrast by a factor of 255. '
            r'The original results remain attributable to that pipeline; they cannot establish '
            r'a limitation of SmolVLM capacity. The current plugin is corrected and a real-processor '
            r'pixel-range regression passes. Corrected-input training and control evaluation are '
            r'a separate recovery study; no successful recovered policy is claimed here.' + '\n'
            r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/F6_cloud.pdf}'
            r'\caption{Separate n0008 confirmation: held-out action/world prediction loss and '
            r'measured inference latency distributions on the RTX 5090. Loss points use the same '
            r'200 held-out samples; each latency curve contains 1,500 predictions. Closed-loop '
            r'development success was 0/10 and is not inferred from prediction loss.}\end{figure}'
        )
    recovery_path = ROOT / 'studies/recovery/diagnostics/recovery_curve.json'
    recovery_figure = ROOT / 'studies/recovery/figures/F8_recovery_progress.pdf'
    if recovery_path.exists() and recovery_figure.exists():
        recovery = read_json(recovery_path)
        first, last = recovery['validation'][0], recovery['validation'][-1]
        shutil.copyfile(recovery_figure, output / 'figures' / recovery_figure.name)
        cloud_section += (
            '\n' + r'\FloatBarrier\section{Corrected-input recovery}' + '\n'
            r'A separate registered study corrects RGB scaling and adapts the last four '
            r'SmolVLM decoder layers. It starts from the verified legacy 10k weights with '
            r'fresh AdamW, batch eight, decoder/other learning rates of '
            r'$10^{-5}/10^{-4}$, and a fixed 20k-update cosine schedule. Native '
            r'data membership, image tiling, action/world architecture and losses remain '
            r'unchanged. Checkpoint selection uses physical arm-command error on 200 fixed '
            r'held-out frames; final robot trials remain reserved.' + '\n\n'
            f"At {last['step']:,} additional updates, held-out arm RMSE was "
            f"{last['arm_rmse']:.3f}, compared with {first['arm_rmse']:.3f} at "
            f"{first['step']:,}; gripper error changed from "
            f"{first['gripper_error_percent']:.1f}\\% to {last['gripper_error_percent']:.1f}\\%. "
            r'These ongoing, single-seed measurements do not demonstrate successful '
            r'control or isolate the effect of each recovery intervention.' + '\n'
            r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/F8_recovery_progress.pdf}'
            r'\caption{Registered RGB-recovery validation. Physical command errors use '
            r'1,303 valid actions from the same 200 held-out frames, excluding padding. '
            r'The gray curve is median training loss per 100 updates. Recovery steps are '
            r'additional to the legacy 10k checkpoint. These are offline diagnostics, '
            r'not LIBERO success measurements.}\end{figure}'
        )
        if recovery.get('development'):
            trial = recovery['development'][-1]
            cloud_section += (
                '\n' + r'\begin{samepage}' + f"The {trial['selected_step']:,}-update checkpoint achieved "
                f"{trial['successes']}/{trial['episodes']} development successes. "
                r'This small diagnostic does not establish final performance retention.'
                + r'\end{samepage}' + '\n'
            )
    final_paper = final_measurements_ready(analysis, selection, adaptation)
    results = analysis.get("results", {})
    phase = analysis.get("rollout_phase", "not measured")
    success_scope = {
        "development": "Success values are development diagnostics, not final benchmark results.",
        "final": "Success values use the reserved final initial states; trial counts are recorded in the accompanying results.",
    }.get(phase, "Robot success has not yet been measured in a matched comparison.")
    for key in ["B16", "Q8", "Q4", "S500"]:
        result = results.get(key, {})
        success = result.get("success")
        timing = result.get("timing")
        score = (
            f"{100 * success['rate']:.1f} [{100 * success['ci95'][0]:.1f}, {100 * success['ci95'][1]:.1f}]"
            if success
            else "Pending"
        )
        latency = (
            f"{timing['pipeline']['median_ms']:.1f} / {timing['pipeline']['p95_ms']:.1f}"
            if timing
            else "Pending"
        )
        memory = f"{timing['peak_allocated_bytes'] / 2**30:.2f}" if timing else "Pending"
        size = (
            f"{timing['serialized_weights_bytes'] / 2**30:.2f}"
            if timing and timing.get("serialized_weights_bytes")
            else "Pending"
        )
        rows.append(f"{key} & {score} & {latency} & {memory} & {size} \\\\")
    environment = next(
        (
            e["environment"]
            for e in checks.values()
            if e.get("environment", {}).get("cuda_compute_verified")
        ),
        {},
    )
    hardware = tex(
        environment.get("gpu", "RTX 3090 (host CUDA allocation verified; policy checks pending)")
    )
    date = datetime.now(UTC).strftime("%Y-%m-%d")
    completed_runs = {}
    adaptation_path = ROOT / "studies/evaluation/adaptation/summary.json"
    checkpoint_steps = (
        read_json(adaptation_path).get("checkpoint_steps", {}) if adaptation_path.exists() else {}
    )
    for path in sorted((ROOT / "studies/evaluation/runs").glob("*/run.json")):
        record = read_json(path)
        summary = record.get("summary", {})
        if (
            record.get("status") == "completed"
            and record.get("phase") == "development"
            and summary.get("episodes") == 100
            and set(summary.get("task_ids", [])) == set(range(10))
        ):
            label = record["variant"]
            if label == "S500":
                step = checkpoint_steps.get(record.get("checkpoint_sha256"))
                label += f" (step {step:,})" if step is not None else " (checkpoint unlinked)"
            completed_runs[record["variant"]] = (
                f"{tex(label)}: {summary['successes']}/{summary['episodes']} successful episodes."
            )
    pilot_progress = (
        "Recorded rollout diagnostics: "
        + " ".join(completed_runs.values())
        + " These preliminary records are not a completed matched comparison."
        if completed_runs
        else "Rollout diagnostics are pending."
    )
    figures = []
    captions = {
        "F1_success": "Manipulation success on the recorded rollout phase; intervals describe the fixed task suite.",
        "F2_resources": "Measured deployment resources on the RTX 3090. Bars show median full observation-to-action latency, peak allocated memory, and safetensors weight size; ticks show p95 latency and peak reserved memory. Batch size one; simulator stepping is excluded. S500 remains pending adaptation.",
        "F3_tradeoff": "Measured success--latency trade-off. Marker area represents peak allocated GPU memory.",
        "F4_tasks": "Task-level manipulation success; task numbers follow the committed initial-state manifest.",
        "F5_adaptation": "Adaptation measurements from the continuing training run. Missing development rollouts are marked explicitly; prediction loss is not robot success.",
    }
    if results.get("S500", {}).get("timing"):
        captions["F2_resources"] = captions["F2_resources"].replace(
            " S500 remains pending adaptation.", ""
        )
    for stem, caption in captions.items():
        folder = "adaptation" if stem.startswith("F5") else "analysis"
        path = ROOT / f"studies/evaluation/{folder}/figures/{stem}.pdf"
        if path.exists():
            shutil.copyfile(path, output / "figures" / path.name)
            figures.append(
                r"\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{figures/"
                + path.name
                + r"}\caption{"
                + caption
                + r"}\end{figure}"
            )
    findings = "Matched efficiency measurements are pending."
    if results.get("B16", {}).get("timing"):
        baseline = results["B16"]["timing"]
        comparisons = []
        for variant in ["Q8", "Q4", "S500"]:
            if results.get(variant, {}).get("timing"):
                measured = results[variant]["timing"]
                memory = 100 * (
                    1 - measured["peak_allocated_bytes"] / baseline["peak_allocated_bytes"]
                )
                ratio = measured["pipeline"]["median_ms"] / baseline["pipeline"]["median_ms"]
                comparisons.append(
                    f"{variant}: {abs(memory):.1f}\\% {'lower' if memory >= 0 else 'higher'} "
                    f"peak allocated memory, {ratio:.2f}$\\times$ baseline latency"
                )
        findings = (
            "; ".join(comparisons)
            + (
                ". These are deployment measurements on one GPU."
                if phase == "final"
                else ". These are deployment measurements on one GPU; final manipulation retention is unmeasured."
            )
            if comparisons
            else "Baseline timing is measured; quantized comparisons are still running."
        )
    neural = [
        f"{variant} {result['neural']['median_ms']:.1f}/{result['neural']['p95_ms']:.1f}"
        for variant, result in results.items()
        if "neural" in result
    ]
    if neural:
        findings += (
            " Neural-only median/p95 latencies (ms) are "
            + ", ".join(neural)
            + ". Each neural path first reproduced native outputs on all 100 bank observations "
            "under identical flow-noise seeds."
        )
    paired = []
    for variant in ["Q8", "Q4", "S500"]:
        comparison = results.get(variant, {}).get("success", {}).get("paired_baseline")
        if comparison:
            low, high = comparison["ci95_pp"]
            paired.append(f"{variant} {comparison['difference_pp']:+.1f} [{low:+.1f}, {high:+.1f}]")
    if paired:
        findings += (
            f" {phase.capitalize()} paired success differences from B16 "
            "(percentage points; 95\\% bootstrap intervals) are " + "; ".join(paired) + "."
        )
    evaluated_step = results.get("S500", {}).get("study_training", {}).get("optimizer_steps")
    checkpoint_scope = (
        f"S500 measurements use the checkpoint at step {evaluated_step:,}."
        if evaluated_step is not None
        else "S500 measurements remain pending."
    )
    split_path = ROOT / "studies/evaluation/training_split.json"
    split_text = "Training-split verification is pending."
    if split_path.exists():
        split = read_json(split_path)
        split_text = (
            f"The frozen local dataset split contains {len(split['train_episodes'])} training "
            f"and {len(split['validation_episodes'])} held-out trajectories "
            f"({split['train_frames']:,} and {split['validation_frames']:,} frames). "
            "Hold out the last ten percent of trajectories per task, rounding up. "
            "For checkpoint selection, evaluate twenty evenly spaced held-out frames per task "
            "with fixed flow-noise seeds and sample-weighted mean loss. Preserve the published "
            "baseline normalization statistics across all arms."
        )
    training_progress = "S500 adaptation has not yet produced optimizer-step evidence."
    for metrics_path in sorted((ROOT / "studies/evaluation/training").glob("*/metrics.jsonl")):
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        steps = [r["step"] for r in metrics if r["phase"] == "training"]
        if steps:
            training_progress = (
                f"The latest S500 training segment has recorded successful optimizer updates "
                f"through step {max(steps):,}. Adaptation and checkpoint selection are incomplete."
            )
    adaptation_path = ROOT / "studies/evaluation/adaptation/summary.json"
    if adaptation_path.exists():
        adaptation = read_json(adaptation_path)
        if adaptation["validation"]:
            latest = adaptation["validation"][-1]
            training_progress += (
                f" At step {latest['step']:,}, held-out total loss was {latest['loss']:.3f} "
                f"over {latest['samples']} fixed samples."
            )
    selection_path = ROOT / "studies/evaluation/selection.json"
    if selection_path.exists():
        selection = read_json(selection_path)
        training_progress = (
            f"The frozen adaptation budget is complete. Held-out loss selected step "
            f"{selection['selected_step']:,} (loss {selection['validation']['loss']:.3f}); "
            "simulator scores were not used for selection."
        )
    source = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,graphicx,booktabs,array,caption,xcolor,hyperref,placeins}
\definecolor{ink}{HTML}{253547}
\hypersetup{colorlinks=true,linkcolor=ink,urlcolor=ink}
\captionsetup{font=small,labelfont=bf}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\tabcolsep}{5pt}
\title{\vspace{-15mm}\textbf{Efficient VLA-JEPA for Robotic Manipulation}\\[3pt]\large CSI 5341: methodology and evaluation progress}
\author{Noah Sprenger \qquad Fernando Nogueira}
\date{DATESTAMP \quad | \quad Working report: measurements in progress}
\begin{document}
\maketitle
\vspace{-7mm}
\begin{abstract}
We study whether reduced-precision inference and a smaller vision--language backbone lower VLA-JEPA's deployment cost while retaining manipulation success. We compare a pretrained BF16 policy, two decoder-quantized variants, and a SmolVLM2-500M variant requiring reduced fine-tuning. The protocol combines matched LIBERO-Spatial trials, inference measurements, and a training-budget study. This working report presents measured deployment costs where available; manipulation comparisons and backbone adaptation remain in progress.
\end{abstract}

\section{Study design}
The baseline is \texttt{lerobot/VLA-JEPA-LIBERO}~\cite{vla}. Q8 and Q4 quantize its trained Qwen language-decoder linear layers using LLM.int8~\cite{int8} (outlier threshold 6) and NF4 with double quantization~\cite{nf4}, respectively; the vision tower, embeddings, action head, and world model remain BF16. Quantization happens \emph{after} policy checkpoint restoration. S500 replaces the backbone with SmolVLM2-500M-Video-Instruct~\cite{smol} and projects its decoder representation into the existing conditioning interface. The adaptation recipe freezes SmolVLM and the V-JEPA~2 encoder~\cite{jepa}, training the adapter and compatible action/world predictors. Qwen is evaluation-only. Dream-RSI is deferred.

\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{figures/architecture.pdf}
\caption{Architecture and comparison arms. Solid arrows denote the inference path; dashed connections denote training supervision. The action head also receives robot state. The schematic omits the action-training loss for clarity. All configurations use identical observation/action settings and world-model loading policy.}
\end{figure}

\section{Evaluation methodology}
\textbf{E1: manipulation.} We use all ten LIBERO-Spatial tasks~\cite{libero}, batch size one, matched initial states and per-episode random seeds. Development uses ten original initial-state vectors per task, excluded by exact hash from the official pruned states. Final evaluation uses all fifty official pruned states per task: 500 trials per configuration. The committed manifest records state-file and state-vector hashes. Development scores are diagnostic and are not pooled with final benchmark scores.

We estimate task-macro-average success and task-level rates with 95\% intervals and task-stratified paired bootstrap differences from B16. Intervals describe uncertainty on this fixed task suite, not generalization to unseen tasks. System failures remain separate from valid unsuccessful robot trials. Final trials do not enter checkpoint selection.

\textbf{E2: efficiency.} On one GPU (HARDWARE), we time a fixed observation/instruction bank with twenty warm-up predictions and 500 measured action-chunk predictions per repetition (three repetitions). CUDA synchronization brackets each measurement. Outcomes are median/p95 neural-only and observation-to-action latency, peak allocated/reserved GPU memory, parameter count, and serialized weight size. Timings exclude simulator execution and buffered-action retrieval; neural-only timing additionally excludes tokenization and image preprocessing. Memory reduction is not assumed to imply a speedup.

\textbf{E3: adaptation.} We split demonstrations by whole trajectories, reserving 10\% for validation. The completed 500-step pilot established execution feasibility and resource cost. The frozen budget is 10,000 optimizer steps (seed 42), with selection checkpoints at 1,500, 5,000, and 10,000 steps from one continuing run. Selection minimizes sample-weighted held-out total loss; ties favor fewer steps. The pilot projects approximately ten GPU-hours including validation and checkpoint overhead. We use one training seed, so training-seed uncertainty is not estimated. The published baseline and S500 have different training histories; their comparison concerns practical deployment under limited adaptation, not the isolated causal effect of backbone size.

\section{Progress and results}
SPLITTEXT
\begin{table}[ht]
\centering\small
\begin{tabular}{@{}lllll@{}}
\toprule
Variant & Success [95\% CI] (\%) & Median / p95 (ms) & VRAM (GiB) & Weights (GiB) \\
\midrule
RESULTROWS
\bottomrule
\end{tabular}
\caption{T1: current measured evidence. SUCCESSSCOPE Latency includes outer observation preparation and action postprocessing; VRAM is peak framework allocation. Weight size uses a consistent safetensors encoding with shared tensors deduplicated. Missing results remain pending. No adaptation is applied to B16/Q8/Q4 in this study. CHECKPOINTSCOPE}
\end{table}

PILOTPROGRESS

TRAININGPROGRESS

\textbf{Measured results.} FINDINGS

RESULTFIGURES

CLOUDSECTION

\FloatBarrier
\section{Decision rule and limitations}
The prespecified engineering target is at least 25\% lower peak inference memory with at most a five-percentage-point success drop from B16. Estimates are reported with uncertainty; an inconclusive interval does not establish performance preservation. Retaining the world model consistently prevents module removal from masquerading as a quantization benefit. A single GPU, limited adaptation budget, and differing pretraining histories limit broader conclusions.

The inherited world-model loss uses bidirectional video embeddings; it does not measure causal future prediction. Deployment receives current observations only. Validation trajectories are held out from this adaptation, not necessarily from published models' earlier training. Checkpoint loading audits all mismatches; documented unused tensors and verified tied aliases are excluded.

Camera observations and outer policy preprocessing are matched, but each backbone retains its native image/token processor. A CPU probe confirms that SmolVLM expands a 224-pixel image into seventeen 512-pixel tiles. Thus the backbone comparison includes different visual token workloads; parameter count alone does not predict latency. Native pixel scaling was checked independently. Neural-only timing is separated from processor-inclusive timing.

\textbf{Summary.} This study measures decoder quantization within a fixed pretrained policy and backbone replacement under limited adaptation. Deployment savings and manipulation retention are separate outcomes; conclusions about successful robot deployment require both.

\begin{thebibliography}{6}\small
\bibitem{vla} Sun et al. VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model. arXiv:2602.10098, 2026.
\bibitem{libero} Liu et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. arXiv:2306.03310, 2023.
\bibitem{smol} Marafioti et al. SmolVLM: Redefining Small and Efficient Multimodal Models. arXiv:2504.05299, 2025.
\bibitem{jepa} Assran et al. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv:2506.09985, 2025.
\bibitem{int8} Dettmers et al. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. \href{https://arxiv.org/abs/2208.07339}{arXiv:2208.07339}, 2022.
\bibitem{nf4} Dettmers et al. QLoRA: Efficient Finetuning of Quantized LLMs. \href{https://arxiv.org/abs/2305.14314}{arXiv:2305.14314}, 2023.
\end{thebibliography}
\end{document}
"""
    source = (
        source.replace("DATESTAMP", date)
        .replace("HARDWARE", hardware)
        .replace("RESULTROWS", "\n".join(rows))
        .replace("PILOTPROGRESS", pilot_progress)
        .replace("SPLITTEXT", split_text)
        .replace("FINDINGS", findings)
        .replace("RESULTFIGURES", "\n".join(figures))
        .replace("CLOUDSECTION", cloud_section)
        .replace("TRAININGPROGRESS", training_progress)
        .replace("SUCCESSSCOPE", success_scope)
        .replace("CHECKPOINTSCOPE", checkpoint_scope)
    )
    if final_paper:
        scores = ", ".join(
            f"{variant} {100 * results[variant]['success']['rate']:.1f}\\%"
            for variant in ["B16", "Q8", "Q4", "S500"]
        )
        source = (
            source.replace("methodology and evaluation progress", "efficiency evaluation")
            .replace(r" \quad | \quad Working report: measurements in progress", "")
            .replace(
                "This working report presents measured deployment costs where available; "
                "manipulation comparisons and backbone adaptation remain in progress.",
                "The completed protocol evaluates 500 reserved trials per configuration. "
                f"Success rates are {scores}. Efficiency and paired uncertainty estimates "
                "characterize the resulting deployment trade-offs.",
            )
            .replace(pilot_progress, "")
            .replace("T1: current measured evidence.", "T1: final matched measurements.")
        )
    if cloud:
        source = source.replace('Dream-RSI is deferred.', 'The optional discovery confirmation is reported separately below.')
    (output / "report.tex").write_text(source)
    for _ in range(2):
        completed = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
            cwd=output,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout[-4000:])
    print(
        json.dumps(
            {
                "report": str((output / "report.pdf").relative_to(ROOT)),
                "smoke_checks": {k: v.get("outcome", v.get("status")) for k, v in checks.items()},
                "final_measurements_ready": final_paper,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "studies/evaluation/report")
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
