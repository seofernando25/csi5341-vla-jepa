# Independent audit of the CSI 5341 experiments

The reviewers found accurate recorded measurements, real engineering defects, and partial recovery. They did **not** find evidence of a competitive SmolVLM policy, a demonstrated n0008 architecture benefit, or a reproduction of the paper's training. The central problem was experimental design and prioritization: a narrow loss search on defective inputs was followed by several coupled recovery interventions, without completing the comparison needed to establish which intervention helped.

This is the original implementation agent's synthesis of three fresh-context AI reviews, not an external human audit. Their unedited reports are linked below. Research remains paused; no experiments or rentals were launched.

## What happened

| Phase | Verified outcome | What it establishes |
| --- | --- | --- |
| Original class comparison | B16 475/500; Q8 474/500; Q4 479/500; S500 11/500 final successes | Quantized versions retained high measured success. This particular SmolVLM transplant failed. |
| GPU Dream-RSI | 23 successful screens and six promotions; n0008 promotion loss 0.2762 versus root 0.3197 | Better prediction loss among the promoted subset, on corrupted image inputs and 60 validation frames. |
| Longer n0008 confirmation | 0/10 development successes; planned matching root arm was not completed | No demonstrated architecture benefit or successful control. |
| Corrected-input recovery | Four decoder layers adapted for 20k additional updates; 31/100 versus B16 82/100 | Partial functionality with a large remaining control gap, on a development cohort. |
| Full decoder and query follow-up | 10k updates; held-out error selected 9.5k; 5/10 versus parent's 4/10 | One gained diagnostic state, insufficient evidence of reliable improvement. |
| Closeout | Final native checkpoints retained and independently verified; rental deleted; target unmet | Durable results and resume state, not completion of the useful-model goal. |

Raw counts were independently recomputed. These protocols and cohorts must not be pooled. [P04–P08](protocol.json), [SCI03–SCI06](science.json).

## Why loss did not establish successful control

**The GPU optimization introduced a real input defect.** Structured processor arguments silently overrode the intended no-rescale setting. Images already in 0–1 were divided by 255 again; the same-frame contrast diagnostic confirms a 255-fold reduction. The GPU RSI ranking therefore describes a degraded visual pipeline. The original PIL-based S500 failure predates this defect, so the defect cannot explain that earlier result. [P01/P11](protocol.json), [SCI02](science.json).

**RSI optimized a limited surrogate.** Screens used 500 updates, promotions 1,500, one seed and 60 consecutive validation frames. Flow-noise evaluation was not independently seeded. Selection optimized loss, not success, latency and VRAM jointly. Only six early candidates received promotions; n0008 won that subset. The longer root comparison remained incomplete. [P02/P04–P06](protocol.json).

**The transplant differed materially from the reference.** B16 restored a published LIBERO policy. SmolVLM used a generic backbone and compatible pretrained heads, initially freezing the backbone and new query embeddings. Exposure and adaptation differed from the paper. This does not isolate model size. The [paper appendix](https://arxiv.org/html/2602.10098v2#A2) and [current author configuration](https://github.com/ginwind/VLA-JEPA/blob/main/scripts/configs/vlajepa_libero_ft.yaml) also differ; the exact converted-checkpoint recipe remains uncertain. [SCI01](science.json).

**Recovery changed several things together.** Processing, trainability, optimizer, batch, schedule and exposure changed, followed by queries and broader decoder adaptation. Gains cannot be assigned to one change. Rotary-frequency rounding is a real defect, but its repair left paired endpoint success at 4/10 in both modes. Parameter casting raised arm MSE by 0.43%; that offline result does not explain the control gap. [SCI04–SCI08](science.json), [P07/P09](protocol.json).

The remaining cause is unresolved. Capacity, exposure and conditioning mismatch are hypotheses. More steps alone are not a proved solution. The inherited world loss uses bidirectional clip embeddings, not a demonstrated causal forecast; the report already discloses this. [SCI07–SCI09](science.json).

## Money and operational failures

The audited closeout recorded **USD11.27 total**: USD2.17 original confirmation and USD9.10 recovery, based on credit differences rather than an invoice. Nineteen hours was elapsed research time; the final rental ran 16 hours 56 minutes. Sampled training showed high GPU use. [Operations review](operations.json).

An hourly cleanup deleted an earlier rental during training, losing post-5k progress that was repeated. An undersized benchmark timeout also forced repeated work. These were operational failures. Final deletion was about 47 minutes after the requested deadline; exports and checks explain part of the delay. Exact avoidable cost is unknown.

All **26 final native files** were independently hash-verified, including weights and optimizer state; counters and scheduler also match. This establishes byte integrity, not a fresh resume reproduction. Auditors assessed provider deletion through retained receipts.

After the independent reviews, the original implementation agent made a [read-only provider check](root_provider_followup.json) at 18:42 UTC: all six known project rentals were absent, and the updated credit difference was **USD11.28**, still below USD14. The small change from the earlier snapshot is not itemized. This follow-up corroborates current cleanup, not historical deletion replies or an invoice.

## Reporting and reproducibility problems

The latest report is largely candid. The evaluation README is stale, and methodology retains obsolete running/pending paragraphs. Confirmation text mixes planned 100-episode work with completed ten-episode checks; dashboard timing groups omit processing and numerical controls. Structured RSI telemetry truncates at step 450 on abbreviated counters, although final scores remain correct. The automatic confirmation command selects n0001 from screens; the actual plan chose n0008 from promotions. [P12–P16](protocol.json), [OP8](operations.json).

These are findings, not corrections applied during this audit. Historical reports, protocols and measurements were preserved.

## Conclusions and next decision

The money produced useful debugging and negative evidence, **not the requested good model**. Prioritization was weak: input checks and a meaningful matched root comparison should have preceded extensive search and recovery.

Use quantization as the main class result. Report the failed transplant honestly, bound optional RSI/recovery to an appendix, and reconcile status labels. Keep paid work paused. Further research needs a separately authorized, discriminating comparison.

## Original reviewer reports

- [Scientific and original-paper review](science.json): 11 findings, primary paper/author-code comparison, raw success and validation checks.
- [RSI and evaluation protocol review](protocol.json): 16 findings, all 204 journal events, 31 raw loss checks, ten rollout cohorts and timing checks.
- [Operations and spending review](operations.json): lifecycle, failures, costs, deadlines, retention and proof levels.
- [Scope and independence record](intake.json).

Log agreement establishes consistency, not absolute proof against fabrication. Missing comparisons and deleted evidence remain limitations; proof levels are explicit in the individual reports.
