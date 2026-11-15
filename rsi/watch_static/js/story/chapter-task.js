/* Chapter 01 — The task: what the run does, how big the policy is, what counts.
 * Reads only baseline() and cycleOf() from the story model; nodes come from the
 * state because the best probe can sit in an earlier cycle than the current one. */

import { el, clear } from "../dom.js";
import { baseline, cycleOf } from "./model.js";

export const id = "task";
export const number = "01";
export const title = "The task";
export const dek =
  "An agent compresses the policy; LIBERO rollouts decide whether it still works, and the score is the memory it removed.";

const NB = "\u00a0";

export function mount(root, ctx) {
  root.classList.add("task-root");
  const stats = el("div", { class: "stat-row task-stats" });
  const figure = el("figure", { class: "figure task-figure" });
  const counts = el("p", { class: "chapter-body task-counts" });
  const foot = el("p", { class: "task-foot" });
  root.append(
    el("p", { class: "chapter-kicker" }, [el("span", { class: "chapter-number" }, number)]),
    el("h2", { class: "chapter-title" }, title),
    el("p", { class: "chapter-dek" }, dek),
    stats,
    figure,
    counts,
    foot,
  );

  let state = null;
  let sig = "";

  function render() {
    const data = state?.server?.data || null;
    if (!data) {
      const line =
        state?.server?.status === "error" ? "The run state is unavailable." : "Reading the run.";
      if (sig === line) return;
      sig = line;
      clear(stats);
      clear(figure);
      clear(counts);
      clear(foot);
      figure.append(el("p", { class: "task-state" }, line));
      return;
    }
    const model = read(state);
    const next = signature(model);
    if (next === sig) return;
    sig = next;
    paint(model);
  }

  function paint(model) {
    clear(stats);
    clear(figure);
    clear(counts);
    clear(foot);

    stats.append(
      stat("smoke gate", pair(model.gate, model.trials)),
      stat("probe budget", model.budget != null ? String(model.budget) : "—"),
      stat("cycles", cyclesText(model)),
    );

    figure.append(
      el("div", { class: "figure-head" }, [el("span", {}, "Policy size")]),
    );

    const readout = el("div", { class: "task-readout" });
    if (model.baseBytes != null) {
      readout.append(
        el("span", { class: "task-from" }, amount(gb(model.baseBytes))),
        el("span", { class: "task-arrow", aria: { hidden: "true" } }, "→"),
      );
    }
    readout.append(
      model.bestMb != null
        ? el("span", { class: "task-best" }, amount(size(model.bestMb)))
        : el("span", { class: "task-empty" }, model.hasProbe ? "size not recorded" : "no probe scored"),
    );
    if (model.reduction != null) {
      readout.append(
        el("span", { class: "task-delta" }, [
          el("span", { class: "u-num" }, `−${ctx.fmt.pct(model.reduction).replace(/\s*%$/, "")}`),
          el("span", { class: "u" }, "%"),
        ]),
      );
    }
    figure.append(readout);

    const fill =
      model.baseBytes != null && model.bestMb != null
        ? clamp((model.bestMb * 1e6) / model.baseBytes, 0, 1)
        : null;
    figure.append(
      el(
        "div",
        {
          class: "task-track",
          dataset: { state: model.baseBytes == null ? "unknown" : fill == null ? "empty" : "filled" },
          role: "img",
          aria: { label: trackLabel(model, fill) },
        },
        fill != null
          ? [el("span", { class: "task-fill", style: { width: `${(fill * 100).toFixed(2)}%` } })]
          : [],
      ),
    );

    if (model.baseBytes != null) {
      figure.append(
        el("div", { class: "task-scale" }, [
          el("span", {}, "0"),
          el("span", {}, gbText(model.baseBytes)),
        ]),
      );
    } else {
      figure.append(el("p", { class: "task-state" }, "Baseline size not recorded."));
    }

    counts.append(
      "A probe scores only if it clears ",
      gatePhrase(model.gate, model.trials),
      " smoke trials; otherwise it scores zero. The best probe of a cycle is then tested against ",
      gatePhrase(model.devGate, model.devTrials),
      " development trials.",
    );

    if (model.cell && model.cycleId) {
      const link = el("button", { class: "task-link", type: "button" }, [
        "Open the evaluation in the Workbench",
        el("span", { class: "task-link-arrow", aria: { hidden: "true" } }, "→"),
      ]);
      link.addEventListener("click", () => {
        ctx.story?.open(`node:${model.cycleId}:${model.cell}`, "evaluation");
      });
      foot.append(link);
    }
  }

  function stat(label, value, note, text = false) {
    return el("div", { class: "stat", dataset: text ? { text: "true" } : null }, [
      el("div", { class: "stat-value" }, value),
      el("div", { class: "stat-label" }, note ? `${label}${NB}·${NB}${note}` : label),
    ]);
  }

  return {
    update(next) {
      state = next;
      render();
    },
    destroy() {
      state = null;
    },
  };
}

/* --- reads --------------------------------------------------------------- */

function read(state) {
  const base = baseline(state);
  const cycles = state?.server?.data?.cycles || [];
  const best = bestScored(cycles);
  const loop = state?.server?.data?.loop || {};
  const current = cycleOf(state);
  const index = current ? cycles.findIndex((cycle) => cycle?.cycle_id === current.cycle_id) : -1;
  const cycleNo = Number.isFinite(loop.iteration) ? loop.iteration + 1 : index >= 0 ? index + 1 : null;
  return {
    baseBytes: base.bytes,
    hasProbe: Boolean(best),
    bestMb: best?.node?.resident_mb ?? null,
    reduction: best?.node?.size_reduction ?? null,
    cycleId: best?.cycle?.cycle_id ?? null,
    cell: best?.node?.cell_id ?? null,
    gate: base.gate,
    trials: base.trials,
    devGate: base.devGate,
    devTrials: base.devTrials,
    budget: base.budget,
    cycleNo,
    cyclesTarget: base.cyclesTarget,
  };
}

/* The best scored node of the whole run, not only the current cycle. */
function bestScored(cycles) {
  let best = null;
  for (const cycle of cycles) {
    for (const node of cycle?.nodes || []) {
      if (node?.cell_id === "root" || node?.status !== "done" || node?.score == null) continue;
      if (!best || node.score > best.node.score) best = { cycle, node };
    }
  }
  return best;
}

/* --- formatting ---------------------------------------------------------- */

function gb(bytes) {
  const value = Number(bytes) / 1e9;
  return { value: value >= 10 ? value.toFixed(1) : value.toFixed(2), unit: "GB" };
}

function size(mb) {
  const value = Number(mb);
  if (value >= 1000) return { value: (value / 1000).toFixed(2), unit: "GB" };
  return { value: String(Math.round(value)), unit: "MB" };
}

function gbText(bytes) {
  const parts = gb(bytes);
  return `${parts.value} ${parts.unit}`;
}

function sizeText(mb) {
  const parts = size(mb);
  return `${parts.value} ${parts.unit}`;
}

/* Value and unit as separate spans: mono type gives a full cell to a space. */
function amount(parts) {
  return [el("span", { class: "u-num" }, parts.value), el("span", { class: "u" }, parts.unit)];
}

function pair(gate, trials) {
  if (gate == null || trials == null) return "—";
  return `${gate} / ${trials}`;
}


function cyclesText(model) {
  if (model.cycleNo == null && model.cyclesTarget == null) return "—";
  const done = model.cycleNo == null ? "—" : String(model.cycleNo);
  return model.cyclesTarget == null ? done : `${done} of ${model.cyclesTarget}`;
}

function gatePhrase(gate, trials) {
  if (gate == null || trials == null) return "the";
  return `${gate} of ${trials}`;
}

function trackLabel(model, fill) {
  const base = model.baseBytes != null ? gbText(model.baseBytes) : "not recorded";
  if (!model.hasProbe) return `Baseline ${base}; no probe scored.`;
  if (model.bestMb == null) return `Baseline ${base}; best probe size not recorded.`;
  const pct = fill == null ? "" : `, ${(fill * 100).toFixed(1)}${NB}% of baseline`;
  return `Baseline ${base}; best probe ${sizeText(model.bestMb)}${pct}.`;
}

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function signature(model) {
  return [
    model.baseBytes,
    model.bestMb,
    model.reduction,
    model.cycleId,
    model.cell,
    model.gate,
    model.trials,
    model.devGate,
    model.devTrials,
    model.budget,
    model.cycleNo,
    model.cyclesTarget,
  ].join("|");
}
