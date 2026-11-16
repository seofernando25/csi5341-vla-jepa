/* Chapter 05 — what the loop changed in its own policy: the kept rewrite, the
 * measured replay gain, and every variant it was chosen from. */

import { changedModel, cycleOf, diffLines } from "./model.js";

export const id = "changed";
export const number = "05";
export const title = "What changed";
export const dek =
  "After each cycle the loop rewrites the policy that chose its probes, keeps the rewrite only if replay value rises, and starts the next cycle with it.";

const CONTEXT = 2;
const MAX_ROWS = 64;
const MAX_FILE_LINES = 500;

const textOf = (value) =>
  typeof value === "string" ? value : value && typeof value.text === "string" ? value.text : null;
const pad = (step) => String(step).padStart(2, "0");
const tone = (value) => (value == null || value === 0 ? "" : value > 0 ? "ok" : "err");

function stepOfSlot(variants, slot) {
  const hit = slot ? variants.find((variant) => variant.slot === slot) : null;
  return hit && hit.step != null ? hit.step : null;
}

/* The state carries the same verdicts when the dream payload is absent. */
function fromSteps(steps) {
  return (Array.isArray(steps) ? steps : []).map((step) => ({
    slot: step.slot || null, step: step.step ?? null, status: step.status || null,
    meanV: typeof step.mean_v === "number" ? step.mean_v : null,
    deltaVsBest: typeof step.delta_vs_best === "number" ? step.delta_vs_best : null,
  }));
}

/* Latest finished cycle that selected a policy; a running cycle is pending. */
function targetCycle(state) {
  const cycles = Array.isArray(state?.server?.data?.cycles) ? state.server.data.cycles : [];
  const chosen = cycles.filter((c) => c?.policy_dev?.status === "done" && c?.policy_dev?.selected_step != null);
  return chosen[chosen.length - 1] || cycleOf(state);
}

function facts(state, payload) {
  const cycle = targetCycle(state);
  const dev = cycle?.policy_dev || {};
  const model = changedModel(state, payload);
  const variants = model.variants.length ? model.variants : fromSteps(dev.steps);
  const selectedStep = dev.selected_step ?? stepOfSlot(variants, model.selected);
  const selectedVariant = variants.find((variant) => variant.step === selectedStep) || null;
  const incumbentVariant = variants.find((variant) => variant.status === "incumbent") || null;
  const incumbentV = model.incumbent?.mean_v ?? dev.incumbent_v ?? null;
  const selectedV = selectedVariant?.meanV ?? dev.selected_v ?? null;
  return {
    cycle, dev, model, variants, selectedStep,
    incumbentStep: incumbentVariant?.step ?? model.incumbent?.step ?? null,
    incumbentV,
    selectedV,
    delta: model.delta ?? (incumbentV != null && selectedV != null ? selectedV - incumbentV : null),
    heldout: model.heldoutDelta ?? dev.heldout_delta ?? null,
    status: dev.status || "pending",
    chosen: selectedStep != null && (dev.status === "done" || model.selected != null),
  };
}

/* Diff: changed hunks with two lines of context, numbered on both sides. */
function numbered(lines) {
  let before = 0;
  let after = 0;
  return lines.map((line) => {
    if (line.kind !== "add") before += 1;
    if (line.kind !== "del") after += 1;
    return { ...line, before: line.kind === "add" ? null : before, after: line.kind === "del" ? null : after };
  });
}

function windowed(lines, context = CONTEXT, max = MAX_ROWS) {
  const marked = new Set();
  lines.forEach((line, index) => {
    if (line.kind === "same") return;
    const from = Math.max(0, index - context);
    const to = Math.min(lines.length - 1, index + context);
    for (let i = from; i <= to; i += 1) marked.add(i);
  });
  const rows = [];
  let index = 0;
  while (index < lines.length && rows.length < max) {
    if (marked.has(index)) {
      rows.push({ type: "line", ...lines[index] });
      index += 1;
      continue;
    }
    let end = index;
    while (end < lines.length && !marked.has(end)) end += 1;
    rows.push({ type: "skip", count: end - index });
    index = end;
  }
  while (rows.length && rows[0].type === "skip") rows.shift();
  while (rows.length && rows[rows.length - 1].type === "skip") rows.pop();
  return { rows, hidden: lines.slice(index).filter((line) => line.kind !== "same").length };
}

function diffBody(before, after, emptyText, ctx) {
  const { el } = ctx.dom;
  const lines = diffLines(before, after);
  const adds = lines.filter((line) => line.kind === "add").length;
  const dels = lines.filter((line) => line.kind === "del").length;
  const body = el("div", { class: "diff", attrs: { "aria-label": "Policy diff" } });
  if (!adds && !dels) {
    body.append(el("p", { class: "changed-missing", text: emptyText }));
    return { body, adds, dels };
  }
  const { rows, hidden } = windowed(numbered(lines));
  for (const row of rows) {
    if (row.type === "skip") {
      const unit = row.count === 1 ? "line" : "lines";
      body.append(el("div", { class: "diff-skip", text: `⋯ ${row.count} unchanged ${unit}` }));
      continue;
    }
    const sign = row.kind === "add" ? "+" : row.kind === "del" ? "−" : "";
    body.append(el("div", { class: "diff-line", dataset: { kind: row.kind } }, [
      el("span", { class: "diff-no", text: row.before ?? "" }),
      el("span", { class: "diff-no", text: row.after ?? "" }),
      el("span", { class: "diff-sign", text: sign }),
      el("code", { class: "diff-text", text: row.text || " " }),
    ]));
  }
  if (hidden) {
    const unit = hidden === 1 ? "line" : "lines";
    body.append(el("div", { class: "diff-more", text: `+${hidden} more changed ${unit}` }));
  }
  return { body, adds, dels };
}

function stats(f, ctx) {
  const { el } = ctx.dom;
  const fmt = ctx.fmt;
  const one = (value, label) =>
    el("div", { class: "stat", dataset: { tone: tone(value) } }, [
      el("span", { class: "stat-value", text: fmt.signed(value, 3) }),
      el("span", { class: "stat-label", text: label }),
    ]);
  return el("div", { class: "stat-row changed-stats" }, [
    one(f.delta, `In-sample V · ${fmt.num(f.incumbentV, 3)} → ${fmt.num(f.selectedV, 3)}`),
    one(f.heldout, "Held-out V, selected − incumbent"),
  ]);
}

function diffFigure(f, data, ctx) {
  const { el } = ctx.dom;
  const figure = el("figure", { class: "figure changed-diff" });
  const step = f.selectedStep;
  const pair = data.beforeLabel ? `${data.beforeLabel} → step_${pad(step)} · selected` : `step_${pad(step)} · selected`;
  const head = el("div", { class: "figure-head" }, [el("span", { text: pair })]);
  figure.append(head);
  if (!data.read) {
    figure.append(el("p", { class: "changed-missing", text: "Reading the policy source." }));
    return figure;
  }
  if (data.before == null || data.after == null) {
    figure.append(el("p", { class: "changed-missing", text: "policy source not recorded" }));
    return figure;
  }
  const incumbent = data.beforeKind === "incumbent";
  const empty = incumbent
    ? "The incumbent and the selected policy are identical line for line."
    : "The recorded seed matches the selected policy line for line.";
  const { body, adds, dels } = diffBody(data.before, data.after, empty, ctx);
  head.append(el("span", { class: "diff-counts", text: `+${adds} −${dels}` }));
  figure.append(body);
  const capped = [data.before, data.after].some((text) => String(text).split("\n").length > MAX_FILE_LINES);
  const href = `#/policy/${encodeURIComponent(f.cycle.cycle_id)}/${step}/diff`;
  figure.append(el("p", { class: "figure-cap" }, [
    incumbent ? "The incumbent's policy against the selected rewrite. " : "The file the agent started from against its rewrite. ",
    "V is replay value, not task success. ",
    capped ? `The diff covers the first ${MAX_FILE_LINES} lines of each file. ` : "",
    el("a", { href, text: "Open in the Workbench" }),
  ]));
  return figure;
}

function variantsTable(f, ctx) {
  const { el } = ctx.dom;
  const fmt = ctx.fmt;
  if (!f.variants.length) return null;
  const head = el("thead", {}, [el("tr", {}, [
    el("th", { attrs: { scope: "col" }, text: "Slot" }),
    el("th", { attrs: { scope: "col" }, text: "Status" }),
    el("th", { class: "num", attrs: { scope: "col" }, text: "Mean V" }),
    el("th", { class: "num", attrs: { scope: "col" }, text: "Δ at step" }),
  ])]);
  const body = el("tbody");
  for (const variant of f.variants) {
    body.append(el("tr", { class: variant.step === f.selectedStep ? "is-selected" : "" }, [
      el("td", { class: "slot", text: variant.slot || "—" }),
      el("td", { text: fmt.verdictLabel(variant.status) }),
      el("td", { class: "num", text: fmt.num(variant.meanV, 3) }),
      el("td", { class: "num", text: fmt.signed(variant.deltaVsBest, 3) }),
    ]));
  }
  return el("figure", { class: "figure changed-variants" }, [
    el("div", { class: "figure-head" }, [el("span", { text: "Variants" }), el("span", { text: "in-sample V" })]),
    el("table", { class: "changed-table" }, [head, body]),
    el("p", { class: "figure-cap", text: "Delta against the best V when the step was replayed. Selection is the highest mean V; ties go to the lower step." }),
  ]);
}

function meta(f, ctx) {
  const bits = [f.cycle.cycle_id, `step ${pad(f.selectedStep)}`];
  if (f.model.selected) bits.push(`selected ${f.model.selected}`);
  return ctx.dom.el("p", { class: "changed-meta", text: bits.join(" · ") });
}

function pending(f, ctx) {
  const { el } = ctx.dom;
  let line = "No policy selected yet. The incumbent is still the policy.";
  if (f.status === "done") line = "No rewrite was selected. The incumbent is still the policy.";
  if (f.status === "skipped") line = "Policy development was skipped. The incumbent is still the policy.";
  const status = f.cycle
    ? `Cycle ${f.cycle.cycle_id} is ${f.cycle.status || "unrecorded"}. Policy development is ${f.status}.`
    : "Waiting for the run.";
  return el("div", { class: "changed-pending" }, [    el("p", { class: "changed-state", text: line }),
    el("p", { class: "changed-note", text: status }),
  ]);
}

export function mount(root, ctx) {
  const { el, clear } = ctx.dom;
  let state = null, signature = null, token = 0;
  let data = { payload: null, before: null, beforeLabel: null, beforeKind: null, after: null, read: false };

  function draw() {
    clear(root);
    root.append(el("p", { class: "chapter-kicker" }, [el("span", { class: "chapter-number" }, number)]));
    root.append(el("h2", { class: "chapter-title", text: title }));
    root.append(el("p", { class: "chapter-dek", text: dek }));
    const f = facts(state, data.payload);
    if (!f.cycle) {
      root.append(el("p", { class: "changed-state", text: "Waiting for the run." }));
      return;
    }
    if (!f.chosen) {
      root.append(pending(f, ctx));
      return;
    }
    root.append(stats(f, ctx));
    root.append(el("p", { class: "changed-note", text: "V is replay value on frozen trees. Held-out worlds are reported, never used to select." }));
    root.append(meta(f, ctx));
    root.append(diffFigure(f, data, ctx));
    const table = variantsTable(f, ctx);
    if (table) root.append(table);
  }

  async function load(cycle, step) {
    const mine = ++token;
    const payload = await ctx.story.dream(cycle.cycle_id);
    const incumbentStep = facts(state, payload).incumbentStep;
    let before = null, beforeLabel = null, beforeKind = null, after = null;
    if (step != null) {
      const source = `policy:${cycle.cycle_id}:${step}`;
      const incumbentSource = incumbentStep != null ? `policy:${cycle.cycle_id}:${incumbentStep}` : null;
      after = textOf(await ctx.story.artifact(source, "policy"));
      if (incumbentSource) {
        before = textOf(await ctx.story.artifact(incumbentSource, "policy"));
        if (before != null) {
          beforeKind = "incumbent";
          beforeLabel = `step_${pad(incumbentStep)} · incumbent`;
        }
        if (before == null) {
          before = textOf(await ctx.story.artifact(incumbentSource, "seed"));
          if (before != null) {
            beforeKind = "incumbent";
            beforeLabel = `step_${pad(incumbentStep)} · incumbent`;
          }
        }
      }
      if (before == null && incumbentStep == null) {
        before = textOf(await ctx.story.artifact(source, "seed"));
        beforeKind = "seed";
        beforeLabel = `step_${pad(step)} · seed`;
      }
    }
    if (mine !== token) return;
    data = { payload, before, beforeLabel, beforeKind, after, read: true };
    draw();
  }

  return {
    update(next) {
      state = next;
      const cycle = targetCycle(state);
      const dev = cycle?.policy_dev || {};
      const nextSignature = [cycle?.cycle_id, cycle?.status, dev.status, dev.selected_step].join("|");
      if (nextSignature === signature) return;
      signature = nextSignature;
      data = { payload: null, before: null, beforeLabel: null, beforeKind: null, after: null, read: false };
      draw();
      if (cycle) load(cycle, dev.selected_step);
    },
    destroy() {
      token += 1;
      clear(root);
    },
  };
}
