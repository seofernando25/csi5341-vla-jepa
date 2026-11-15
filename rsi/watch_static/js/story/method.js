/* Chapter 06 — the loop in order, and the documented deviations from the paper.
 * Static content: every row points at the place the thing is visible, or says so. */

export const id = "method";
export const number = "06";
export const title = "Method";
export const dek =
  "The loop the site is watching, in the order it runs, and where it departs from the paper.";

const LOOP = [
  { label: "grid_plan", text: "Read the static grid plan from protocol", where: "Tree", href: "#/story/tree" },
  { label: "solve", text: "Policy picks slots within the probe budget", where: "Probe", href: "#/story/probe" },
  { label: "fork", text: "Fork the parent snapshot; one agent writes", where: "Probe", href: "#/story/probe" },
  { label: "evaluate", text: "Freeze the workspace; run the fixed evaluator", where: "Probe", href: "#/story/probe" },
  { label: "classify", text: "Score is size reduction if smoke passes", where: "Probe", href: "#/story/probe" },
  { label: "promote", text: "Save the tree and manifest; promote top-k", where: "Tree", href: "#/story/tree" },
  { label: "replay", text: "Replay the history across the beta grid", where: "Dreaming", href: "#/story/dreaming" },
  { label: "rewrite", text: "Rewrite the policy; audit, replay, select", where: "What changed", href: "#/story/changed" },
  { label: "reload", text: "Next cycle loads the selected policy", where: "What changed", href: "#/story/changed" },
];

const DEVIATIONS = [
  {
    label: "resident",
    text: "Resident bytes are the max of tensor walk and live CUDA",
    where: "Evaluation; rule not shown",
    href: "#/story/probe",
  },
  { label: "budget", text: "Replay has no probe budget or round cap" },
  { label: "parallel", text: "Replay parallelism is the CLI flag, default four" },
  { label: "beta", text: "Beta is never updated between cycles" },
  { label: "smoke", text: "Smoke is not a separate state set" },
  {
    label: "dev gate",
    text: "The development gate is recorded for promoted nodes only",
    where: "Evaluation; not used to select",
    href: "#/story/probe",
  },
  { label: "lambda", text: "Lambda and the beta grid live in code" },
  { label: "attainment", text: "Attainment is baseline-relative, not score over S_i" },
  { label: "host", text: "The selected policy runs in the host process" },
  { label: "bwrap", text: "--no-bwrap runs agents and replay on the host" },
  {
    label: "grid",
    text: "Static grid plan; no adaptive planner",
    where: "Tree; plan not shown",
    href: "#/story/tree",
  },
  { label: "unused", text: "The lambda × W grid and seed-stability helpers are unused" },
];

function pad(index) {
  return String(index).padStart(2, "0");
}

function where(item, ctx) {
  const { el } = ctx.dom;
  if (!item.href) return el("span", { class: "method-where is-none", text: "not shown" });
  return el("a", { class: "method-where", href: item.href, text: item.where });
}

function row(item, index, ctx) {
  const { el } = ctx.dom;
  return el("li", { class: index == null ? "method-row is-plain" : "method-row" }, [
    index == null ? null : el("span", { class: "method-num", text: pad(index) }),
    el("span", { class: "method-label", text: item.label }),
    el("span", { class: "method-text", text: item.text }),
    where(item, ctx),
  ]);
}

function block(heading, items, ctx, numbered) {
  const { el } = ctx.dom;
  const list = el(numbered ? "ol" : "ul", { class: "method-list" });
  items.forEach((item, index) => list.append(row(item, numbered ? index + 1 : null, ctx)));
  return el("section", { class: "method-block" }, [
    el("h3", { class: "method-head", text: heading }),
    list,
  ]);
}

export function mount(root, ctx) {
  const { el, clear } = ctx.dom;

  function draw() {
    clear(root);
    root.append(el("p", { class: "chapter-kicker" }, [
      el("span", { class: "chapter-number" }, number),
      el("span", {}, "how it works"),
    ]));
    root.append(el("h2", { class: "chapter-title", text: title }));
    root.append(el("p", { class: "chapter-dek", text: dek }));
    const body = el("div", { class: "chapter-body method" });
    body.append(block("The loop", LOOP, ctx, true));
    body.append(block("Deviations from the paper", DEVIATIONS, ctx, false));
    body.append(
      el("p", {
        class: "method-note",
        text: "V is replay value on frozen trees, not LIBERO success. The paper states no numbered algorithm; this is the repo's loop.",
      }),
    );
    root.append(body);
  }

  draw();
  return {
    update() {},
    destroy() {
      clear(root);
    },
  };
}
