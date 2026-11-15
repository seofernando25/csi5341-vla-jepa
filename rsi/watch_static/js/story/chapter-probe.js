/* Chapter 02 — One probe: one recorded node, from the prompt it was given to
 * the score it earned. The transcript is fetched once per node and cached. */

import { el, clear } from "../dom.js";
import { probeNode, probeTimeline, reasoningLines } from "./model.js";
import {
  clip,
  displayText,
  elapsedOf,
  firstHeading,
  proposalBody,
  signature,
  toneOf,
} from "./probe-parts.js";

export const id = "probe";
export const number = "02";
export const title = "One probe";
export const dek = "One node of the search, from the prompt it was given to the score it earned.";

const GIVEN_LIMIT = 420;
const REASONING_LIMIT = 5;
const TOOL_LIMIT = 6;
const DETAIL_LIMIT = 100;

export function mount(root, ctx) {
  root.classList.add("probe-root");
  const card = el("div", { class: "figure probe" });
  const head = el("div", { class: "figure-head" });
  const idRow = el("div", { class: "probe-id-row" });
  const steps = el("ol", { class: "probe-steps" });
  const foot = el("div", { class: "probe-foot" });
  card.append(head, idRow, steps, foot);
  root.append(
    el("div", { class: "chapter-kicker" }, number),
    el("h2", { class: "chapter-title" }, title),
    el("p", { class: "chapter-dek" }, dek),
    card,
  );

  const cache = new Map(); // source -> {events, context, proposal}
  const loading = new Set();
  let alive = true;
  let state = null;
  let source = "";
  let expanded = false;
  let sig = "";

  /* Load a node's three payloads once; a stale reply is cached, never painted. */
  function ensure(key) {
    if (!key || cache.has(key) || loading.has(key)) return;
    loading.add(key);
    const settle = (entry) => {
      loading.delete(key);
      cache.set(key, entry);
      if (alive) render();
    };
    Promise.all([
      ctx.story.events(key),
      ctx.story.context(key),
      ctx.story.artifact(key, "proposal"),
    ]).then(
      ([events, context, proposal]) => settle({ events, context, proposal }),
      () => settle({ events: null, context: null, proposal: null }),
    );
  }

  function render() {
    const probe = probeNode(state);
    const key = probe ? ctx.sel.sourceOf("node", probe.cycle.cycle_id, probe.node.cell_id) : "";
    if (key !== source) {
      source = key;
      expanded = false;
    }
    ensure(key);
    const entry = cache.get(key) || null;
    const next = signature(probe, entry, loading.has(key), expanded);
    if (next === sig) return;
    sig = next;
    paint(probe, entry);
  }

  function paint(probe, entry) {
    clear(head);
    clear(idRow);
    clear(steps);
    clear(foot);
    if (!probe) {
      head.append(el("span", {}, "probe"));
      idRow.append(el("p", { class: "probe-empty" }, "No probe recorded in this run yet."));
      return;
    }
    const node = probe.node;
    const cycle = probe.cycle;
    const elapsed = elapsedOf(node, ctx.fmt);
    const where = [`cycle ${cycle.cycle_id}`];
    if (node.branch != null) where.push(`branch ${node.branch}`);
    if (node.attempt != null) where.push(`attempt ${node.attempt}`);
    head.append(
      el("span", {}, "probe"),
      el("span", { class: "probe-head-meta mono" }, where.join(" · ")),
    );
    idRow.append(
      el("h3", { class: "probe-cell" }, node.cell_id),
      el("span", { class: "probe-pill", dataset: toneOf(node.status) }, ctx.fmt.nodeStatusLabel(node.status)),
      elapsed ? el("span", { class: "probe-elapsed" }, `elapsed ${elapsed}`) : null,
    );

    steps.append(
      step("01", "Given", givenPart(entry)),
      step("02", "Reasoning", reasoningPart(entry)),
      step("03", "Tool calls", toolsPart(entry)),
      step("04", "Proposal", proposalPart(entry, node)),
      step("05", "Verdict", verdictPart(node)),
    );

    const open = el("button", { class: "probe-open", type: "button" }, "Open this probe in the Workbench");
    open.addEventListener("click", () => ctx.story.open(source, "transcript"));
    foot.append(open);
  }

  function step(index, label, part) {
    return el("li", { class: "probe-step" }, [
      el("span", { class: "probe-step-index mono", aria: { hidden: "true" } }, index),
      el("div", { class: "probe-step-main" }, [
        el("div", { class: "probe-step-label" }, [
          el("span", {}, label),
          part.note ? el("span", { class: "probe-step-note" }, part.note) : null,
        ]),
        ...part.body,
      ]),
    ]);
  }

  function note(text) {
    return el("p", { class: "probe-note" }, text);
  }

  function givenPart(entry) {
    if (!entry) return { body: [note("Loading…")] };
    const section = (entry.context?.sections || []).find((item) => item.id === "message");
    const full = section && section.recorded !== false ? displayText(section.text) : "";
    if (!full) return { body: [note("not recorded")] };
    const shown = expanded ? full : full.slice(0, GIVEN_LIMIT).trimEnd();
    const body = [el("p", { class: "probe-given" }, shown + (expanded || full.length <= GIVEN_LIMIT ? "" : "…"))];
    if (full.length > GIVEN_LIMIT) {
      const more = el("button", {
        class: "probe-more",
        type: "button",
        aria: { expanded: String(expanded) },
      }, expanded ? "Show less" : "Show all");
      more.addEventListener("click", () => {
        expanded = !expanded;
        render();
      });
      body.push(more);
    }
    return { body };
  }

  function reasoningPart(entry) {
    if (!entry) return { body: [note("Loading…")] };
    if (!entry.events?.events?.length) return { body: [note("transcript not recorded")] };
    const lines = reasoningLines(entry.events, REASONING_LIMIT);
    if (!lines.length) return { body: [note("not recorded")] };
    const total = reasoningLines(entry.events, Infinity).length;
    return {
      note: `${lines.length} of ${total}`,
      body: lines.map((line) => el("blockquote", { class: "probe-quote" }, `“${line}”`)),
    };
  }

  function toolsPart(entry) {
    if (!entry) return { body: [note("Loading…")] };
    if (!entry.events?.events?.length) return { body: [note("transcript not recorded")] };
    const timeline = probeTimeline(entry.events);
    const tools = timeline.tools.slice(0, TOOL_LIMIT);
    if (!tools.length) return { body: [note("not recorded")] };
    const rows = tools.map((tool) => {
      const detail = tool.summary || tool.command || "";
      return el("div", { class: "probe-tool" }, [
        el("span", { class: "probe-dot", dataset: { tone: tool.isError ? "err" : "ok" }, aria: { hidden: "true" } }),
        el("span", { class: "probe-tool-name mono" }, tool.name),
        el("span", { class: "probe-tool-detail mono" }, clip(detail, DETAIL_LIMIT)),
      ]);
    });
    return {
      note: `${tools.length} of ${timeline.counts.tool_call ?? timeline.tools.length}`,
      body: rows,
    };
  }

  function proposalPart(entry, node) {
    if (!entry) return { body: [note("Loading…")] };
    const text = String(entry.proposal?.text || "");
    const headline = firstHeading(text) || node.headline || "";
    const body = [];
    if (headline) body.push(el("p", { class: "probe-proposal-head" }, headline));
    const lines = proposalBody(text);
    if (lines) body.push(el("p", { class: "probe-proposal-body" }, lines));
    if (!body.length) body.push(note("proposal not recorded"));
    return { body };
  }

  function verdictPart(node) {
    if (node.score == null) {
      const bits = [ctx.fmt.nodeStatusLabel(node.status)];
      const elapsed = elapsedOf(node, ctx.fmt);
      if (elapsed) bits.push(`${elapsed} elapsed`);
      return { body: [note(`Not scored yet · ${bits.join(" · ")}`)] };
    }
    const stats = [
      stat("score", ctx.fmt.pct(node.score)),
      node.successes != null && node.episodes != null
        ? stat("smoke", `${node.successes} / ${node.episodes}`)
        : null,
      node.size_reduction != null ? stat("size", `−${ctx.fmt.pct(node.size_reduction)}`) : null,
      node.eval_seconds != null ? stat("eval", ctx.fmt.secs(node.eval_seconds)) : null,
      node.agent_cost_usd != null ? stat("cost", ctx.fmt.usd(node.agent_cost_usd)) : null,
      node.fail_class ? stat("outcome", ctx.fmt.failLabel(node.fail_class)) : null,
    ].filter(Boolean);
    return { body: [el("div", { class: "probe-stats" }, stats)] };
  }

  function stat(label, value) {
    return el("div", { class: "probe-stat" }, [
      el("div", { class: "probe-stat-value" }, value),
      el("div", { class: "probe-stat-label" }, label),
    ]);
  }

  return {
    update(next) {
      state = next;
      render();
    },
    destroy() {
      alive = false;
      cache.clear();
      loading.clear();
    },
  };
}
