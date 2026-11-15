/* Inspector: facts for the active source, lineage breadcrumb and the cycle score chart. */

import { el, clear, icon } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";
import { createChart } from "./chart.js";

const STEP_TONE = { accepted: "ok", not_better: "warn", rejected: "err", running: "live", pending: "idle" };

export function mount(root, ctx) {
  const { dispatch, act, go } = ctx;

  const close = el(
    "button",
    { class: "icon-btn", type: "button", title: "Close inspector", "aria-label": "Close inspector" },
    icon("close", 13),
  );
  close.addEventListener("click", () => {
    dispatch(act.inspector({ open: false }));
    if (matchMedia("(max-width: 900px)").matches) dispatch(act.drawer(null));
  });

  const head = el("div", { class: "inspector-head" }, [el("span", {}, "Inspector"), close]);
  const body = el("div", { class: "inspector-body" });
  root.append(head, body);

  const chart = createChart();
  let sig = "";

  function render(state) {
    const tab = sel.activeTab(state);
    const source = tab ? tab.id : null;
    const node = tab && tab.kind === "node" ? sel.nodeForSource(state, source) : null;
    const step = tab && tab.kind === "policy" ? sel.stepForSource(state, source) : null;
    const cycle = tab ? sel.cycleById(state, tab.cycle) : null;
    const next = signature(tab, node, step, cycle, sel.loopLive(state));
    if (next === sig) return;
    sig = next;
    clear(body);

    if (!tab) {
      chart.destroy();
      body.append(el("p", { class: "empty" }, "Nothing selected."));
      return;
    }
    if (tab.kind === "node") {
      if (!node) {
        chart.destroy();
        body.append(el("p", { class: "empty" }, [el("strong", {}, "Node not in this cycle"), source]));
        return;
      }
      renderNode(state, tab, cycle, node);
      return;
    }
    chart.destroy();
    if (!step) {
      body.append(el("p", { class: "empty" }, [el("strong", {}, "Step not found"), source]));
      return;
    }
    renderStep(cycle, step);
  }

  function renderNode(state, tab, cycle, node) {
    const phase = sel.nodePhase(state, node);
    const isRoot = node.cell_id === "root";
    const title = isRoot ? "baseline" : node.cell_id;

    body.append(
      el("div", { class: "insp-head" }, [
        el("h2", { class: "insp-title truncate", title }, title),
        el(
          "span",
          { class: "pill", dataset: { tone: phaseTone(phase, node) }, title: `recorded status: ${node.status}` },
          fmt.nodeStatusLabel(phase),
        ),
      ]),
      el("p", { class: "insp-sub" }, subtitle(cycle, node)),
    );

    const filled = [node.score, node.successes, node.agent_cost_usd, node.elapsed_s ?? node.agent_seconds].filter(
      (value) => value != null,
    ).length;
    if (filled < 2) {
      body.append(el("p", { class: "insp-empty" }, "Not scored yet"));
    } else {
      const stats = el("dl", { class: "insp-stats" });
      stat(stats, "score", fmt.pct(node.score));
      stat(
        stats,
        "smoke",
        node.successes == null || node.episodes == null ? "—" : `${node.successes}/${node.episodes}`,
      );
      stat(stats, "cost", fmt.usd(node.agent_cost_usd));
      if (node.elapsed_s != null) stat(stats, "elapsed", fmt.dur(node.elapsed_s));
      else stat(stats, "agent", fmt.dur(node.agent_seconds));
      body.append(stats);
    }

    const chain = sel.lineage(state, tab.cycle, node.cell_id);
    if (chain.length > 1) body.append(section("Lineage", lineageNav(tab, chain, go)));

    const chartBox = el("div", { class: "insp-chart" });
    body.append(
      el("section", { class: "insp-section" }, [
        el("h3", { class: "insp-label" }, [
          el("span", {}, `Score · ${cycleLabel(cycle)}`),
          el("span", { class: "insp-legend", "aria-hidden": "true" }, [
            el("i", { class: "insp-key", dataset: { kind: "dot" } }),
            "score",
            el("i", { class: "insp-key", dataset: { kind: "line" } }),
            "best",
          ]),
        ]),
        chartBox,
      ]),
    );
    chart.update(chartBox, chartData(state, cycle, node.cell_id));

    const facts = el("dl", { class: "insp-facts" });
    fact(facts, "parent", parentValue(state, tab, node, go));
    fact(facts, "branch", node.branch == null ? "—" : String(node.branch));
    fact(facts, "attempt", node.attempt == null ? "—" : String(node.attempt));
    fact(facts, "commit", node.commit ? fmt.shortId(node.commit, 8) : "—", {
      mono: true,
      title: node.commit || null,
    });
    fact(facts, "headline", node.headline || "—", { wrap: true });
    body.append(section("Facts", facts));
  }

  function renderStep(cycle, step) {
    const dev = cycle?.policy_dev || {};
    const selected = dev.selected_step != null && dev.selected_step === step.step;

    body.append(
      el("div", { class: "insp-head" }, [
        el("h2", { class: "insp-title" }, `Step ${step.step}`),
        el(
          "span",
          { class: "pill", dataset: { tone: STEP_TONE[step.status] || "idle" } },
          fmt.verdictLabel(step.status),
        ),
      ]),
      el("p", { class: "insp-sub" }, `${cycleLabel(cycle)} · policy development`),
    );
    if (selected) body.append(el("div", { class: "insp-selected" }, [icon("check", 11), "selected"]));

    const stats = el("dl", { class: "insp-stats" });
    stat(stats, "mean V", fmt.num(step.mean_v));
    stat(stats, "delta vs best", fmt.signed(step.delta_vs_best));
    stat(stats, "step", String(step.step));
    stat(stats, "attempts", "—", "not recorded");
    body.append(stats);

    body.append(section("Reason", el("p", { class: "insp-reason wrap-any" }, step.reason || "—")));

    const facts = el("dl", { class: "insp-facts" });
    fact(facts, "incumbent V", fmt.num(dev.incumbent_v));
    fact(facts, "selected V", fmt.num(dev.selected_v));
    fact(facts, "held-out delta", fmt.signed(dev.heldout_delta));
    body.append(section("Selection", facts));
  }

  render(ctx.store.getState());
  return { update: render, destroy() { chart.destroy(); } };
}

function signature(tab, node, step, cycle, live) {
  if (!tab) return "none";
  if (tab.kind === "node") {
    return JSON.stringify([
      tab.id,
      live,
      node && [
        node.status,
        node.fail_class,
        node.score,
        node.successes,
        node.episodes,
        node.agent_seconds,
        node.elapsed_s,
        node.agent_cost_usd,
        node.commit,
        node.headline,
        node.parent_id,
        node.branch,
        node.attempt,
        node.round,
      ],
      (cycle?.nodes || []).map((n) => [n.cell_id, n.parent_id, n.score, n.status, n.fail_class]),
    ]);
  }
  return JSON.stringify([tab.id, tab.cycle, step, cycle?.policy_dev]);
}

function chartData(state, cycle, highlight) {
  if (!cycle) return { points: [], highlight };
  const rows = sel.treeRows(state, cycle).filter((row) => row.node.cell_id !== "root");
  let best = null;
  const points = rows.map((row, index) => {
    const score = row.node.score == null ? null : Number(row.node.score);
    if (score != null && !Number.isNaN(score)) best = best == null ? score : Math.max(best, score);
    return { x: index, score, best, cell: row.node.cell_id };
  });
  return { points, highlight };
}

function lineageNav(tab, chain, go) {
  const nav = el("nav", { class: "insp-lineage", "aria-label": "Lineage" });
  chain.forEach((ancestor, index) => {
    if (index) nav.append(el("span", { class: "insp-sep", "aria-hidden": "true" }, "›"));
    const label = ancestor.cell_id === "root" ? "baseline" : ancestor.cell_id;
    if (index === chain.length - 1) {
      nav.append(
        el("span", { class: "insp-crumb is-current", "aria-current": "true", title: ancestor.cell_id }, label),
      );
      return;
    }
    const button = el("button", { class: "insp-crumb", type: "button", title: `Open ${ancestor.cell_id}` }, label);
    button.addEventListener("click", () => go(sel.sourceOf("node", tab.cycle, ancestor.cell_id)));
    nav.append(button);
  });
  return nav;
}

function parentValue(state, tab, node, go) {
  if (!node.parent_id) return "—";
  const label = node.parent_id === "root" ? "baseline" : node.parent_id;
  if (!sel.nodeById(state, tab.cycle, node.parent_id)) return el("span", { class: "mono" }, label);
  const button = el("button", { class: "insp-link mono", type: "button", title: `Open ${node.parent_id}` }, label);
  button.addEventListener("click", () => go(sel.sourceOf("node", tab.cycle, node.parent_id)));
  return button;
}

function section(title, child) {
  return el("section", { class: "insp-section" }, [el("h3", { class: "insp-label" }, title), child]);
}

function stat(list, label, value, title) {
  list.append(
    el("div", { class: "insp-stat" }, [
      el("dt", { class: "insp-stat-label" }, label),
      el("dd", { class: "insp-stat-value num", title: title || null }, value == null ? "—" : String(value)),
    ]),
  );
}

function fact(list, label, value, opts = {}) {
  const dd = el("dd", {
    class: `insp-fact-value${opts.mono ? " mono" : ""}${opts.wrap ? " wrap-any" : " truncate"}`,
    title: opts.title || null,
  });
  if (value instanceof Node) dd.append(value);
  else dd.textContent = value == null ? "—" : String(value);
  list.append(el("div", { class: "insp-fact" }, [el("dt", { class: "insp-fact-label" }, label), dd]));
}

function subtitle(cycle, node) {
  const bits = [cycleLabel(cycle)];
  if (node.branch != null) bits.push(`branch ${node.branch}`);
  if (node.attempt != null) bits.push(`attempt ${node.attempt}`);
  return bits.join(" · ");
}

function cycleLabel(cycle) {
  return cycle ? `cycle ${cycle.iteration ?? cycle.cycle_id}` : "no cycle";
}

function phaseTone(phase, node) {
  if (phase === "agent" || phase === "evaluating") return "live";
  if (phase === "stopped") return "warn";
  if (phase !== "done") return "idle";
  if (node.fail_class === "ok") return "ok";
  if (node.fail_class === "gate_near_miss" || node.fail_class === "gate_failed") return "warn";
  return node.fail_class ? "err" : "idle";
}
