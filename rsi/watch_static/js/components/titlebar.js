/* Title bar: run identity, loop status, cycle switcher, compact metrics, actions. */

import { el, clear, icon, setText, modLabel } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";

export function mount(root, ctx) {
  const { dispatch, act } = ctx;

  const brand = el("div", { class: "brand" }, [
    el("span", { class: "brand-mark", "aria-hidden": "true" }, "R"),
    el("span", { class: "brand-name" }, "Dream-RSI"),
    el("span", { class: "brand-study" }, ""),
  ]);
  const status = el("span", { class: "pill", dataset: { tone: "idle" } }, [
    el("span", { class: "dot", dataset: { tone: "idle" } }),
    el("span", { class: "status-text" }, "connecting"),
  ]);
  const cycles = el("div", { class: "cycle-switch", role: "group", "aria-label": "Cycle" });
  const metrics = el("div", { class: "metrics" });
  const actions = el("div", { class: "titlebar-actions" });
  root.append(
    brand,
    el("span", { class: "titlebar-sep", "aria-hidden": "true" }),
    status,
    el("span", { class: "titlebar-sep", "aria-hidden": "true" }),
    cycles,
    metrics,
    el("div", { class: "titlebar-spacer" }),
    actions,
  );

  const metric = (label, id) => {
    const value = el("span", { class: "metric-value" });
    const node = el("div", { class: "metric", dataset: { metric: id } }, [
      el("span", { class: "metric-label" }, label),
      value,
    ]);
    return { node, value };
  };
  const best = metric("best score", "best");
  const scored = metric("scored", "scored");
  const cost = metric("agent cost", "cost");
  const elapsed = metric("elapsed", "elapsed");
  const gpu = metric("gpu", "gpu");
  metrics.append(best.node, scored.node, cost.node, elapsed.node, gpu.node);

  const modeSwitch = el("nav", { class: "mode-switch", "aria-label": "Mode" }, [
    el("a", { class: "mode-link", href: "#/story/task" }, "Story"),
    el("span", { class: "mode-link", "aria-current": "page" }, "Workbench"),
  ]);

  const paletteBtn = el("button", { class: "palette-trigger", type: "button", title: "Command palette" }, [
    icon("command", 13),
    el("span", {}, "Jump to…"),
    el("kbd", { class: "kbd" }, `${modLabel()} K`),
  ]);
  paletteBtn.addEventListener("click", () => dispatch(act.palette({ open: true, q: "" })));

  const themeBtn = el("button", { class: "icon-btn", type: "button", title: "Toggle theme" }, icon("moon", 14));
  themeBtn.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    dispatch(act.theme(next));
  });

  const sidebarBtn = el("button", { class: "icon-btn", type: "button", title: "Toggle sidebar" }, icon("panelLeft", 14));
  sidebarBtn.addEventListener("click", () => {
    if (matchMedia("(max-width: 900px)").matches) {
      const open = ctx.store.getState().ui.drawer !== "tree";
      dispatch(act.drawer(open ? "tree" : null));
      return;
    }
    const open = !ctx.store.getState().ui.sidebar.open;
    dispatch(act.sidebar({ open }));
  });

  const inspectorBtn = el("button", { class: "icon-btn", type: "button", title: "Toggle inspector" }, icon("panelRight", 14));
  inspectorBtn.addEventListener("click", () => {
    if (matchMedia("(max-width: 900px)").matches) {
      const open = ctx.store.getState().ui.drawer !== "inspector";
      dispatch(act.drawer(open ? "inspector" : null));
      return;
    }
    const open = !ctx.store.getState().ui.inspector.open;
    dispatch(act.inspector({ open }));
  });

  actions.append(modeSwitch, paletteBtn, themeBtn, sidebarBtn, inspectorBtn);

  let cycleSig = "";
  let statusSig = "";

  function renderCycles(state) {
    const list = state.server.data?.cycles || [];
    const sig = list.map((c) => `${c.cycle_id}:${c.iteration}:${c.status}`).join("|") + `#${state.ui.cycleId}`;
    if (sig === cycleSig) return;
    cycleSig = sig;
    clear(cycles);
    for (const cycle of list) {
      const button = el(
        "button",
        {
          type: "button",
          "aria-pressed": state.ui.cycleId === cycle.cycle_id ? "true" : "false",
          title: cycle.status,
        },
        String(cycle.iteration ?? cycle.cycle_id),
      );
      button.addEventListener("click", () => dispatch(act.cycle(cycle.cycle_id)));
      cycles.append(button);
    }
    cycles.hidden = list.length < 2;
  }

  function renderStatus(state) {
    const loop = sel.loop(state);
    const tone =
      loop?.status === "running" ? "live" : loop?.status === "error" ? "err" : loop?.status === "done" ? "ok" : "idle";
    const label = sel.loopStatusLabel(state);
    const sig = `${tone}:${label}:${loop?.stage || ""}`;
    if (sig === statusSig) return;
    statusSig = sig;
    status.dataset.tone = tone;
    status.querySelector(".dot").dataset.tone = tone;
    setText(status.querySelector(".status-text"), label);
    status.title = loop?.stop_reason || label;
  }

  function renderMetrics(state) {
    const loop = sel.loop(state);
    const cycle = sel.currentCycle(state);
    const budget = sel.probeBudget(state);
    const scoredCount = sel.scoredCount(cycle);
    const bestNode = sel.bestOf(state);
    setText(best.value, bestNode ? fmt.pct(bestNode.node.score) : "—");
    best.node.title = bestNode
      ? `${bestNode.node.cell_id} in cycle ${bestNode.cycle.iteration ?? bestNode.cycle.cycle_id}`
      : "no scored node yet";
    setText(scored.value, budget == null ? String(scoredCount) : `${scoredCount} / ${budget}`);
    scored.node.title = budget == null ? "nodes scored in this cycle" : "nodes scored / probe budget of this run";
    setText(cost.value, fmt.usd(loop?.agent_cost_usd));
    const liveSeconds =
      loop?.status === "running" && loop.elapsed_s != null
        ? loop.elapsed_s + Math.max(0, (Date.now() - (state.server.at || Date.now())) / 1000)
        : loop?.elapsed_s;
    setText(elapsed.value, fmt.elapsed(liveSeconds));
    const gpuData = state.server.data?.gpu;
    setText(
      gpu.value,
      gpuData ? `${fmt.mb(gpuData.memory_used_mb)} / ${fmt.mb(gpuData.memory_total_mb)}` : "—",
    );
    gpu.node.title = gpuData ? `${gpuData.name} · ${gpuData.utilization_pct}% utilisation` : "no GPU reported";
    setText(brand.querySelector(".brand-study"), state.server.data?.study_dir || "");
    themeBtn.replaceChildren(icon(document.documentElement.dataset.theme === "dark" ? "sun" : "moon", 14));
  }

  function render(state) {
    renderCycles(state);
    renderStatus(state);
    renderMetrics(state);
  }

  render(ctx.store.getState());
  return { update: render, destroy() {} };
}
