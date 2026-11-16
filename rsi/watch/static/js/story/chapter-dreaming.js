/* Chapter 04 — Dreaming: after a cycle ends, candidate exploration policies are
 * replayed on the recorded trees. One payload per world; rounds grow with the
 * scrubber. V is replay-only and in-sample. */

import { el, clear } from "../dom.js";
import { cycleOf, dreamModel } from "./model.js";
import { worldFigure, pathsFigure, betaLabel } from "./dreaming-figures.js";
import { curvesFigure, vFigure } from "./dreaming-charts.js";

export const id = "dreaming";
export const number = "04";
export const title = "Dreaming";
export const dek =
  "After a cycle, candidate exploration policies are replayed on the recorded trees with no GPU; each candidate is a path of probes through the same world.";

const EMPTY = "No policy development recorded for this cycle.";

export function mount(root, ctx) {
  root.append(
    el("div", { class: "chapter-kicker" }, number),
    el("h2", { class: "chapter-title" }, title),
    el("p", { class: "chapter-dek" }, dek),
  );
  const body = el("div", { class: "chapter-body dream" });
  root.append(body);

  let state = null;
  let payload = null;
  let cycleId = null;
  let worldId = null;
  let beta = null;
  let selected = null;
  let loading = false, loadingCycle = null, devSeen = null;
  let token = 0;
  let alive = true;
  let sig = "";
  let paths = null;
  let pathsHost = null;
  let pending = null;
  let betaButtons = [];

  /* --- data ------------------------------------------------------------ */

  async function load() {
    const cycle = dreamCycle(state);
    if (!cycle) return false;
    const mine = ++token;
    const cid = cycle.cycle_id;
    loading = true;
    loadingCycle = cid;
    const base = await ctx.story.dream(cid);
    if (!alive || mine !== token) return false;
    if (!base || base.status !== "ready" || !(base.worlds || []).length) {
      payload = base || null;
      cycleId = cid;
      loading = false;
      loadingCycle = null;
      sig = "";
      render();
      return false;
    }
    const wanted = worldId && base.worlds.some((world) => world.id === worldId) ? worldId : base.worlds[0].id;
    const full = await ctx.story.dream(cid, wanted);
    if (!alive || mine !== token) return false;
    loading = false;
    loadingCycle = null;
    if (!full) return false;
    payload = full;
    cycleId = cid;
    worldId = wanted;
    if (beta == null) beta = defaultBeta(payload);
    sig = "";
    render();
    return true;
  }

  /* --- render ---------------------------------------------------------- */

  function render() {
    const cycle = dreamCycle(state);
    const status = !cycle
      ? "empty"
      : payload
        ? payload.status === "ready" ? "ready" : payload.status === "pending" ? "pending" : "empty"
        : loading ? "loading" : "empty";
    const next = [cycle?.cycle_id, status, worldId].join("|");
    if (next === sig) {
      if (status === "pending") refreshPending(cycle);
      return;
    }
    sig = next;
    if (paths) paths.destroy();
    paths = null;
    pathsHost = null;
    pending = null;
    betaButtons = [];
    clear(body);
    if (status === "ready") paintReady(payload);
    else if (status === "pending") paintPending(cycle);
    else if (status === "loading") body.append(el("p", { class: "dream-note" }, "Loading the replay."));
    else body.append(el("p", { class: "dream-empty" }, EMPTY));
  }

  function paintReady(data) {
    const model = dreamModel(data);
    selected = model.verdict?.selected_slot || null;
    const world = model.world(worldId) || (model.worlds || [])[0] || null;
    if (!world) return body.append(el("p", { class: "dream-empty" }, EMPTY));
    worldId = world.id;
    body.append(controls(model, data));
    body.append(worldFigure(world, model.tree(world.id)));
    pathsHost = el("div", { class: "dream-paths-host" });
    body.append(pathsHost);
    mountPaths(model);
    body.append(curvesFigure(model, world.id, selected));
    body.append(vFigure(model, selected));
    body.append(heldout(model));
    body.append(el("p", { class: "dream-honesty" }, "V is replay-only and in-sample."));
  }

  function mountPaths(model) {
    if (paths) paths.destroy();
    paths = pathsFigure({ payload, model, worldId, beta, selected, reduced: Boolean(ctx.story.reduced) });
    clear(pathsHost);
    pathsHost.append(paths.node);
  }

  function controls(model, data) {
    const select = el("select", { class: "dream-select", aria: { label: "World" } },
      (model.worlds || []).map((world) => el("option", { value: world.id }, world.id)));
    select.value = worldId;
    select.addEventListener("change", async () => {
      const previous = worldId;
      worldId = select.value;
      if (!(await load()) && alive) {
        worldId = previous;
        select.value = previous;
      }
    });
    const seg = el("div", { class: "dream-seg", role: "group", aria: { label: "Beta" } });
    betaButtons = betaGrid(model).map((value) => {
      const button = el("button", {
        class: "dream-seg-btn",
        type: "button",
        dataset: { beta: String(value) },
        aria: { pressed: String(value === beta) },
      }, betaLabel(value));
      button.addEventListener("click", () => setBeta(value));
      seg.append(button);
      return button;
    });
    return el("div", { class: "dream-controls" }, [
      el("label", { class: "dream-field" }, [el("span", { class: "dream-field-label" }, "World"), select]),
      el("div", { class: "dream-field" }, [el("span", { class: "dream-field-label" }, "Beta"), seg]),
      el("span", { class: "dream-controls-meta" }, metaLine(model, data)),
    ]);
  }

  function setBeta(value) {
    beta = value;
    for (const button of betaButtons) button.setAttribute("aria-pressed", String(Number(button.dataset.beta) === value));
    if (payload) mountPaths(dreamModel(payload));
  }

  /* --- pending --------------------------------------------------------- */

  function paintPending(cycle) {
    pending = { stats: el("dl", { class: "dream-stats" }) };
    body.append(el("div", { class: "dream-pending" }, [
      el("p", { class: "dream-pending-title" }, `Cycle ${cycleNumber(cycle)} is still exploring.`),
      el("p", { class: "dream-pending-sub" }, "Dreaming starts when the cycle ends."),
      pending.stats,
    ]));
    refreshPending(cycle);
  }

  function refreshPending(cycle) {
    if (!pending) return;
    const data = state?.server?.data || {};
    const nodes = cycle?.nodes || [];
    const scored = nodes.filter((node) => node.cell_id !== "root" && node.score != null).length;
    const budget = data.loop?.probe_budget;
    const items = [
      ["probes scored", budget == null ? String(scored) : `${scored} / ${budget}`],
      ["elapsed", clock(data.loop?.elapsed_s)],
      ["nodes", String(nodes.length)],
    ];
    clear(pending.stats);
    for (const [label, value] of items) {
      pending.stats.append(el("div", { class: "dream-stat" }, [
        el("dt", { class: "dream-stat-label" }, label),
        el("dd", { class: "dream-stat-value" }, value),
      ]));
    }
  }

  /* --- shell ----------------------------------------------------------- */

  return {
    update(next) {
      state = next;
      const cycle = dreamCycle(state);
      if (!cycle) {
        payload = null; cycleId = null; sig = "";
        render();
        return;
      }
      if (cycle.cycle_id !== cycleId || !payload) {
        if (cycle.cycle_id !== cycleId) {
          payload = null;
          worldId = null;
          beta = null;
          devSeen = null;
        }
        render();
        if (!(loading && loadingCycle === cycle.cycle_id)) load();
        return;
      }
      const dev = cycle.policy_dev?.status || null;
      if (payload.status === "pending" && dev && dev !== "pending" && dev !== devSeen) {
        devSeen = dev;
        payload = null;
        sig = "";
        render();
        load();
        return;
      }
      devSeen = dev;
      render();
    },
    destroy() {
      alive = false;
      token += 1;
      if (paths) paths.destroy();
    },
  };
}

/* --- helpers ----------------------------------------------------------- */

/* Dreaming follows the latest cycle that finished policy development, else the active cycle. */
function dreamCycle(state) {
  const cycles = state?.server?.data?.cycles || [];
  const done = cycles.filter((c) => c?.policy_dev?.status === "done" && c?.policy_dev?.selected_step != null);
  return done.length ? done[done.length - 1] : cycleOf(state);
}

function defaultBeta(data) {
  const grid = betaGrid(dreamModel(data));
  if (!grid.length) return null;
  return grid.includes(0.5) ? 0.5 : grid[Math.floor(grid.length / 2)];
}

function betaGrid(model) {
  const grid = (model.protocol?.beta_grid || []).map(Number).filter(Number.isFinite);
  if (grid.length) return grid;
  const seen = new Set();
  for (const per of Object.values(model.curves || {})) {
    for (const points of Object.values(per || {})) {
      for (const point of points || []) if (Number.isFinite(Number(point?.beta))) seen.add(Number(point.beta));
    }
  }
  return [...seen].sort((a, b) => a - b);
}

function cycleNumber(cycle) {
  return cycle?.iteration == null ? "This cycle" : cycle.iteration + 1;
}

function clock(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return "—";
  const total = Math.max(0, Math.round(Number(seconds)));
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  if (minutes < 60) return `${minutes}:${String(rest).padStart(2, "0")}`;
  return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function metaLine(model, data) {
  const parts = [`${(model.candidates || []).length} candidates`, `${(model.worlds || []).length} worlds`];
  const date = data?.recorded_utc ? new Date(data.recorded_utc) : null;
  if (date && !Number.isNaN(date.getTime())) {
    parts.push(`recorded ${date.toISOString().slice(0, 16).replace("T", " ")} UTC`);
  }
  return parts.join(" · ");
}

function heldout(model) {
  const held = model.heldout;
  const delta = Number(held?.delta_vs_incumbent);
  if (!held || held.delta_vs_incumbent == null || !Number.isFinite(delta)) {
    return el("p", { class: "dream-heldout" }, [el("span", { class: "dream-heldout-label" }, "held-out not recorded")]);
  }
  const parts = [
    el("span", { class: "dream-heldout-label" }, "Held-out V "),
    el("span", { class: "dream-heldout-value" }, `${delta > 0 ? "+" : ""}${delta.toFixed(3)}`),
  ];
  if (held.num_worlds != null) parts.push(el("span", { class: "dream-heldout-label" }, ` over ${held.num_worlds} worlds`));
  return el("div", { class: "dream-heldout" }, parts);
}
