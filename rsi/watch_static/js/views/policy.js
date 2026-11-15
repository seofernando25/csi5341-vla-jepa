/* Policy development ledger in the sidebar: one timeline row per policy step. */
import { el, clear } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";

const VERDICT_TONE = { accepted: "ok", not_better: "warn", rejected: "err", running: "live" };
const DEV_TONE = { done: "ok", running: "live", skipped: "warn" };

export function mount(root, ctx) {
  const { dispatch, act } = ctx;
  const head = el("div", { class: "section-head policy-head" });
  const summary = el("div", { class: "policy-summary" });
  const body = el("div", { class: "sidebar-body policy-body" });
  root.append(head, summary, body);
  let sig = "";
  let focusSource = null;

  function renderHead(dev) {
    const pill = el("span", {
      class: "pill policy-status",
      dataset: { tone: DEV_TONE[dev?.status] || "" },
    });
    pill.textContent = dev?.status || "—";
    const selected = el("span", {
      class: "policy-selected subtle num",
    });
    selected.textContent = dev?.selected_step != null ? `step ${dev.selected_step}` : "—";
    head.append(el("span", {}, "Policy development"), pill, selected);
  }

  function renderSummary(dev) {
    const cells = [
      ["incumbent V", fmt.num(dev?.incumbent_v)],
      ["selected V", fmt.num(dev?.selected_v)],
      ["held-out Δ", fmt.signed(dev?.heldout_delta)],
    ];
    for (const [label, value] of cells) {
      summary.append(
        el("span", { class: "policy-kv" }, [
          el("i", {}, label),
          el("b", { class: "num" }, value),
        ]),
      );
    }
  }

  function deltaTone(value) {
    if (value == null || Number.isNaN(Number(value))) return "";
    return value > 0 ? "ok" : value < 0 ? "err" : "";
  }

  function rowNode(state, row, index) {
    const step = row.step;
    const tone = VERDICT_TONE[step.status] || "";
    const open = state.ui.activeId === row.source;
    const button = el(
      "button",
      {
        class: "policy-row",
        type: "button",
        tabIndex: row.selected || (!state.ui.activeId && index === 0) ? 0 : -1,
        dataset: { source: row.source, tone, selected: row.selected ? "1" : "0", open: open ? "1" : "0" },
        "aria-current": open ? "true" : "false",
      },
      [
        el("span", { class: "dot policy-node", dataset: { tone } }),
        el("span", { class: "policy-main" }, [
          el("span", { class: "policy-line" }, [
            el("span", { class: "policy-step mono" }, `step ${step.step}`),
            el("span", { class: "policy-verdict", dataset: { tone } }, fmt.verdictLabel(step.status)),
            row.selected ? el("span", { class: "policy-badge" }, "selected") : null,
            el("span", {
              class: "policy-delta num",
              dataset: { tone: deltaTone(step.delta_vs_best) },
            }, fmt.signed(step.delta_vs_best)),
          ]),
          el("span", { class: "policy-line policy-meta" }, [
            el("span", { class: "num" }, `V ${fmt.num(step.mean_v)}`),
            step.attempts != null ? el("span", { class: "num" }, fmt.plural(step.attempts, "attempt")) : null,
          ]),
          step.reason ? el("span", { class: "policy-reason truncate", title: step.reason }, step.reason) : null,
        ]),
      ],
    );
    button.addEventListener("click", () => dispatch(act.open(row.source)));
    button.addEventListener("focus", () => {
      focusSource = row.source;
    });
    button.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
      event.preventDefault();
      const rows = [...body.querySelectorAll(".policy-row")];
      const at = rows.indexOf(button);
      const target = rows[at + (event.key === "ArrowDown" ? 1 : -1)];
      if (!target) return;
      for (const other of rows) other.tabIndex = -1;
      target.tabIndex = 0;
      target.focus();
    });
    return el("li", { class: "policy-item" }, button);
  }

  function render(state) {
    const cycle = sel.currentCycle(state);
    const dev = cycle?.policy_dev || null;
    const rows = sel.policyRows(state, cycle);
    const next = JSON.stringify([
      cycle?.cycle_id,
      dev?.status,
      dev?.selected_step,
      dev?.incumbent_v,
      dev?.selected_v,
      dev?.heldout_delta,
      rows.map((row) => [
        row.step.step,
        row.step.status,
        row.step.mean_v,
        row.step.delta_vs_best,
        row.step.attempts,
        row.step.reason,
        row.selected,
      ]),
      state.ui.activeId,
    ]);
    if (next === sig) return;
    sig = next;
    const hadFocus = body.contains(document.activeElement);
    clear(head);
    clear(summary);
    clear(body);
    renderHead(dev);
    summary.hidden = !dev;
    if (dev) renderSummary(dev);
    if (!cycle) {
      body.append(el("p", { class: "empty" }, "No cycle recorded."));
      return;
    }
    if (!rows.length) {
      body.append(
        el("p", { class: "empty" }, [
          el("strong", {}, dev?.status === "pending" ? "Pending" : "No steps"),
          dev?.status === "pending" ? "Starts when the cycle ends." : "No policy steps in this cycle.",
        ]),
      );
      return;
    }
    const list = el("ol", { class: "policy-timeline", "aria-label": "Policy steps" }, rows.map((row, index) => rowNode(state, row, index)));
    body.append(list);
    if (hadFocus && focusSource) {
      const target = list.querySelector(`[data-source="${cssEscape(focusSource)}"]`);
      if (target) target.focus({ preventScroll: true });
    }
  }

  render(ctx.store.getState());
  return { update: render, destroy() {} };
}

function cssEscape(value) {
  return String(value).replace(/["\\]/g, "\\$&");
}
