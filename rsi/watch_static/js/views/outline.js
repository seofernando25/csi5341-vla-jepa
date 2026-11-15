/* Transcript outline rail: block navigation, search, filters and scroll-spy.
   Navigation only: it never renders block bodies and never dispatches on scroll. */

import { el, clear, icon, setText, rafBatch } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";

const FILTERS = [
  ["all", "All"],
  ["messages", "Messages"],
  ["reasoning", "Reasoning"],
  ["tools", "Tools"],
  ["errors", "Errors"],
];

/* The icon set has no brain/spark glyph; reasoning borrows the code glyph. */
const KIND_ICON = {
  text: "list",
  thinking: "code",
  tool: "terminal",
  toolresult: "terminal",
  system: "dot",
  result: "dot",
};

const SEARCH_MS = 120;

export function mount(root, ctx) {
  const { dispatch, act } = ctx;

  /* Header: label, count, follow toggle. */
  const count = el("span", { class: "count num" });
  const follow = el(
    "button",
    { class: "icon-btn outline-follow", type: "button", "aria-label": "Follow new output", aria: { pressed: "true" } },
    icon("arrowDown", 13),
  );
  follow.addEventListener("click", () => dispatch(act.follow(!ctx.store.getState().ui.follow)));
  const head = el("div", { class: "section-head outline-head" }, [el("span", {}, "Outline"), count, follow]);

  /* Search: input, hit count, prev/next. */
  const input = el("input", {
    class: "outline-search-input",
    id: "outline-search",
    type: "text",
    placeholder: "Search transcript",
    autocomplete: "off",
    spellcheck: "false",
    aria: { label: "Search transcript" },
  });
  const hitsLabel = el("span", { class: "outline-hits-label num" });
  const hitButton = (label, glyph, delta) => {
    const button = el(
      "button",
      { class: "outline-hit-btn", type: "button", aria: { label }, title: label },
      icon(glyph, 11),
    );
    button.addEventListener("click", () => step(delta));
    return button;
  };
  const prev = hitButton("Previous hit", "arrowUp", -1);
  const next = hitButton("Next hit", "arrowDown", 1);
  const hits = el("div", { class: "outline-hits", hidden: true }, [hitsLabel, prev, next]);

  /* Filter chips: shown whenever the query is empty. */
  const chips = el("div", { class: "outline-chips", role: "group", aria: { label: "Filter transcript" } });
  const chipButtons = new Map();
  for (const [id, label] of FILTERS) {
    const chip = el(
      "button",
      { class: "outline-chip", type: "button", dataset: { filter: id }, aria: { pressed: "false" } },
      label,
    );
    chip.addEventListener("click", () => dispatch(act.filter(id)));
    chipButtons.set(id, chip);
    chips.append(chip);
  }

  const top = el("div", { class: "outline-top" }, [
    head,
    el("div", { class: "outline-tools" }, [input, hits, chips]),
  ]);
  const list = el("div", { class: "outline-scroll" });
  root.append(top, list);

  let chrome = "";
  let listKey = "";
  let tailKey = "";
  let revealed = "";
  let rowCount = 0;
  let rowByIndex = new Map();
  let activeRow = null;
  let focusRow = null;
  let lastQ = null;
  let timer = null;
  let txEl = null;
  let txHandler = null;

  /* --- scroll-spy ------------------------------------------------------- */
  function blockIndex(node, fallback) {
    const value = Number(node.dataset.block ?? node.dataset.index);
    return Number.isFinite(value) ? value : fallback;
  }

  function setActive(index) {
    const row = rowByIndex.get(index) || null;
    if (row === activeRow) return;
    if (activeRow) activeRow.dataset.active = "0";
    activeRow = row;
    if (row) row.dataset.active = "1";
  }

  const spy = rafBatch(() => {
    if (!txEl) return;
    const nodes = txEl.querySelectorAll(".tx-block");
    if (!nodes.length) {
      setActive(-1);
      return;
    }
    const box = txEl.getBoundingClientRect();
    const probe = box.top + Math.min(96, box.height * 0.3);
    let index = -1;
    let position = 0;
    for (const node of nodes) {
      if (node.getBoundingClientRect().top > probe) break;
      index = blockIndex(node, position);
      position += 1;
    }
    setActive(index < 0 ? blockIndex(nodes[0], 0) : index);
  });

  /* The transcript view replaces its scroller when the tab or view changes. */
  function attachScroller() {
    const host = document.getElementById("view-host");
    const tx = host ? host.querySelector(".tx-scroll, .tx") : null;
    if (tx === txEl) return false;
    if (txEl && txHandler) txEl.removeEventListener("scroll", txHandler);
    txEl = tx;
    txHandler = null;
    if (!txEl) {
      setActive(-1);
      return true;
    }
    txHandler = () => spy();
    txEl.addEventListener("scroll", txHandler, { passive: true });
    return true;
  }

  /* --- search ----------------------------------------------------------- */
  function step(delta) {
    const state = ctx.store.getState();
    const found = sel.searchHits(sel.transcriptBlocks(state), state.ui.search.q);
    if (!found.length) return;
    const current = state.ui.focusBlock ? found.indexOf(state.ui.focusBlock.index) : -1;
    const target =
      current < 0 ? (delta > 0 ? 0 : found.length - 1) : (current + delta + found.length) % found.length;
    if (target !== state.ui.search.focus) dispatch(act.search({ focus: target }));
    dispatch(act.focusBlock(found[target]));
  }

  function pushQuery(value) {
    clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      lastQ = value;
      dispatch(act.search({ q: value }));
    }, SEARCH_MS);
  }

  function clearQuery() {
    clearTimeout(timer);
    timer = null;
    lastQ = "";
    if (input.value) input.value = "";
    dispatch(act.search({ q: "" }));
  }

  input.addEventListener("input", () => pushQuery(input.value));
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      clearTimeout(timer);
      timer = null;
      if (input.value !== ctx.store.getState().ui.search.q) {
        lastQ = input.value;
        dispatch(act.search({ q: input.value }));
      }
      step(event.shiftKey ? -1 : 1);
    } else if (event.key === "Escape") {
      event.preventDefault();
      clearQuery();
    }
  });

  /* --- rows ------------------------------------------------------------- */
  function renderRow(block) {
    const error = block.isError === true;
    const title = sel.blockTitle(block);
    const meta = error ? "error" : block.kind === "thinking" && block.chars ? fmt.chars(block.chars) : "";
    const row = el(
      "button",
      { class: "outline-row", type: "button", title, dataset: { kind: block.kind, index: String(block.index) } },
      [
        el("span", { class: "outline-kind", "aria-hidden": "true" }, icon(KIND_ICON[block.kind] || "dot", 11)),
        el("span", { class: "outline-label truncate" }, title),
        meta ? el("span", { class: "outline-meta num" }, meta) : null,
      ],
    );
    if (error) row.dataset.error = "1";
    row.addEventListener("click", () => dispatch(act.focusBlock(block.index)));
    return row;
  }

  function appendRow(block) {
    const row = renderRow(block);
    rowByIndex.set(block.index, row);
    list.append(row);
  }

  function rebuild(blocks, visible) {
    clear(list);
    rowByIndex = new Map();
    activeRow = null;
    focusRow = null;
    rowCount = visible.length;
    if (!blocks.length) {
      list.append(el("p", { class: "outline-empty" }, "No turns yet."));
      return;
    }
    if (!visible.length) {
      list.append(el("p", { class: "outline-empty" }, "No matching turns."));
      return;
    }
    for (const block of visible) appendRow(block);
  }

  /* Appends are the common case (events stream in); rebuild only on a real change. */
  function syncList(blocks, visible, prefix) {
    const key = `${prefix}${visible.map((block) => block.index).join(",")}`;
    let changed = false;
    if (key !== listKey) {
      if (listKey && key.startsWith(`${listKey},`)) {
        for (let i = rowCount; i < visible.length; i += 1) appendRow(visible[i]);
        rowCount = visible.length;
      } else {
        rebuild(blocks, visible);
      }
      listKey = key;
      changed = true;
    }
    const tail = visible[visible.length - 1];
    const tailSig = tail
      ? `${prefix}${tail.kind}:${tail.chars || 0}:${tail.isError ? 1 : 0}:${tail.done ? 1 : 0}:${(tail.output || "").length}`
      : `${prefix}-`;
    if (tailSig !== tailKey) {
      tailKey = tailSig;
      if (tail) {
        const row = renderRow(tail);
        rowByIndex.set(tail.index, row);
        list.replaceChild(row, list.lastChild);
        changed = true;
      }
    }
    return changed;
  }

  function revealRow(row) {
    const box = list.getBoundingClientRect();
    const rect = row.getBoundingClientRect();
    if (rect.top < box.top) list.scrollTop -= box.top - rect.top;
    else if (rect.bottom > box.bottom) list.scrollTop += rect.bottom - box.bottom;
  }

  /* --- render ----------------------------------------------------------- */
  function syncQuery(q) {
    if (q === lastQ) return;
    lastQ = q;
    if (input.value !== q) input.value = q;
  }

  function render(state) {
    const attached = attachScroller();
    const q = state.ui.search.q;
    syncQuery(q);

    const blocks = sel.transcriptBlocks(state);
    const visible = sel.filterBlocks(blocks, state.ui.filter, q);
    const found = sel.searchHits(blocks, q);
    const focusBlock = state.ui.focusBlock;
    const focus = focusBlock ? focusBlock.index : -1;
    const focusKey = focus >= 0 ? `${focus}@${focusBlock.at}` : "";
    const prefix = `${state.transcript.source}|${state.ui.filter}|${q}|`;

    const chromeSig = [prefix, focusKey, state.ui.follow ? 1 : 0, found.length, visible.length].join("|");
    if (chromeSig !== chrome) {
      chrome = chromeSig;
      follow.setAttribute("aria-pressed", state.ui.follow ? "true" : "false");
      for (const [id, chip] of chipButtons) {
        chip.setAttribute("aria-pressed", state.ui.filter === id ? "true" : "false");
      }
      hits.hidden = !q;
      chips.hidden = !!q;
      setText(hitsLabel, fmt.plural(found.length, "hit"));
      prev.disabled = !found.length;
      next.disabled = !found.length;
      setText(count, visible.length === blocks.length ? String(blocks.length) : `${visible.length}/${blocks.length}`);
      count.title =
        visible.length === blocks.length
          ? `${blocks.length} turns`
          : `${visible.length} of ${blocks.length} turns`;
    }

    const changed = syncList(blocks, visible, prefix);
    const row = focus >= 0 ? rowByIndex.get(focus) || null : null;
    if (row !== focusRow) {
      if (focusRow) focusRow.dataset.focus = "0";
      focusRow = row;
      if (row) row.dataset.focus = "1";
    }
    if (row && (changed || focusKey !== revealed)) {
      revealed = focusKey;
      revealRow(row);
      setActive(focus);
    } else if (changed && state.ui.follow) {
      list.scrollTop = list.scrollHeight;
    }
    if (changed || attached) spy();
  }

  render(ctx.store.getState());
  return {
    update: render,
    destroy() {
      clearTimeout(timer);
      if (txEl && txHandler) txEl.removeEventListener("scroll", txHandler);
      txEl = null;
      txHandler = null;
    },
  };
}
