/* Transcript: turn-structured feed for one node or policy step.
   Blocks arrive grouped from the selectors. This view renders them incrementally: a delta only
   replaces the block that grew, new blocks append, and only a source/filter/search change
   rebuilds the feed. Reasoning bodies are filled on first open, so a 10k block costs one line. */

import { el, clear, icon, setText, copyButton, prefersReducedMotion } from "../dom.js";
import { renderMarkdown } from "../markdown.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";

const OUT_LINES = 40;
const BOTTOM_SLACK = 24;
const TOOL_ICONS = {
  bash: "terminal",
  shell: "terminal",
  read: "file",
  write: "file",
  edit: "file",
  grep: "search",
  glob: "search",
  ls: "list",
  task: "list",
};

/* js/markdown.js renders assistant prose; a throwing renderer degrades to <pre>. */

export function mount(root, ctx) {
  const { store, dispatch, act } = ctx;
  root.classList.add("transcript-root");

  /* --- chrome ------------------------------------------------------------ */
  const srcLabel = el("span", { class: "tx-head-src mono truncate" });
  const countLabel = el("span", { class: "tx-head-count num" });
  const stateText = el("span", { class: "tx-head-state-text" });
  const stateDot = el("span", { class: "dot" });
  const statePill = el("span", { class: "tx-head-state" }, [stateDot, stateText]);
  const copy = copyButton(() => transcriptText(store.getState()), "Copy");
  copy.title = "Copy transcript";
  const head = el("div", { class: "tx-head" }, [
    srcLabel,
    countLabel,
    el("span", { class: "tx-head-spacer" }),
    statePill,
    copy,
  ]);

  const feed = el("div", { class: "tx-feed" });
  const scroller = el("div", { class: "tx-scroll", role: "log",
      "aria-live": "off", "aria-label": "Transcript", tabIndex: 0 }, [
    feed,
  ]);
  const jump = el("button", { class: "tx-jump", type: "button", hidden: true }, [
    icon("arrowDown", 12),
    "Jump to latest",
  ]);
  root.append(el("div", { class: "tx-view" }, [head, scroller, jump]));

  /* --- view state -------------------------------------------------------- */
  let sig = "";
  let viewKey = null;
  let renderedSigs = [];
  let renderedEls = [];
  let unseen = false;
  let following = true;
  let flashEl = null;
  let lastFocusAt = 0;
  let lastEventCount = 0;

  /* --- block builders ---------------------------------------------------- */
  function buildBlock(block) {
    const row = el("article", {
      class: `tx-block tx-${block.kind}`,
      dataset: { kind: block.kind, block: String(block.index) },
    });
    if (block.kind === "thinking") row.append(reasoning(block));
    else if (block.kind === "text") row.append(message(block));
    else if (block.kind === "tool") row.append(toolRow(block, false));
    else if (block.kind === "toolresult") row.append(toolRow(block, true));
    else if (block.kind === "system") row.append(systemLine(block));
    else if (block.kind === "result") row.append(resultCard(block));
    return row;
  }

  function reasoning(block) {
    const body = el("pre", { class: "tx-think-body" });
    const details = el("details", { class: "tx-think" }, [
      el("summary", { class: "tx-think-sum" }, [
        el("span", { class: "tx-chev" }, icon("chevronRight", 11)),
        el("span", { class: "tx-think-label" }, "Reasoning"),
        el("span", { class: "tx-think-title truncate" }, sel.blockTitle(block)),
        el("span", { class: "tx-think-meta num" }, `${fmt.chars(block.chars)} chars`),
      ]),
      body,
    ]);
    details.addEventListener("toggle", () => {
      if (details.open && !body.dataset.filled) {
        body.textContent = block.text;
        body.dataset.filled = "1";
      }
    });
    return details;
  }

  function message(block) {
    const body = el("div", { class: "tx-md" });
    try {
      const fragment = renderMarkdown(block.text);
      if (fragment && fragment.childNodes.length) {
        body.append(fragment);
        return body;
      }
    } catch {
      /* fall through to the plain fallback */
    }
    body.append(el("pre", { class: "tx-plain" }, block.text));
    return body;
  }

  function toolSummary(block, orphan) {
    if (orphan) return fmt.firstLine(block.output, 160);
    const raw = String(block.summary || block.tool || "");
    const name = String(block.tool || "");
    if (name && raw.toLowerCase().startsWith(name.toLowerCase())) {
      const rest = raw.slice(name.length).replace(/^[\s:·—-]+/, "");
      return rest || "";
    }
    return raw;
  }

  function toolRow(block, orphan) {
    const running = !orphan && !block.done;
    const state = block.isError ? "error" : running ? "running" : "done";
    const summary = toolSummary(block, orphan);
    const glyph = TOOL_ICONS[String(block.tool || "").toLowerCase()] || "code";
    return el("details", { class: "tx-tool", dataset: { state, orphan: orphan ? "1" : "0" } }, [
      el("summary", { class: "tx-tool-row" }, [
        el("span", { class: "tx-tool-glyph" }, icon(glyph, 12)),
        el("span", { class: "tx-tool-name mono" }, block.tool || "tool"),
        el("span", { class: "tx-tool-sum mono truncate", title: summary || undefined }, summary || "—"),
        el("span", { class: "tx-tool-state" }, state),
      ]),
      toolBody(block, orphan, running),
    ]);
  }

  function toolBody(block, orphan, running) {
    const body = el("div", { class: "tx-tool-body" });
    if (!orphan && block.input && Object.keys(block.input).length) {
      body.append(section("input", json(block.input)));
    }
    if (running) body.append(el("p", { class: "tx-tool-note" }, "running…"));
    else body.append(section("output", block.output || "(no output recorded)"));
    return body;
  }

  function section(label, text) {
    const lines = String(text).split("\n");
    const pre = el("pre", { class: "tx-sec-pre" }, lines.slice(0, OUT_LINES).join("\n"));
    const wrap = el("div", { class: "tx-sec" }, [el("div", { class: "tx-sec-label" }, label), pre]);
    if (lines.length > OUT_LINES) {
      const more = el("button", { class: "tx-more", type: "button" }, `Show all ${lines.length} lines`);
      more.addEventListener("click", () => {
        pre.textContent = text;
        more.remove();
      });
      wrap.append(more);
    }
    return wrap;
  }

  function json(value) {
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }

  function systemLine(block) {
    return el("div", { class: "tx-system" }, [
      el("span", { class: "tx-system-dot" }),
      el("span", { class: "tx-system-text" }, block.text),
    ]);
  }

  function resultCard(block) {
    const meta = block.meta || {};
    const bits = [];
    if (meta.num_turns != null) bits.push(fmt.plural(meta.num_turns, "turn"));
    if (meta.cost_usd != null) bits.push(fmt.usd(meta.cost_usd));
    if (meta.duration_s != null) bits.push(fmt.dur(meta.duration_s));
    return el("div", { class: "tx-result", dataset: { error: block.isError ? "1" : "0" } }, [
      el("div", { class: "tx-result-head" }, [
        icon(block.isError ? "alert" : "check", 12),
        el("span", {}, "result"),
        block.isError ? el("span", { class: "tx-badge" }, "error") : null,
        bits.length ? el("span", { class: "tx-result-meta num" }, bits.join(" · ")) : null,
      ]),
      el("p", { class: "tx-result-text" }, block.text || "(no final text)"),
    ]);
  }

  /* --- incremental feed -------------------------------------------------- */
  function blockSig(block) {
    if (block.kind === "tool") {
      return `t${block.i0}:${block.i1}:${block.done ? 1 : 0}:${block.isError ? 1 : 0}:${(block.output || "").length}:${block.summary || ""}`;
    }
    if (block.kind === "toolresult") return `r${block.i0}:${block.isError ? 1 : 0}:${(block.output || "").length}`;
    if (block.kind === "result") return `z${block.i0}:${block.isError ? 1 : 0}:${(block.text || "").length}`;
    if (block.kind === "system") return `s${block.i0}:${(block.text || "").length}`;
    return `${block.kind}${block.i0}:${block.i1}:${block.chars}`;
  }

  function rebuild(blocks) {
    clear(feed);
    renderedEls = [];
    renderedSigs = [];
    for (const block of blocks) {
      const node = buildBlock(block);
      feed.append(node);
      renderedEls.push(node);
      renderedSigs.push(blockSig(block));
    }
  }

  function replaceAt(index, node) {
    const old = renderedEls[index];
    if (!old) {
      feed.append(node);
    } else {
      const wasOpen = old.querySelector("details")?.open === true;
      old.replaceWith(node);
      const next = node.querySelector("details");
      if (wasOpen && next) next.open = true;
    }
    renderedEls[index] = node;
  }

  function sync(blocks) {
    const sigs = blocks.map(blockSig);
    const limit = Math.min(sigs.length, renderedSigs.length);
    let same = 0;
    while (same < limit && sigs[same] === renderedSigs[same]) same += 1;
    if (same === sigs.length && same === renderedSigs.length) return false;
    if (sigs.length < renderedSigs.length || same < renderedSigs.length - 1) {
      rebuild(blocks);
      return true;
    }
    for (let index = same; index < sigs.length; index += 1) replaceAt(index, buildBlock(blocks[index]));
    renderedEls.length = sigs.length;
    renderedSigs = sigs;
    return true;
  }

  function stateNode(kind, state) {
    if (kind === "loading") {
      return el("div", { class: "tx-state tx-state-loading" }, [
        el("span", { class: "spinner" }),
        el("span", {}, "Loading transcript…"),
      ]);
    }
    if (kind === "error") {
      return el("div", { class: "tx-state tx-state-error" }, [
        el("strong", {}, "Stream error"),
        el("p", { class: "wrap-any" }, state.transcript.error || state.transport.error || "The event stream stopped."),
      ]);
    }
    if (kind === "filtered") {
      const button = el("button", { class: "btn", type: "button" }, "Clear filter");
      button.addEventListener("click", () => {
        dispatch(act.filter("all"));
        dispatch(act.search({ q: "" }));
      });
      return el("div", { class: "tx-state" }, [
        el("strong", {}, "No matches"),
        el("p", {}, "No block matches the current filter or search."),
        button,
      ]);
    }
    return el("div", { class: "tx-state" }, [
      el("strong", {}, "No transcript"),
      el("p", {}, "No transcript was recorded for this source."),
    ]);
  }

  function showState(kind, state) {
    clear(feed);
    renderedEls = [];
    renderedSigs = [];
    viewKey = null;
    unseen = false;
    setJump(false);
    feed.append(stateNode(kind, state));
  }

  /* --- header, follow, focus --------------------------------------------- */
  function renderHead(state, events, blocks, shown) {
    setText(srcLabel, state.transcript.source || "no source");
    srcLabel.title = state.transcript.source || "";
    const narrowed = state.ui.filter !== "all" || Boolean(state.ui.search.q);
    setText(countLabel, narrowed ? `${shown}/${blocks} blocks` : fmt.plural(events, "event"));
    const label = sel.connectionLabel(state);
    setText(stateText, label);
    const tone = label === "live" ? "live" : label === "stream error" ? "err" : label === "final" ? "ok" : "";
    if (tone) {
      statePill.dataset.tone = tone;
      stateDot.dataset.tone = tone;
    } else {
      delete statePill.dataset.tone;
      delete stateDot.dataset.tone;
    }
    statePill.title = state.transport.reason ? `${label} · ${state.transport.reason}` : label;
  }

  function pin() {
    scroller.scrollTop = scroller.scrollHeight;
  }

  function setJump(show) {
    jump.hidden = !show;
  }

  function onScroll() {
    const gap = scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop;
    if (gap <= BOTTOM_SLACK) {
      if (unseen) {
        unseen = false;
        setJump(false);
      }
      return;
    }
    if (store.getState().ui.follow) dispatch(act.follow(false));
  }

  function flash(node) {
    if (flashEl && flashEl !== node) flashEl.classList.remove("tx-flash");
    flashEl = node;
    node.classList.remove("tx-flash");
    void node.offsetWidth;
    node.classList.add("tx-flash");
    node.addEventListener(
      "animationend",
      () => {
        node.classList.remove("tx-flash");
        if (flashEl === node) flashEl = null;
      },
      { once: true },
    );
  }

  function focusBlock(state) {
    const target = state.ui.focusBlock;
    if (!target || target.at === lastFocusAt) return;
    lastFocusAt = target.at;
    let node = feed.querySelector(`[data-block="${Number(target.index)}"]`);
    if (!node) {
      node =
        renderedEls.find((item) => Number(item.dataset.block) >= target.index) ||
        renderedEls[renderedEls.length - 1] ||
        null;
    }
    if (!node) return;
    node.scrollIntoView({ block: "center", behavior: prefersReducedMotion() ? "auto" : "smooth" });
    flash(node);
  }

  /* --- render ------------------------------------------------------------ */
  function render(state) {
    const transcript = state.transcript;
    const blocks = sel.transcriptBlocks(state);
    const filtered = sel.filterBlocks(blocks, state.ui.filter, state.ui.search.q);
    renderHead(state, transcript.events.length, blocks.length, filtered.length);

    const key = `${transcript.source}|${state.ui.filter}|${state.ui.search.q}`;
    const newEvents = transcript.events.length > lastEventCount;
    lastEventCount = transcript.events.length;
    let grew = false;
    if (transcript.status === "loading") showState("loading", state);
    else if (transcript.status === "error") showState("error", state);
    else if (!blocks.length) showState("empty", state);
    else if (!filtered.length) showState("filtered", state);
    else if (key !== viewKey) {
      rebuild(filtered);
      viewKey = key;
      grew = true;
    } else {
      grew = sync(filtered);
    }

    if (state.ui.follow) {
      if (grew || !following) pin();
      if (unseen) {
        unseen = false;
        setJump(false);
      }
    } else if (grew && newEvents) {
      unseen = true;
      setJump(true);
    }
    following = state.ui.follow;
    focusBlock(state);
  }

  function transcriptText(state) {
    const blocks = sel.transcriptBlocks(state);
    return blocks
      .map((block) => {
        if (block.kind === "thinking") return `[reasoning] ${block.text}`;
        if (block.kind === "text") return block.text;
        if (block.kind === "tool") {
          return `[${block.tool}] ${toolSummary(block, false)}${block.output ? `\n${block.output}` : ""}`;
        }
        if (block.kind === "toolresult") {
          return `[${block.tool} result]${block.isError ? " (error)" : ""}\n${block.output || ""}`;
        }
        if (block.kind === "system") return `[system] ${block.text}`;
        if (block.kind === "result") return `[result]${block.isError ? " (error)" : ""} ${block.text || ""}`;
        return "";
      })
      .join("\n\n");
  }

  function sigOf(state) {
    const transcript = state.transcript;
    const ui = state.ui;
    const transport = state.transport;
    const focus = ui.focusBlock ? `${ui.focusBlock.index}@${ui.focusBlock.at}` : "";
    return [
      transcript.source,
      transcript.events.length,
      transcript.at,
      transcript.status,
      transcript.error || "",
      ui.filter,
      ui.search.q,
      ui.follow ? 1 : 0,
      focus,
      transport.status,
      transport.frozen ? 1 : 0,
      transport.reason || "",
    ].join("|");
  }

  function update(state) {
    const next = sigOf(state);
    if (next === sig) return;
    sig = next;
    render(state);
  }

  jump.addEventListener("click", () => {
    unseen = false;
    setJump(false);
    dispatch(act.follow(true));
    scroller.scrollTo({
      top: scroller.scrollHeight,
      behavior: prefersReducedMotion() ? "auto" : "smooth",
    });
  });
  scroller.addEventListener("scroll", onScroll, { passive: true });

  const initial = store.getState();
  following = initial.ui.follow;
  render(initial);
  sig = sigOf(initial);

  return {
    update,
    destroy() {
      scroller.removeEventListener("scroll", onScroll);
    },
  };
}
