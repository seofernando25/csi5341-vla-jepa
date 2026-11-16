/* Diff: unified patch for a node, seed -> candidate for a policy step. */

import { el, clear, icon } from "../dom.js";
import * as sel from "../selectors.js";
import { parsePatch, diffLines, fileStats } from "../diff.js";
import { highlight } from "../highlight.js";

const EMPTY_NODE = "No changes recorded against the parent.";
const EMPTY_POLICY = "No candidate code recorded.";
const EMPTY_SEED = "No changes against the seed.";

export function mount(root, ctx) {
  const view = el("div", { class: "code-view diff-view" });
  const files = el("div", { class: "diff-files" });
  const head = el("div", { class: "diff-head" });
  const scroll = el("div", { class: "diff-scroll" });
  view.append(files, el("div", { class: "diff-main" }, [head, scroll]));
  root.append(view);

  let signature = "";
  let items = [];
  let sections = [];
  let current = -1;
  let note = null;
  let ticking = false;
  let dead = false;

  function setCurrent(index) {
    if (!items.length) return;
    current = Math.max(0, Math.min(items.length - 1, index));
    const item = items[current];
    clear(head);
    const parts = [
      icon("file", 13),
      el("span", { class: "diff-path mono truncate", title: item.file.path }, item.file.path),
      statsNode(item.stats),
    ];
    if (note) parts.push(el("span", { class: "diff-note" }, note));
    head.append(...parts);
    files.querySelectorAll(".diff-file-item").forEach((button, index2) => {
      button.setAttribute("aria-current", index2 === current ? "true" : "false");
    });
  }

  function goTo(index) {
    const section = sections[index];
    if (!section) return;
    scroll.scrollTop = section.offsetTop;
    setCurrent(index);
  }

  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      if (dead || !sections.length) return;
      const top = scroll.scrollTop + 1;
      let index = 0;
      for (let i = 0; i < sections.length; i++) if (sections[i].offsetTop <= top) index = i;
      if (index !== current) setCurrent(index);
    });
  }
  scroll.addEventListener("scroll", onScroll, { passive: true });

  function paint(list, options = {}) {
    note = options.note || null;
    delete view.dataset.state;
    items = list.map((file) => ({
      file,
      stats: fileStats(file),
      lang: options.lang || langFor(file.path),
    }));
    clear(files);
    files.append(
      el("div", { class: "diff-files-head" }, [
        el("span", {}, "Files"),
        el("span", { class: "count" }, String(items.length)),
      ]),
    );
    sections = [];
    items.forEach((item, index) => {
      const button = el(
        "button",
        { class: "diff-file-item", type: "button", title: item.file.path, "aria-current": "false" },
        [
          el("span", { class: "diff-file-name truncate" }, item.file.path),
          el("span", { class: "diff-file-stats num" }, [
            el("span", { class: "adds" }, `+${item.stats.adds}`),
            el("span", { class: "dels" }, `-${item.stats.dels}`),
          ]),
        ],
      );
      button.addEventListener("click", () => goTo(index));
      files.append(button);

      const section = el("section", { class: "diff-file", dataset: { index: String(index) } });
      if (items.length > 1) {
        section.append(
          el("div", { class: "diff-file-title" }, [
            el("span", { class: "mono" }, item.file.path),
            el("span", { class: "diff-file-stats num" }, [
              el("span", { class: "adds" }, `+${item.stats.adds}`),
              el("span", { class: "dels" }, `-${item.stats.dels}`),
            ]),
          ]),
        );
      }
      for (const hunk of item.file.hunks) section.append(hunkNode(hunk, item.lang));
      scroll.append(section);
      sections.push(section);
    });
    setCurrent(0);
  }

  function show(node) {
    view.dataset.state = "message";
    scroll.append(node);
  }

  function empty(text) {
    show(el("p", { class: "empty" }, [el("strong", {}, text)]));
  }

  function loading(text) {
    show(el("p", { class: "empty" }, [el("span", { class: "spinner" }), text]));
  }

  function failed(message) {
    show(el("p", { class: "empty" }, [el("strong", {}, "Diff unavailable"), message]));
  }

  function render(state) {
    const tab = sel.activeTab(state);
    const input = tab ? read(state, tab) : null;
    const next = signatureOf(input);
    if (next === signature) return;
    signature = next;

    clear(files);
    clear(head);
    clear(scroll);
    items = [];
    sections = [];
    current = -1;
    note = null;

    if (!input) return empty("No editor open");

    if (input.kind === "node") {
      const entry = input.diff;
      if (!entry || entry.status === "loading") return loading("Loading diff…");
      if (entry.status === "error") {
        return notFound(entry.error) ? empty(EMPTY_NODE) : failed(entry.error);
      }
      const list = parsePatch(entry.text || "").files.filter((file) => file.hunks.length);
      if (!list.length) return empty(EMPTY_NODE);
      return paint(list);
    }

    const candidate = input.candidate;
    if (!candidate || candidate.status === "loading") return loading("Loading candidate…");
    if (candidate.status === "error") {
      return notFound(candidate.error) ? empty(EMPTY_POLICY) : failed(candidate.error);
    }
    const seed = input.seed;
    if (!seed || seed.status === "loading") return loading("Loading seed…");
    const seedText = seed.status === "ready" ? seed.text || "" : "";
    const missingSeed = seed.status === "error";
    const file = diffLines(seedText, candidate.text || "", { path: "candidate.py" });
    if (!file.hunks.length) return empty(missingSeed ? EMPTY_POLICY : EMPTY_SEED);
    paint([file], { lang: "python", note: missingSeed ? "seed not recorded" : null });
  }

  render(ctx.store.getState());
  return {
    update: render,
    destroy() {
      dead = true;
      scroll.removeEventListener("scroll", onScroll);
    },
  };
}

function read(state, tab) {
  if (tab.kind === "policy") {
    return {
      source: tab.id,
      kind: "policy",
      seed: sel.artifact(state, tab.id, "seed"),
      candidate: sel.artifact(state, tab.id, "policy"),
    };
  }
  return { source: tab.id, kind: "node", diff: sel.artifact(state, tab.id, "diff") };
}

function signatureOf(input) {
  if (!input) return "none";
  const part = (entry) => (entry ? `${entry.status}:${entry.at ?? 0}` : "-");
  if (input.kind === "policy") {
    return `${input.source}|${input.kind}|${part(input.seed)}|${part(input.candidate)}`;
  }
  return `${input.source}|${input.kind}|${part(input.diff)}`;
}

function statsNode(stats) {
  return el("span", { class: "diff-stats num" }, [
    el("span", { class: "adds" }, `+${stats.adds}`),
    el("span", { class: "dels" }, `-${stats.dels}`),
  ]);
}

function hunkNode(hunk, lang) {
  const node = el("div", { class: "diff-hunk" });
  node.append(el("div", { class: "diff-hunk-head" }, hunk.header));
  for (const line of hunk.lines) node.append(lineNode(line, lang));
  return node;
}

function lineNode(line, lang) {
  const sign = line.kind === "add" ? "+" : line.kind === "del" ? "-" : line.kind === "ctx" ? " " : "";
  return el("div", { class: "diff-line", dataset: { kind: line.kind } }, [
    el("span", { class: "ln old num" }, line.oldNo == null ? "" : String(line.oldNo)),
    el("span", { class: "ln new num" }, line.newNo == null ? "" : String(line.newNo)),
    el("span", { class: "diff-sign", "aria-hidden": "true" }, sign),
    el(
      "span",
      { class: "diff-text" },
      line.kind === "meta" ? line.text : highlight(line.text, lang),
    ),
  ]);
}

function langFor(path) {
  const name = String(path || "");
  if (name.endsWith(".py")) return "python";
  if (name.endsWith(".json")) return "json";
  return "text";
}

function notFound(message) {
  return /not found|404/i.test(String(message || ""));
}
