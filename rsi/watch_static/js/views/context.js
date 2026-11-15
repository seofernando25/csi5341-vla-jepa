/* Context: exactly what the agent received, one collapsible card per section.
   `payload.sections` is the source of truth; `files`/`command` are a legacy fallback. */

import { el, clear, icon, copyButton, prefersReducedMotion } from "../dom.js";
import * as sel from "../selectors.js";
import * as fmt from "../format.js";

const CAP = 40_000;
const START_CLOSED = new Set(["earlier", "evidence", "Earlier node material", "Replay evidence"]);

export function mount(root, ctx) {
  root.classList.add("context-root");
  const head = el("header", { class: "ctx-head" });
  const outline = el("nav", { class: "ctx-outline", aria: { label: "Context sections" } });
  const content = el("div", { class: "ctx-content", tabIndex: 0 });
  const body = el("div", { class: "ctx-body" }, [outline, content]);
  root.append(el("div", { class: "ctx" }, [head, body]));

  const closed = new Set();
  let nodes = new Map();
  let active = "";
  let sig = "";

  /* --- small builders --------------------------------------------------- */

  function copyWrap(getText, label) {
    const wrap = el("span", { class: "ctx-copy" }, copyButton(getText, label));
    wrap.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
    return wrap;
  }

  function cellButton(cell, cycle) {
    const button = el("button", { class: "ctx-cell", type: "button", title: `Open node ${cell}` }, cell);
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      ctx.go(sel.sourceOf("node", cycle, cell));
    });
    return button;
  }

  function sizeLabel(chars, tokens) {
    return `${fmt.chars(chars)} chars · ${fmt.tokens(tokens)}`;
  }

  function truncNote() {
    return el("p", { class: "ctx-trunc" }, `truncated at ${fmt.chars(CAP)} chars`);
  }

  function pre(text) {
    return el("pre", { class: "ctx-pre", tabIndex: 0 }, text);
  }

  function fileBlock(file) {
    const text = String(file.text ?? "");
    const block = el("div", { class: "ctx-file" });
    block.append(
      el("div", { class: "ctx-file-head" }, [
        icon("file", 11),
        el("span", { class: "ctx-file-name mono truncate" }, file.name || "file"),
        el("span", { class: "ctx-file-size num muted" }, sizeLabel(file.chars ?? text.length, file.tokens ?? Math.ceil(text.length / 4))),
        el("span", { class: "ctx-grow" }),
        copyWrap(() => text, "Copy file"),
      ]),
    );
    if (file.path) {
      block.append(el("div", { class: "ctx-file-path mono subtle truncate", title: file.path }, file.path));
    }
    if (file.truncated || text.length >= CAP) block.append(truncNote());
    block.append(text ? pre(text) : el("p", { class: "ctx-note" }, "Empty."));
    return block;
  }

  function itemFiles(item) {
    if (item.files?.length) return item.files;
    return [{ name: item.name || "item", text: item.text ?? "", chars: item.chars, tokens: item.tokens }];
  }

  function itemNode(item, cycle) {
    if (!item.cell_id) return fileBlock(itemFiles(item)[0]);
    const files = itemFiles(item);
    const chars = files.reduce((sum, file) => sum + Number(file.chars ?? (file.text || "").length), 0);
    const node = el("details", { class: "ctx-item", open: true });
    node.append(
      el("summary", { class: "ctx-item-head" }, [
        icon("chevronRight", 11),
        cellButton(item.cell_id, cycle),
        el("span", { class: "ctx-item-meta muted truncate" }, files.map((file) => file.name).join(" · ")),
        el("span", { class: "ctx-item-size num muted" }, fmt.chars(chars)),
        el("span", { class: "ctx-grow" }),
        copyWrap(() => files.map((file) => `# ${file.name}\n${file.text ?? ""}`).join("\n\n"), "Copy item"),
      ]),
    );
    const inner = el("div", { class: "ctx-item-body" });
    for (const file of files) inner.append(fileBlock(file));
    node.append(inner);
    return node;
  }

  function fieldsBlock(fields) {
    const list = el("dl", { class: "ctx-fields" });
    for (const [key, value] of Object.entries(fields)) {
      list.append(
        el("div", { class: "ctx-field" }, [
          el("dt", { class: "muted" }, key),
          el("dd", { class: "mono wrap-any" }, fieldValue(value)),
        ]),
      );
    }
    return list;
  }

  function fieldValue(value) {
    if (typeof value === "boolean") return value ? "yes" : "no";
    if (value == null) return "—";
    return typeof value === "object" ? JSON.stringify(value) : String(value);
  }

  function sectionSize(section) {
    const keys = Object.keys(section.fields || {});
    if (keys.length) return `${keys.length} fields`;
    return sizeLabel(section.chars, section.tokens);
  }

  function outlineSize(section) {
    if (section.recorded === false) return "—";
    const keys = Object.keys(section.fields || {});
    return keys.length ? String(keys.length) : fmt.chars(section.chars);
  }

  function sectionText(section) {
    if (section.recorded === false) return section.note || "";
    if (section.items?.length) {
      return section.items
        .map((item) => itemFiles(item).map((file) => `# ${file.name}\n${file.text ?? ""}`).join("\n\n"))
        .join("\n\n");
    }
    const keys = Object.keys(section.fields || {});
    if (keys.length) return keys.map((key) => `${key}: ${fieldValue(section.fields[key])}`).join("\n");
    return String(section.text ?? "");
  }

  function sectionNode(section, cycle) {
    const id = section.id || section.title;
    const node = el("details", { class: "ctx-section", dataset: { section: id } });
    node.open = !closed.has(id) && !START_CLOSED.has(id);
    node.addEventListener("toggle", () => {
      if (node.open) closed.delete(id);
      else closed.add(id);
    });
    node.append(
      el("summary", { class: "ctx-sec-head" }, [
        icon("chevronRight", 12),
        el("span", { class: "ctx-sec-title truncate" }, section.title || id),
        section.recorded === false
          ? el("span", { class: "ctx-badge" }, "not recorded")
          : el("span", { class: "ctx-sec-size num" }, sectionSize(section)),
        el("span", { class: "ctx-grow" }),
        section.recorded === false ? null : copyWrap(() => sectionText(section), "Copy section"),
      ]),
    );

    if (section.recorded === false) {
      node.append(el("p", { class: "ctx-note" }, section.note || "Not recorded."));
      return node;
    }
    const inner = el("div", { class: "ctx-sec-body" });
    if (section.note) inner.append(el("p", { class: "ctx-sec-note muted" }, section.note));
    if (section.path) inner.append(el("div", { class: "ctx-file-path mono subtle truncate", title: section.path }, section.path));
    if (section.items?.length) {
      for (const item of section.items) inner.append(itemNode(item, cycle));
    } else if (Object.keys(section.fields || {}).length) {
      inner.append(fieldsBlock(section.fields));
    } else {
      const text = String(section.text ?? "");
      if (section.truncated || text.length >= CAP) inner.append(truncNote());
      inner.append(text ? pre(text) : el("p", { class: "ctx-note" }, "Empty."));
    }
    node.append(inner);
    return node;
  }

  /* --- payload -> sections ---------------------------------------------- */

  function sectionsOf(payload) {
    const given = Array.isArray(payload.sections) ? payload.sections.filter(Boolean) : [];
    if (given.length) return given;
    const out = (payload.files || []).map((file) => {
      const text = String(file.text ?? "");
      return {
        id: `file-${file.name}`,
        title: file.name || "file",
        recorded: true,
        text,
        chars: text.length,
        tokens: Math.ceil(text.length / 4),
        path: file.path,
        note: null,
        items: [],
        fields: {},
        truncated: file.truncated === true,
      };
    });
    const fields = Object.fromEntries(Object.entries(payload.command || {}).filter(([, v]) => v != null));
    if (Object.keys(fields).length) {
      out.push({
        id: "command",
        title: "Agent configuration",
        recorded: true,
        text: null,
        chars: 0,
        tokens: 0,
        path: null,
        note: "From logs/command.json.",
        items: [],
        fields,
      });
    }
    return out;
  }

  /* --- header, outline, states ------------------------------------------ */

  function renderHead(parsed, payload) {
    const kind = parsed?.kind || payload?.kind || "node";
    head.append(
      el("div", { class: "ctx-head-row" }, [
        el("h2", { class: "ctx-title" }, "Context"),
        el("span", { class: "pill ctx-kind" }, kind === "policy" ? "policy step" : "node"),
        el("span", { class: "ctx-grow" }),
        payload?.log
          ? el("span", { class: "ctx-log mono truncate", title: payload.log }, payload.log)
          : null,
      ]),
    );
  }

  function setActive(id) {
    if (!id || id === active) return;
    active = id;
    for (const button of outline.querySelectorAll(".ctx-outline-item")) {
      const on = button.dataset.id === id;
      button.dataset.active = on ? "true" : "false";
      if (on) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
    }
  }

  function focusSection(id) {
    const node = nodes.get(id);
    if (!node) return;
    node.open = true;
    node.scrollIntoView({ block: "start", behavior: prefersReducedMotion() ? "auto" : "smooth" });
    setActive(id);
  }

  function renderOutline(sections) {
    outline.append(el("div", { class: "ctx-outline-head" }, "Sections"));
    const list = el("ol", { class: "ctx-outline-list" });
    for (const section of sections) {
      const id = section.id || section.title;
      const button = el(
        "button",
        { class: "ctx-outline-item", type: "button", dataset: { id }, title: section.title || id },
        [
          el("span", { class: "ctx-outline-title truncate" }, section.title || id),
          el("span", { class: "ctx-outline-size num" }, outlineSize(section)),
        ],
      );
      button.addEventListener("click", () => focusSection(id));
      list.append(el("li", {}, button));
    }
    outline.append(list);
  }

  function stateBlock(title, message) {
    body.dataset.state = "empty";
    content.append(
      el("div", { class: "ctx-state" }, [
        title ? el("strong", {}, title) : null,
        el("span", { class: "muted" }, message),
      ]),
    );
  }

  function onScroll() {
    if (!nodes.size) return;
    const top = content.scrollTop + 8;
    let current = active;
    for (const [id, node] of nodes) if (node.offsetTop <= top) current = id;
    setActive(current);
  }

  /* --- render ----------------------------------------------------------- */

  function render(state) {
    const source = sel.activeSource(state);
    const entry = source ? sel.contextFor(state, source) : null;
    const next = `${source}|${entry?.status || "none"}|${entry?.at || 0}`;
    if (next === sig) return;
    sig = next;

    const parsed = source ? sel.parseSource(source) : null;
    const payload = entry?.payload || null;
    clear(head);
    clear(outline);
    clear(content);
    nodes = new Map();
    active = "";
    body.dataset.state = "empty";
    renderHead(parsed, payload);

    if (!entry) {
      stateBlock("No context loaded", "Open a node.");
      return;
    }
    if (entry.status === "loading") {
      content.append(el("div", { class: "ctx-state" }, [el("span", { class: "spinner" }), el("span", {}, "Loading context…")]));
      return;
    }
    if (entry.status === "error") {
      content.append(
        el("div", { class: "ctx-state" }, [
          el("strong", {}, "Context unavailable"),
          el("span", { class: "mono wrap-any" }, entry.error || "unknown error"),
        ]),
      );
      return;
    }

    const sections = sectionsOf(payload || {});
    const given = Array.isArray(payload?.sections) ? payload.sections.filter(Boolean) : [];
    if (!given.length && !(payload?.files || []).length) {
      stateBlock(null, "No context recorded for this source.");
      return;
    }
    body.dataset.state = "ready";
    renderOutline(sections);
    const wrap = el("div", { class: "ctx-sections" });
    for (const section of sections) {
      const node = sectionNode(section, parsed?.cycle);
      nodes.set(section.id || section.title, node);
      wrap.append(node);
    }
    content.append(wrap);
    setActive(sections[0].id || sections[0].title);
    content.scrollTop = 0;
  }

  content.addEventListener("scroll", onScroll, { passive: true });
  render(ctx.store.getState());
  return {
    update: render,
    destroy() {
      content.removeEventListener("scroll", onScroll);
    },
  };
}
