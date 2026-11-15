/* Minimal markdown → DOM. Every character becomes a text node: no innerHTML, no raw HTML.
   Supports ATX headings, paragraphs, one-level lists, fenced code, inline code, bold,
   italic, safe links, blockquotes, rules and pipe tables. */

import { el } from "./dom.js";

/* The highlighter is optional: js/highlight.js is another module and may be absent.
   A missing or throwing highlighter degrades to plain text. */
let highlighter;
const highlighterReady = import("./highlight.js")
  .then((mod) => {
    highlighter = typeof mod.highlightInto === "function" ? mod : null;
  })
  .catch(() => {
    highlighter = null;
  });

const FENCE = /^ {0,3}(`{3,}|~{3,})\s*([\w+#.-]*)\s*$/;
const HEADING = /^ {0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$/;
const RULE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/;
const ITEM = /^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$/;
const QUOTE = /^ {0,3}>\s?(.*)$/;
const TABLE_SEP = /^ {0,3}\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$/;
const SCHEME = /^https?:\/\//i;

export function renderMarkdown(text) {
  const frag = document.createDocumentFragment();
  const lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
  for (const node of blocks(lines)) frag.append(node);
  return frag;
}

function blocks(lines) {
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }
    const fence = line.match(FENCE);
    if (fence) {
      const close = fence[1].slice(0, 3);
      const body = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith(close)) body.push(lines[i++]);
      i += 1;
      out.push(codeBlock(body.join("\n"), fence[2]));
      continue;
    }
    const heading = line.match(HEADING);
    if (heading) { out.push(el(`h${heading[1].length}`, {}, renderInline(heading[2]))); i += 1; continue; }
    if (RULE.test(line)) { out.push(el("hr")); i += 1; continue; }
    if (QUOTE.test(line)) {
      const body = [];
      while (i < lines.length && QUOTE.test(lines[i])) body.push(lines[i++].match(QUOTE)[1]);
      out.push(el("blockquote", {}, blocks(body)));
      continue;
    }
    if (isTable(lines, i)) {
      const [table, next] = tableBlock(lines, i);
      out.push(table);
      i = next;
      continue;
    }
    if (ITEM.test(line)) {
      const [list, next] = listBlock(lines, i);
      out.push(list);
      i = next;
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !startsBlock(lines, i)) para.push(lines[i++].trim());
    if (para.length) out.push(el("p", {}, renderInline(para.join(" "))));
    else i += 1;
  }
  return out;
}

function startsBlock(lines, i) {
  const line = lines[i];
  return FENCE.test(line) || HEADING.test(line) || RULE.test(line) || QUOTE.test(line) || ITEM.test(line) || isTable(lines, i);
}

function isTable(lines, i) {
  return Boolean(lines[i + 1] && lines[i].includes("|") && TABLE_SEP.test(lines[i + 1]));
}

function tableBlock(lines, start) {
  const table = el("table");
  const head = el("thead");
  head.append(tableRow("th", lines[start]));
  table.append(head);
  const body = el("tbody");
  let i = start + 2;
  while (i < lines.length && lines[i].trim() && lines[i].includes("|")) body.append(tableRow("td", lines[i++]));
  if (body.childNodes.length) table.append(body);
  return [el("div", { class: "md-table" }, table), i];
}

function tableRow(tag, line) {
  const row = el("tr");
  for (const cell of line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|")) {
    row.append(el(tag, {}, renderInline(cell.trim())));
  }
  return row;
}

function listBlock(lines, start) {
  const first = lines[start].match(ITEM);
  const ordered = /^\d/.test(first[2]);
  const base = first[1].length;
  const list = el(ordered ? "ol" : "ul");
  let item = null;
  let i = start;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      const next = lines[i + 1] && lines[i + 1].match(ITEM);
      if (!next || next[1].length < base) break;
      i += 1;
      continue;
    }
    const match = line.match(ITEM);
    if (match && match[1].length < base) break;
    if (match && match[1].length <= base + 1 && /^\d/.test(match[2]) !== ordered) break;
    if (match && match[1].length <= base + 1) {
      item = el("li", {}, renderInline(match[3]));
      list.append(item);
      i += 1;
      continue;
    }
    if (match) {
      const [sub, next] = listBlock(lines, i);
      (item || list).append(sub);
      i = next;
      continue;
    }
    if (item && /^\s{2,}/.test(line)) {
      item.append(" ", renderInline(line.trim()));
      i += 1;
      continue;
    }
    break;
  }
  return [list, i];
}

function codeBlock(code, lang) {
  const pre = el("pre", { class: "md-pre" });
  if (lang) pre.dataset.lang = lang;
  pre.textContent = code;
  const paint = () => {
    if (!highlighter) return;
    try {
      highlighter.highlightInto(pre, code, lang);
    } catch {
      pre.textContent = code;
    }
  };
  if (highlighter !== undefined) paint();
  else highlighterReady.then(paint);
  return pre;
}

export function renderInline(text) {
  const frag = document.createDocumentFragment();
  const src = String(text ?? "");
  let plain = "";
  let i = 0;
  const flush = () => {
    if (plain) frag.append(document.createTextNode(plain));
    plain = "";
  };
  while (i < src.length) {
    const ch = src[i];
    if (ch === "`") {
      const end = src.indexOf("`", i + 1);
      if (end > i + 1) {
        flush();
        frag.append(el("code", {}, src.slice(i + 1, end)));
        i = end + 1;
        continue;
      }
    } else if (ch === "*" && src[i + 1] === "*") {
      const end = src.indexOf("**", i + 2);
      if (end > i + 2) {
        flush();
        frag.append(el("strong", {}, renderInline(src.slice(i + 2, end))));
        i = end + 2;
        continue;
      }
    } else if (ch === "*" || ch === "_") {
      const opens = ch === "*" || !(i > 0 && /\w/.test(src[i - 1]));
      const end = opens ? italicEnd(src, i, ch) : -1;
      if (end > i + 1) {
        flush();
        frag.append(el("em", {}, renderInline(src.slice(i + 1, end))));
        i = end + 1;
        continue;
      }
    } else if (ch === "[") {
      const link = linkAt(src, i);
      if (link) {
        const url = safeUrl(link.url);
        const external = url && !url.startsWith("#");
        flush();
        frag.append(
          url
            ? el("a", { href: url, target: external ? "_blank" : null, rel: external ? "noopener noreferrer" : null }, renderInline(link.label))
            : renderInline(link.label || link.url),
        );
        i += link.length;
        continue;
      }
    }
    plain += ch;
    i += 1;
  }
  flush();
  return frag;
}

function italicEnd(src, start, marker) {
  if (/\s/.test(src[start + 1] || " ")) return -1;
  for (let i = start + 1; i < src.length; i += 1) {
    if (src[i] !== marker) continue;
    if (marker === "*" && (src[i + 1] === "*" || src[i - 1] === "*")) continue;
    if (marker === "_" && /\w/.test(src[i + 1] || "")) continue;
    if (i === start + 1 || /\s/.test(src[i - 1])) continue;
    return i;
  }
  return -1;
}

function linkAt(src, start) {
  const close = src.indexOf("]", start + 1);
  if (close < 0 || src[close + 1] !== "(") return null;
  let depth = 1;
  let i = close + 2;
  for (; i < src.length; i += 1) {
    if (src[i] === "\n") return null;
    if (src[i] === "(") depth += 1;
    else if (src[i] === ")" && --depth === 0) break;
  }
  if (depth !== 0) return null;
  return { label: src.slice(start + 1, close), url: src.slice(close + 2, i), length: i - start + 1 };
}

function safeUrl(raw) {
  const url = String(raw || "").trim();
  if (!url) return null;
  return SCHEME.test(url) || url.startsWith("#") ? url : null;
}
