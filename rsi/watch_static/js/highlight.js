/* Tiny tokenizers. No dependencies, no markup strings: spans are built with createElement.
 *
 * highlight(text, lang)        -> DocumentFragment  (lang: python | json | diff | text)
 * highlightInto(node, text, lang) -> void
 * highlightLines(text, lang)   -> DocumentFragment[]  (one fragment per line, for gutters)
 *
 * Input is capped at 200k characters, matching the artifact cap in rsi/watch.py.
 */

import { el, clear } from "./dom.js";

const CAP = 200_000;

const PY_KEYWORDS = new Set(
  (
    "False None True and as assert async await break class continue def del elif else except " +
    "finally for from global if import in is lambda nonlocal not or pass raise return try " +
    "while with yield match case"
  ).split(" "),
);

const PY_BUILTINS = new Set(
  (
    "abs all any bool bytes callable chr dict dir divmod enumerate eval filter float format " +
    "frozenset getattr globals hasattr hash hex id input int isinstance issubclass iter len " +
    "list map max min next object oct open ord pow print property range repr reversed round " +
    "set setattr slice sorted staticmethod str sum super tuple type vars zip " +
    "Exception BaseException ValueError TypeError KeyError IndexError RuntimeError " +
    "NotImplementedError AttributeError ImportError OSError StopIteration"
  ).split(" "),
);

const NUMBER_RE =
  /(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?\d+)?)[jJ]?/y;

const OPERATORS = "+-*/%=<>!&|^~:@";

export function highlight(text, lang = "text") {
  return build(tokenize(String(text ?? "").slice(0, CAP), lang));
}

export function highlightInto(node, text, lang = "text") {
  clear(node);
  node.append(highlight(text, lang));
}

export function highlightLines(text, lang = "text") {
  const source = String(text ?? "").slice(0, CAP);
  const rows = [];
  let row = [];
  const flush = () => {
    rows.push(row);
    row = [];
  };
  for (const token of tokenize(source, lang)) {
    const parts = token.text.split("\n");
    parts.forEach((part, index) => {
      if (index) flush();
      if (part) row.push({ cls: token.cls, kind: token.kind, text: part });
    });
  }
  rows.push(row);
  if (source.endsWith("\n") && rows.length > 1 && rows[rows.length - 1].length === 0) rows.pop();
  return rows.map(build);
}

/* --- tokenizers ---------------------------------------------------------- */

function tokenize(text, lang) {
  if (lang === "python") return pythonTokens(text);
  if (lang === "json") return jsonTokens(text);
  if (lang === "diff") return diffTokens(text);
  return text ? [{ cls: null, text }] : [];
}

function pythonTokens(text) {
  const out = [];
  const n = text.length;
  let i = 0;
  let after = ""; // "def" | "class": the next identifier is the declared name
  const push = (cls, value) => {
    if (value) out.push({ cls, text: value });
  };

  while (i < n) {
    const ch = text[i];
    if (ch === "\n") {
      push(null, "\n");
      i++;
      after = "";
      continue;
    }
    if (ch === " " || ch === "\t" || ch === "\r") {
      let j = i + 1;
      while (j < n && (text[j] === " " || text[j] === "\t" || text[j] === "\r")) j++;
      push(null, text.slice(i, j));
      i = j;
      continue;
    }
    if (ch === "#") {
      let j = text.indexOf("\n", i);
      if (j < 0) j = n;
      push("com", text.slice(i, j));
      i = j;
      continue;
    }
    if (ch === "@" && /[A-Za-z_]/.test(text[i + 1] || "")) {
      let j = i + 1;
      while (j < n && /[\w.]/.test(text[j])) j++;
      push("op", "@");
      push("fn", text.slice(i + 1, j));
      i = j;
      continue;
    }
    if (ch === '"' || ch === "'") {
      const j = scanString(text, i);
      push("str", text.slice(i, j));
      i = j;
      continue;
    }
    if (/[0-9]/.test(ch) || (ch === "." && /[0-9]/.test(text[i + 1] || ""))) {
      NUMBER_RE.lastIndex = i;
      const match = NUMBER_RE.exec(text);
      if (match && match.index === i) {
        push("num", match[0]);
        i += match[0].length;
        continue;
      }
    }
    if (/[A-Za-z_]/.test(ch)) {
      let j = i + 1;
      while (j < n && /[A-Za-z0-9_]/.test(text[j])) j++;
      const word = text.slice(i, j);
      if (j < n && (text[j] === '"' || text[j] === "'") && /^[rRbBuUfF]{1,2}$/.test(word)) {
        const k = scanString(text, j);
        push("str", text.slice(i, k));
        i = k;
        continue;
      }
      if (after === "def") {
        push("fn", word);
        after = "";
      } else if (after === "class") {
        push("type", word);
        after = "";
      } else if (PY_KEYWORDS.has(word)) {
        push("key", word);
        if (word === "def" || word === "class") after = word;
      } else if (PY_BUILTINS.has(word)) {
        push("type", word);
      } else if (text[j] === "(") {
        push("fn", word);
      } else {
        push(null, word);
      }
      i = j;
      continue;
    }
    let j = i + 1;
    while (j < n && OPERATORS.includes(text[j])) j++;
    push("op", text.slice(i, j));
    i = j;
  }
  return out;
}

function jsonTokens(text) {
  const out = [];
  const n = text.length;
  let i = 0;
  while (i < n) {
    const ch = text[i];
    if (ch === '"') {
      const j = scanString(text, i);
      let k = j;
      while (k < n && /\s/.test(text[k])) k++;
      out.push({ cls: text[k] === ":" ? "key" : "str", text: text.slice(i, j) });
      i = j;
      continue;
    }
    if (/[-0-9]/.test(ch)) {
      NUMBER_RE.lastIndex = i;
      const match = NUMBER_RE.exec(text);
      if (match && match.index === i) {
        out.push({ cls: "num", text: match[0] });
        i += match[0].length;
        continue;
      }
    }
    if (/[A-Za-z]/.test(ch)) {
      let j = i + 1;
      while (j < n && /[A-Za-z]/.test(text[j])) j++;
      const word = text.slice(i, j);
      const literal = word === "true" || word === "false" || word === "null";
      out.push({ cls: literal ? "num" : null, text: word });
      i = j;
      continue;
    }
    if (/\s/.test(ch)) {
      let j = i + 1;
      while (j < n && /\s/.test(text[j])) j++;
      out.push({ cls: null, text: text.slice(i, j) });
      i = j;
      continue;
    }
    out.push({ cls: "op", text: ch });
    i++;
  }
  return out;
}

function diffTokens(text) {
  const out = [];
  const lines = text.split("\n");
  lines.forEach((line, index) => {
    const kind =
      line.startsWith("+++") || line.startsWith("---") || line.startsWith("diff ") || line.startsWith("index ")
        ? "meta"
        : line.startsWith("@@")
          ? "hunk"
          : line.startsWith("+")
            ? "add"
            : line.startsWith("-")
              ? "del"
              : "ctx";
    out.push({ cls: "diff", kind, text: line + (index < lines.length - 1 ? "\n" : "") });
  });
  return out;
}

/* --- helpers ------------------------------------------------------------- */

function scanString(text, start) {
  const quote = text[start];
  const triple = text.startsWith(quote.repeat(3), start);
  const delim = triple ? quote.repeat(3) : quote;
  const n = text.length;
  let i = start + delim.length;
  while (i < n) {
    const ch = text[i];
    if (ch === "\\") {
      i += 2;
      continue;
    }
    if (text.startsWith(delim, i)) return i + delim.length;
    if (!triple && ch === "\n") return i;
    i++;
  }
  return n;
}

function build(tokens) {
  const fragment = document.createDocumentFragment();
  for (const token of tokens) {
    if (!token.cls) {
      fragment.append(document.createTextNode(token.text));
      continue;
    }
    const span = el("span", { class: `tok-${token.cls}` }, token.text);
    if (token.kind) span.dataset.kind = token.kind;
    fragment.append(span);
  }
  return fragment;
}
