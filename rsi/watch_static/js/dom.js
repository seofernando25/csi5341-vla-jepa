/* DOM helpers. Every node is built with createElement/createElementNS: no markup strings. */

import { cleanText } from "./format.js";

const SVG_NS = "http://www.w3.org/2000/svg";

export function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  apply(node, props);
  append(node, children);
  return node;
}

export function svg(tag, props = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
  apply(node, props, true);
  append(node, children);
  return node;
}

function apply(node, props, isSvg = false) {
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === "class") node.setAttribute("class", value);
    else if (key === "text") node.textContent = cleanText(value);
    else if (key === "dataset") for (const [k, v] of Object.entries(value)) node.dataset[k] = String(v);
    else if (key === "style") Object.assign(node.style, value);
    else if (key === "on") for (const [name, fn] of Object.entries(value)) node.addEventListener(name, fn);
    else if (key === "attrs") for (const [k, v] of Object.entries(value)) node.setAttribute(k, String(v));
    else if (key === "value") node.value = value;
    else if (key === "checked") node.checked = Boolean(value);
    else if (key === "disabled") node.disabled = Boolean(value);
    else if (key === "tabIndex") node.tabIndex = value;
    else if (key === "href") node.setAttribute("href", value);
    else if (key === "role") node.setAttribute("role", value);
    else if (key === "title") node.setAttribute("title", value);
    else if (key === "id") node.id = value;
    else if (key === "type") node.type = value;
    else if (key === "aria") for (const [k, v] of Object.entries(value)) node.setAttribute(`aria-${k}`, String(v));
    else if (isSvg) node.setAttribute(key, String(value));
    else node.setAttribute(key, String(value));
  }
}

function append(node, children) {
  const list = Array.isArray(children) ? children : [children];
  for (const child of list) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function frag(children = []) {
  const out = document.createDocumentFragment();
  append(out, children);
  return out;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

export function setText(node, value) {
  const next = cleanText(value);
  if (node.textContent !== next) node.textContent = next;
}

export function on(node, type, fn, opts) {
  node.addEventListener(type, fn, opts);
  return () => node.removeEventListener(type, fn, opts);
}

export function icon(name, size = 14) {
  const paths = ICONS[name] || ICONS.dot;
  return svg(
    "svg",
    { class: "icon", width: size, height: size, viewBox: "0 0 16 16", "aria-hidden": "true", focusable: "false" },
    paths.map((d) =>
      svg("path", {
        d,
        fill: "none",
        stroke: "currentColor",
        "stroke-width": 1.4,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
      }),
    ),
  );
}

const ICONS = {
  dot: ["M8 8h.01"],
  chevronRight: ["M6 3.5 10.5 8 6 12.5"],
  chevronDown: ["M3.5 6 8 10.5 12.5 6"],
  close: ["M4 4l8 8", "M12 4l-8 8"],
  search: ["M7 12.5a5.5 5.5 0 1 0 0-11 5.5 5.5 0 0 0 0 11Z", "M11 11l3 3"],
  copy: ["M5.5 5.5h7v7h-7z", "M10.5 5.5v-2h-7v7h2"],
  check: ["M3.5 8.5 6.5 11.5 12.5 4.5"],
  terminal: ["M3 4l4 4-4 4", "M9 12h4"],
  file: ["M4 2.5h5l3 3v8H4z", "M9 2.5v3h3"],
  code: ["M6 5 3 8l3 3", "M10 5l3 3-3 3"],
  diff: ["M4 3v10", "M2.5 6.5h3", "M10 3v10", "M8.5 9.5h3"],
  list: ["M5 4h8", "M5 8h8", "M5 12h8", "M2.5 4h.01", "M2.5 8h.01", "M2.5 12h.01"],
  alert: ["M8 2.5 14 13H2z", "M8 6.5v3.5", "M8 11.5h.01"],
  play: ["M5 3l8 5-8 5z"],
  panelLeft: ["M2.5 3.5h11v9h-11z", "M6.5 3.5v9"],
  panelRight: ["M2.5 3.5h11v9h-11z", "M9.5 3.5v9"],
  command: ["M6 3a2 2 0 1 0-2 2h8a2 2 0 1 0-2-2v10a2 2 0 1 0 2-2H4a2 2 0 1 0 2 2z"],
  arrowUp: ["M8 13V3", "M3.5 7.5 8 3l4.5 4.5"],
  arrowDown: ["M8 3v10", "M3.5 8.5 8 13l4.5-4.5"],
  refresh: ["M13 8a5 5 0 1 1-1.6-3.7", "M13 2.5V6h-3.5"],
  external: ["M9 3h4v4", "M13 3 7 9", "M11 9.5v3h-8v-8h3"],
  sun: ["M8 10.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z", "M8 1.5v1.5", "M8 13v1.5", "M1.5 8H3", "M13 8h1.5", "M3.4 3.4l1 1", "M11.6 11.6l1 1", "M12.6 3.4l-1 1", "M4.4 11.6l-1 1"],
  moon: ["M13 9.5A5.5 5.5 0 0 1 6.5 3 5.5 5.5 0 1 0 13 9.5Z"],
  filter: ["M2.5 3.5h11l-4 5v4l-3 1.5v-5.5z"],
  jump: ["M4 8h8", "M9 5l3 3-3 3"],
};

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = el("textarea", { style: { position: "fixed", opacity: "0" } });
    area.value = text;
    document.body.append(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch {
      ok = false;
    }
    area.remove();
    return ok;
  }
}

export function copyButton(getText, label = "Copy") {
  const button = el("button", { class: "copy-btn", type: "button", title: `${label} to clipboard` }, [
    icon("copy", 12),
    label,
  ]);
  button.addEventListener("click", async () => {
    const ok = await copyText(getText());
    button.dataset.copied = ok ? "true" : "false";
    setText(button.lastChild, ok ? "Copied" : "Failed");
    setTimeout(() => {
      delete button.dataset.copied;
      setText(button.lastChild, label);
    }, 1400);
  });
  return button;
}

export function toast(message, tone = "info") {
  const root = document.getElementById("toast-root");
  if (!root) return;
  const node = el("div", { class: "toast", role: "status", dataset: { tone } }, message);
  root.append(node);
  setTimeout(() => node.remove(), 2200);
}

export function rafBatch(fn) {
  let pending = false;
  return (...args) => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      fn(...args);
    });
  };
}

export function debounce(fn, ms = 120) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

export function isMac() {
  return /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
}

export function modLabel() {
  return isMac() ? "⌘" : "Ctrl";
}

export function prefersReducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}
