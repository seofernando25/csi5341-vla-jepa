/* Pure formatters. No DOM, no store: importable under node. */

const NB = "\u00a0";

export function pct(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}${NB}%`;
}

export function num(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

export function signed(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const n = Number(value);
  return `${n > 0 ? "+" : ""}${n.toFixed(digits)}`;
}

export function usd(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `$${Number(value).toFixed(digits)}`;
}

export function secs(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return `${Math.round(Number(value))}${NB}s`;
}

export function dur(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return "—";
  let s = Math.round(Number(seconds));
  if (s < 60) return `${s}${NB}s`;
  const m = Math.floor(s / 60);
  s %= 60;
  if (m < 60) return s ? `${m}${NB}min ${s}${NB}s` : `${m}${NB}min`;
  const h = Math.floor(m / 60);
  return `${h}${NB}h ${m % 60}${NB}min`;
}

export function elapsed(seconds) {
  if (seconds == null) return "—";
  const s = Math.max(0, Math.round(Number(seconds)));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h) return `${h}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
  return `${m}:${String(ss).padStart(2, "0")}`;
}

export function mb(value) {
  if (value == null) return "—";
  const n = Number(value);
  if (n >= 1024) return `${(n / 1024).toFixed(1)}${NB}GB`;
  return `${Math.round(n)}${NB}MB`;
}

export function chars(value) {
  if (value == null) return "—";
  const n = Number(value);
  if (n < 1000) return String(n);
  if (n < 1e6) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k`;
  return `${(n / 1e6).toFixed(1)}M`;
}

export function tokens(value) {
  if (value == null) return "—";
  return `~${chars(value)}${NB}tok`;
}

export function clock(iso) {
  if (!iso) return "—";
  const date = iso instanceof Date ? iso : new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function ago(iso, now = Date.now()) {
  if (!iso) return "—";
  const then = iso instanceof Date ? iso.getTime() : new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const s = Math.max(0, Math.round((now - then) / 1000));
  if (s < 5) return "just now";
  if (s < 60) return `${s}${NB}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}${NB}min ago`;
  if (s < 86_400) return `${Math.floor(s / 3600)}${NB}h ago`;
  return `${Math.floor(s / 86_400)}${NB}d ago`;
}

export function shortId(value, keep = 8) {
  const text = String(value || "");
  return text.length <= keep ? text : text.slice(0, keep);
}

export function plural(count, one, many = `${one}s`) {
  return `${count}${NB}${count === 1 ? one : many}`;
}

export function cleanText(value) {
  if (value == null) return "";
  return String(value).replace(/\[object Object\]/g, "");
}

export function firstSentence(text, max = 160) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) return "";
  const match = clean.match(/^(.{0,240}?[.!?])(?:\s|$)/);
  const out = match ? match[1] : clean;
  return out.length > max ? `${out.slice(0, max - 1)}…` : out;
}

export function firstLine(text, max = 120) {
  const line = String(text || "")
    .split("\n")
    .map((l) => l.trim())
    .find(Boolean);
  if (!line) return "";
  return line.length > max ? `${line.slice(0, max - 1)}…` : line;
}

export const FAIL_LABEL = {
  ok: "passed",
  gate_near_miss: "near miss",
  gate_failed: "gate failed",
  implementation_failure: "implementation failed",
  runtime_failure: "runtime failure",
  agent_failure: "agent failed",
  agent_timeout: "agent timed out",
};

export const FAIL_SHORT = {
  ok: "passed",
  gate_near_miss: "near miss",
  gate_failed: "gate failed",
  implementation_failure: "failed",
  runtime_failure: "runtime",
  agent_failure: "agent failed",
  agent_timeout: "timeout",
};

export function failLabel(value, short = false) {
  if (!value) return "—";
  return (short ? FAIL_SHORT : FAIL_LABEL)[value] || value;
}

export const NODE_STATUS = {
  pending: "queued",
  agent: "agent",
  evaluating: "evaluating",
  stopped: "stopped",
  done: "done",
};

export function nodeStatusLabel(value) {
  return NODE_STATUS[value] || value || "—";
}

export const VERDICT = {
  accepted: "accepted",
  not_better: "not better",
  timeout: "timeout",
  rejected: "rejected",
  running: "running",
  pending: "pending",
};

export function verdictLabel(value) {
  return VERDICT[value] || value || "—";
}

export function titleCase(value) {
  const text = String(value || "");
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}
