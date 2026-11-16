/* Chapter 02 — pure helpers for the probe chapter. No DOM, no ctx. */

export function elapsedOf(node, fmt) {
  const seconds = node.elapsed_s ?? node.agent_seconds;
  return seconds == null ? "" : fmt.dur(seconds);
}

export function toneOf(status) {
  if (status === "done") return { tone: "ok" };
  if (status === "agent" || status === "evaluating") return { tone: "live" };
  if (status === "stopped") return { tone: "warn" };
  return {};
}

export function clip(text, max) {
  const value = String(text || "").replace(/\s+/g, " ").trim();
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

export function displayText(text) {
  return String(text)
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\n{2,}/g, "\u0000")
    .replace(/\n/g, " ")
    .replace(/\u0000/g, "\n\n")
    .replace(/[ \t]+/g, " ")
    .trim();
}

export function firstHeading(text) {
  const line = String(text)
    .split("\n")
    .map((item) => item.trim())
    .find((item) => /^#\s+\S/.test(item));
  return line ? line.replace(/^#+\s*/, "").trim() : "";
}

export function proposalBody(text) {
  const lines = String(text).split("\n");
  const start = lines.findIndex((line) => /^#\s+\S/.test(line.trim()));
  const body = lines
    .slice(start >= 0 ? start + 1 : 0)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"))
    .map((line) => line.replace(/^([-*]|\d+\.)\s+/, "").replace(/`/g, ""));
  const shown = body.slice(0, 2).join(" ").trim();
  if (!shown) return "";
  const cut = body.length > 2 || !/[.!?:]$/.test(shown);
  return cut ? `${shown.replace(/[,;:]$/, "")}…` : shown;
}

export function signature(probe, entry, pending, expanded) {
  if (!probe) return "none";
  const node = probe.node;
  return [
    probe.cycle.cycle_id,
    node.cell_id,
    node.status,
    node.score,
    node.successes,
    node.episodes,
    node.size_reduction,
    node.eval_seconds,
    node.agent_cost_usd,
    node.fail_class,
    node.elapsed_s,
    node.agent_seconds,
    node.headline,
    entry ? "data" : pending ? "pending" : "none",
    expanded,
  ].join("|");
}
