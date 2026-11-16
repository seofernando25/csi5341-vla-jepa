/* View metadata shared by the reducer, the tab bar and the registry. No imports, no DOM. */

export const NODE_VIEWS = [
  { id: "transcript", label: "Transcript", key: "1" },
  { id: "context", label: "Context", key: "2" },
  { id: "proposal", label: "Proposal", key: "3" },
  { id: "diff", label: "Diff", key: "4" },
  { id: "solution", label: "Solution", key: "5" },
  { id: "evaluation", label: "Evaluation", key: "6" },
];

export const POLICY_VIEWS = [
  { id: "prompt", label: "Prompt", key: "1" },
  { id: "diff", label: "Diff", key: "2" },
  { id: "replay", label: "Replay", key: "3" },
  { id: "verdict", label: "Verdict", key: "4" },
  { id: "transcript", label: "Transcript", key: "5" },
];

export function viewsFor(kind) {
  return kind === "policy" ? POLICY_VIEWS : NODE_VIEWS;
}

export function defaultView(kind) {
  return kind === "policy" ? "prompt" : "transcript";
}

export function isView(kind, view) {
  return viewsFor(kind).some((entry) => entry.id === view);
}

export function viewLabel(kind, view) {
  return viewsFor(kind).find((entry) => entry.id === view)?.label || view;
}
