/* View registry: the only place that knows every view module. */

import * as transcript from "./transcript.js";
import * as context from "./context.js";
import * as proposal from "./proposal.js";
import * as diff from "./diff.js";
import * as solution from "./solution.js";
import * as evaluation from "./evaluation.js";
import * as policyStep from "./policy_step.js";

const NODE = { transcript, context, proposal, diff, solution, evaluation };
const POLICY = {
  prompt: { mount: policyStep.mountPrompt },
  diff: { mount: policyStep.mountDiff },
  replay: { mount: policyStep.mountReplay },
  verdict: { mount: policyStep.mountVerdict },
};

export function mountView(kind, view, root, ctx) {
  const table = kind === "policy" ? POLICY : NODE;
  const entry = table[view] || (kind === "policy" ? POLICY.prompt : NODE.transcript);
  if (!entry || typeof entry.mount !== "function") return null;
  return entry.mount(root, ctx);
}
