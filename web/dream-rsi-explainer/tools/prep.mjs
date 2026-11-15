// Build the search-tree geometry for the 3D stage from the frozen protocol grid plan.
// No run data is read: the shape comes from studies/program-v1/protocol.json (dream_rsi.grid_plan),
// so the tree is the pre-registered plan, not a record of attempts.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const root = new URL("..", import.meta.url).pathname;
const protocolPath = root + "../../studies/program-v1/protocol.json";
const plan = JSON.parse(readFileSync(protocolPath, "utf8")).dream_rsi.grid_plan;

const nodes = [{ id: "root", parent: null, status: "root", round: 0 }];
for (let r = 0; r < plan.rounds_cap; r++) {
  const rid = `r${r + 1}`;
  nodes.push({ id: rid, parent: "root", status: "planned", round: r + 1 });
  for (let b = 0; b < plan.max_branches; b++) {
    const bid = `${rid}-b${b + 1}`;
    nodes.push({ id: bid, parent: rid, status: "planned", round: r + 1 });
    for (let a = 0; a < plan.max_attempts_per_branch; a++) {
      nodes.push({ id: `${bid}-a${a + 1}`, parent: bid, status: "planned", round: r + 1 });
    }
  }
}

mkdirSync(root + "build", { recursive: true });
writeFileSync(root + "build/tree.json", JSON.stringify(nodes));
console.log(`${nodes.length} nodes from the protocol grid plan (illustrative, no run data)`);
