// Compact the real Dream-RSI search tree (studies/dream-rsi/attempts.jsonl) for the 3D scene.
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
const root = new URL("..", import.meta.url).pathname;
mkdirSync(root + "build", { recursive: true });
const rows = readFileSync(root + "../../studies/dream-rsi/attempts.jsonl", "utf8").trim().split("\n").map((l) => JSON.parse(l));
const nodes = [{ id: "root", parent: null, status: "root", cycle: -1 }, ...rows.map((r) => ({ id: r.attempt_id, parent: r.parent, status: r.status, cycle: r.cycle }))];
writeFileSync(root + "build/tree.json", JSON.stringify(nodes));
console.log(`${nodes.length} nodes`);
