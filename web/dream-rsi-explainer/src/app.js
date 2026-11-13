// Main entry: ties timeline + 3D stage + 2D infographics + word-highlighted captions into window.__render(t).
import { clamp, COL, lerp, mixHex, outBack, smooth } from "./ease.js";
import { createStage } from "./stage3d.js";
import { buildPages, drawCaptions, drawWipe } from "./captions.js";
import { drawHUD, sceneIndexAt } from "./scenes.js";

const W = 1280, H = 720;
const tl = window.__TIMELINE;
const tree = window.__TREE;
window.__DUR = tl.total;

const cues = {};
for (const s of tl.sentences) for (const tok of s.tokens) if (tok.cue) cues[tok.cue] = tok.start;

await document.fonts.ready;

const stage = createStage(document.getElementById("bg"), W, H, tree);
const hud = document.getElementById("hud");
hud.width = W;
hud.height = H;
const ctx = hud.getContext("2d");
const pages = buildPages(ctx, tl.sentences);
const wipeStarts = tl.scenes.slice(1).map((s) => s.start - 0.3);

// Per-scene 3D pose: [brainVis, brainX, brainScale, treeVis, treeX, treeScale, camZ, camY]
const POSE = {
  hook: [1, 2.3, 1.05, 0, 0, 0.7, 7.6, 0.3],
  idea: [0, 0, 0.8, 1, 2.2, 0.78, 8.2, 1.8],
  dream: [0, 0, 0.8, 1, -2.5, 0.78, 8.0, 2.0],
  loop: [0, 0, 0.8, 0.28, 0, 0.95, 9.2, 2.4],
  project: [1, 3.0, 0.95, 0, 0, 0.75, 7.8, 0.4],
  fail: [0.9, -2.7, 0.85, 0, 0, 0.75, 8.0, 0.3],
  code: [0.9, -2.8, 0.85, 0, 0, 0.75, 8.0, 0.5],
  arms: [0.4, 3.0, 0.65, 0, 0, 0.8, 8.4, 0.6],
  audit: [0, 0, 0.8, 0.3, 0, 0.95, 9.0, 2.2],
  fix: [0.6, 2.6, 0.8, 0.25, -1.5, 0.85, 8.2, 1.0],
  live: [0.35, 3.1, 0.62, 0, 0, 0.75, 8.2, 0.3],
  jump: [0.5, -2.8, 0.7, 0, 0, 0.8, 8.2, 0.5],
  tasks: [0, 0, 0.8, 0.3, 0, 0.95, 9.0, 2.2],
  close: [0.95, 0, 0.95, 0.22, 0, 1.05, 7.8, 0.6],
};

const SLAM_TIMES = [cues.zero, cues.who, cues.never, cues.zero2, cues.lossnot, cues.r3, cues.fail6, cues.robot];

function state3D(t) {
  const idx = sceneIndexAt(t, tl.scenes);
  const sc = tl.scenes[idx];
  const prev = tl.scenes[Math.max(0, idx - 1)];
  const s0 = idx === 0 ? 0 : sc.start - 0.3;
  const k = idx === 0 ? 1 : smooth((t - s0) / 0.65);
  const A = POSE[prev.id], B = POSE[sc.id];
  const P = A.map((a, i) => lerp(a, B[i], k));

  let [bv, bx, bs, tv, tx, ts, cz, cy] = P;
  let brainColor = COL.cyan;
  let shatter = 0;
  if (sc.id === "hook") {
    const kb = smooth((t - (cues.better - 0.1)) / 0.3);
    const kz = smooth((t - (cues.zero - 0.05)) / 0.2);
    brainColor = mixHex(mixHex(COL.cyan, COL.green, kb), COL.red, kz);
    shatter = clamp((t - cues.zero) / 0.9) * 0.85;
  } else if (sc.id === "project") {
    const ks = smooth((t - cues.smol) / 0.45);
    brainColor = mixHex(COL.cyan, COL.magenta, ks);
    bs = lerp(1.05, 0.58, ks);
  } else if (sc.id === "fail") {
    brainColor = COL.red;
    shatter = clamp((t - cues.zero2) / 1.4) * 0.95;
  } else if (sc.id === "live" || sc.id === "close" || sc.id === "jump" || sc.id === "tasks" || sc.id === "fix") {
    brainColor = COL.green;
  }

  let pulse = 0;
  let sx = 0, sy = 0;
  for (const st of SLAM_TIMES) {
    const dt = t - st;
    if (dt >= 0 && dt < 0.55) {
      const e = Math.exp(-dt * 7);
      pulse = Math.max(pulse, e);
      sx += Math.sin(dt * 85) * e * 14;
      sy += Math.cos(dt * 73) * e * 10;
    }
  }

  const grow = (i) => {
    if (sc.id === "idea") return outBack(clamp((t - (sc.start + 0.6 + i * 0.15)) / 0.32));
    return 1;
  };
  const ghost = sc.id === "dream" ? 1 : 0;
  const runK = sc.id === "dream" && t >= cues.walk - 0.1 ? 4 : sc.id === "idea" && t >= cues.gpu - 0.3 ? 1 : 0;
  return {
    bv, bx, bs, tv, tx, ts, cz, cy,
    brainColor, shatter, pulse, grow, ghost,
    runT: Math.max(0, t - (sc.id === "dream" ? cues.walk : cues.gpu)),
    runK,
    nodePulse: (i) => (sc.id === "idea" && t >= cues.who && i < 6 ? 0.35 * Math.sin(t * 10 + i) : 0),
    edgeColor: ghost ? 0x6a7cff : 0x3df0ff,
    shake: [sx / 60, sy / 60],
    pxShake: [sx, sy],
  };
}

window.__render = (t) => {
  const s = state3D(t);
  stage.render(t, s);
  ctx.clearRect(0, 0, W, H);
  ctx.save();
  ctx.translate(s.pxShake[0], s.pxShake[1]);
  drawHUD(ctx, t, tl, cues, tree, W, H);
  ctx.restore();
  drawWipe(ctx, t, wipeStarts, W, H);
  drawCaptions(ctx, pages, t, W, H);
};

window.__READY = true;
