// Main entry: paper background + scene painters + chrome + wipes + captions.
import { PAPER, MUTED, HAIRLINE, BROWN, INK, mono, eyebrow } from "./lib.js";
import { createBackground } from "./background.js";
import { buildPages, drawCaptions, drawWipe } from "./captions.js";
import { DRAW } from "./scenes.js";
import { clamp, outBack } from "./ease.js";

const W = 1280, H = 720;
const tl = window.__TIMELINE;
const cues = {};
for (const s of tl.sentences) for (const tok of s.tokens) if (tok.cue) cues[tok.cue] = tok.start;
const wipeStarts = tl.scenes.slice(1).map((s) => s.start - 0.3);
window.__DUR = tl.total;

const bg = createBackground(W, H);
const hud = document.getElementById("hud");
hud.width = W;
hud.height = H;
const ctx = hud.getContext("2d");

function sceneAt(t) {
  for (let i = tl.scenes.length - 1; i >= 0; i--) if (t >= tl.scenes[i].start - 0.3) return i;
  return 0;
}

await document.fonts.ready;
const pages = buildPages(ctx, tl.sentences);

function drawChrome(t, idx) {
  const sc = tl.scenes[idx];
  const s0 = idx === 0 ? 0 : sc.start - 0.3;
  // top hairline + progress
  ctx.strokeStyle = HAIRLINE;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(48, 46);
  ctx.lineTo(W - 48, 46);
  ctx.stroke();
  ctx.strokeStyle = BROWN;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(48, 46);
  ctx.lineTo(48 + (W - 96) * clamp(t / tl.total), 46);
  ctx.stroke();
  // eyebrow: scene numeral + label, small pop on scene change
  const pop = outBack(clamp((t - s0) / 0.4), 2.0);
  ctx.save();
  ctx.translate(48, 22);
  ctx.scale(0.7 + 0.3 * pop, 0.7 + 0.3 * pop);
  ctx.translate(-48, -22);
  ctx.globalAlpha = clamp(pop);
  eyebrow(ctx, String(idx + 1).padStart(2, "0"), sc.chip.replace(/\s+/g, " "), 48, 22);
  ctx.restore();
  ctx.globalAlpha = 1;
  mono(ctx, "C S I   5 3 4 1   ·   V L A - J E P A", W - 48, 22, { size: 11, color: MUTED, align: "right", spacing: 1.6 });
}

window.__render = (t) => {
  bg.draw(ctx, t);
  const idx = sceneAt(t);
  const sc = tl.scenes[idx];
  const s0 = idx === 0 ? 0 : sc.start - 0.3;
  const painter = DRAW[sc.id];
  if (painter) painter(ctx, t, s0, cues, W, H);
  drawChrome(t, idx);
  drawWipe(ctx, t, wipeStarts, W, H);
  drawCaptions(ctx, pages, t, W, H);
};

window.__READY = true;
