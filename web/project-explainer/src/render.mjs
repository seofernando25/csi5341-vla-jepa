// Direct Skia render path: no browser, no screenshots. Frames stream straight
// into ffmpeg as raw RGBA; the audio master is muxed in the same process.
import { spawn } from "node:child_process";
import { once } from "node:events";
import { readFileSync, writeFileSync } from "node:fs";
import { Canvas, FontLibrary } from "skia-canvas";

const root = new URL("..", import.meta.url).pathname;
process.chdir(root);

FontLibrary.use("Inter", [
  "node_modules/@fontsource/inter/files/inter-latin-400-normal.woff2",
  "node_modules/@fontsource/inter/files/inter-latin-500-normal.woff2",
  "node_modules/@fontsource/inter/files/inter-latin-600-normal.woff2",
  "node_modules/@fontsource/inter/files/inter-latin-800-normal.woff2",
]);
FontLibrary.use("JetBrains Mono", [
  "node_modules/@fontsource/jetbrains-mono/files/jetbrains-mono-latin-400-normal.woff2",
  "node_modules/@fontsource/jetbrains-mono/files/jetbrains-mono-latin-700-normal.woff2",
]);

import { clamp, outBack } from "./ease.js";
import { eyebrow, mono, HAIRLINE, BROWN, MUTED } from "./lib.js";
import { fillNoise, createBackground } from "./background.js";
import { buildPages, drawCaptions, drawWipe } from "./captions.js";
import { DRAW } from "./scenes.js";

const W = 1280, H = 720, FPS = 30;
const tl = JSON.parse(readFileSync("build/timeline.json", "utf8"));
const total = Math.ceil(tl.total * FPS);

const cues = {};
for (const s of tl.sentences) for (const tok of s.tokens) if (tok.cue) cues[tok.cue] = tok.start;
const wipeStarts = tl.scenes.slice(1).map((s) => s.start - 0.3);

const canvas = new Canvas(W, H);
const ctx = canvas.getContext("2d");
const noise = new Canvas(128, 128);
fillNoise(noise.getContext("2d"));
const bg = createBackground(W, H, noise);
const pages = buildPages(ctx, tl.sentences);

function sceneAt(t) {
  for (let i = tl.scenes.length - 1; i >= 0; i--) if (t >= tl.scenes[i].start - 0.3) return i;
  return 0;
}

function drawChrome(t, idx) {
  const sc = tl.scenes[idx];
  const s0 = idx === 0 ? 0 : sc.start - 0.3;
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

const ff = spawn(
  "ffmpeg",
  [
    "-y", "-v", "error",
    "-f", "rawvideo", "-pix_fmt", "rgba", "-s", `${W}x${H}`, "-r", String(FPS), "-i", "pipe:0",
    "-i", "build/master.wav",
    "-c:v", "libx264", "-profile:v", "high", "-level:v", "4.0",
    "-preset", "medium", "-crf", "21", "-maxrate", "4M", "-bufsize", "8M",
    "-pix_fmt", "yuv420p",
    "-g", "60", "-keyint_min", "30", "-sc_threshold", "0",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-c:a", "aac", "-profile:a", "aac_low", "-ar", "48000", "-ac", "2", "-b:a", "192k",
    "-movflags", "+faststart", "-shortest",
    "dist/project-explainer-720p30.mp4",
  ],
  { stdio: ["pipe", "inherit", "inherit"] },
);
ff.on("error", (e) => {
  console.error("ffmpeg spawn failed:", e);
  process.exit(1);
});
const finished = new Promise((resolve, reject) => {
  ff.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited ${code}`))));
});

const t0 = Date.now();
for (let i = 0; i < total; i++) {
  const t = i / FPS;
  const idx = sceneAt(t);
  const sc = tl.scenes[idx];
  const s0 = idx === 0 ? 0 : sc.start - 0.3;
  bg.draw(ctx, t);
  const painter = DRAW[sc.id];
  if (painter) painter(ctx, t, s0, cues, W, H);
  drawChrome(t, idx);
  drawWipe(ctx, t, wipeStarts, W, H);
  drawCaptions(ctx, pages, t, W, H);
  const buf = canvas.toBufferSync("raw");
  if (!ff.stdin.write(buf)) await once(ff.stdin, "drain");
  if ((i + 1) % 600 === 0) console.log(`${i + 1}/${total} (${((Date.now() - t0) / (i + 1)) | 0} ms/frame)`);
}
ff.stdin.end();
await finished;
console.log(`render done: dist/project-explainer-720p30.mp4 (${total} frames @ ${FPS}fps, ${((Date.now() - t0) / 1000).toFixed(1)}s)`);
