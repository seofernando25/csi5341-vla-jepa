// Frame-accurate capture at 30 fps. Set FRAMES=a,b,c to spot-check only those frames.
import { mkdirSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { chromium } from "playwright-core";

const root = new URL("..", import.meta.url).pathname;
const FPS = 30;
const framesDir = root + "frames";
const frameName = (i) => `f${String(i).padStart(5, "0")}.png`;
const timeline = JSON.parse(readFileSync(root + "build/timeline.json", "utf8"));

const MIME = {
  html: "text/html",
  js: "text/javascript",
  json: "application/json",
  woff2: "font/woff2",
};

const server = Bun.serve({
  port: 8472,
  async fetch(req) {
    const url = new URL(req.url);
    const p = root + (url.pathname === "/" ? "src/index.html" : decodeURIComponent(url.pathname.slice(1)));
    try {
      const ext = p.split(".").pop();
      return new Response(await Bun.file(p).arrayBuffer(), {
        headers: { "Content-Type": MIME[ext] || "application/octet-stream" },
      });
    } catch {
      return new Response("nf", { status: 404 });
    }
  },
});

const browser = await chromium.launch({
  executablePath: "/usr/bin/chromium",
  args: ["--no-sandbox", "--disable-gpu", "--force-color-profile=srgb"],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
page.on("pageerror", (e) => console.error("PAGEERROR:", String(e).slice(0, 300)));
page.on("response", (r) => {
  if (r.status() >= 400) console.error("HTTP", r.status(), r.url());
});
await page.addInitScript((tl) => {
  window.__TIMELINE = tl;
}, timeline);
await page.goto("http://localhost:8472/src/index.html");
await page.waitForFunction(() => window.__READY === true, null, { timeout: 30000 });

const DUR = await page.evaluate(() => window.__DUR);
const total = Math.ceil(DUR * FPS);
const only = process.env.FRAMES ? process.env.FRAMES.split(",").map(Number) : null;
const idx = only || [...Array(total).keys()];
if (only) {
  mkdirSync(framesDir, { recursive: true });
} else {
  // Frames left by an earlier cut would be muxed as if they belonged to this one, so start empty.
  rmSync(framesDir, { recursive: true, force: true });
  mkdirSync(framesDir, { recursive: true });
}
console.log(`duration ${DUR.toFixed(2)}s -> ${idx.length}/${total} frames @${FPS}fps`);

const t0 = Date.now();
for (let k = 0; k < idx.length; k++) {
  const i = idx[k];
  await page.evaluate((t) => window.__render(t), i / FPS);
  await page.screenshot({ path: framesDir + "/" + frameName(i), type: "png" });
  if ((k + 1) % 300 === 0) console.log(`${k + 1}/${idx.length} (${((Date.now() - t0) / (k + 1)) | 0} ms/frame)`);
}
await browser.close();
server.stop();

const want = new Set(idx.map(frameName));
const written = new Set(readdirSync(framesDir).filter((f) => f.endsWith(".png")));
const missing = [...want].filter((f) => !written.has(f));
const extra = only ? [] : [...written].filter((f) => !want.has(f));
if (missing.length || extra.length) {
  throw new Error(
    `frames/ does not match the capture: ${written.size} png, expected ${want.size}` +
      (extra.length ? `; unexpected ${extra.slice(0, 3).join(", ")}` : "") +
      (missing.length ? `; missing ${missing.slice(0, 3).join(", ")}` : ""),
  );
}
console.log("capture done");
