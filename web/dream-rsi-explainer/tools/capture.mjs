// Frame-accurate GPU capture at 24 fps. Set FRAMES=a,b,c to spot-check specific frames.
import { mkdirSync, readFileSync } from "node:fs";
import { chromium } from "playwright-core";

const root = new URL("..", import.meta.url).pathname;
const FPS = 24;
const timeline = JSON.parse(readFileSync(root + "build/timeline.json", "utf8"));
const tree = JSON.parse(readFileSync(root + "build/tree.json", "utf8"));

const MIME = {
  html: "text/html",
  js: "text/javascript",
  json: "application/json",
  woff2: "font/woff2",
};

const server = Bun.serve({
  port: 8471,
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
  args: [
    "--no-sandbox",
    "--use-angle=vulkan",
    "--enable-features=Vulkan",
    "--ignore-gpu-blocklist",
    "--enable-gpu-rasterization",
  ],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
page.on("pageerror", (e) => console.error("PAGEERROR:", String(e).slice(0, 300)));
page.on("response", (r) => {
  if (r.status() >= 400) console.error("HTTP", r.status(), r.url());
});
await page.addInitScript(
  ([tl, tr]) => {
    window.__TIMELINE = tl;
    window.__TREE = tr;
  },
  [timeline, tree],
);
await page.goto("http://localhost:8471/src/index.html");
await page.waitForFunction(() => window.__READY === true, null, { timeout: 30000 });

const DUR = await page.evaluate(() => window.__DUR);
const total = Math.ceil(DUR * FPS);
const only = process.env.FRAMES ? process.env.FRAMES.split(",").map(Number) : null;
const idx = only || [...Array(total).keys()];
mkdirSync(root + "frames", { recursive: true });
console.log(`duration ${DUR.toFixed(2)}s -> ${idx.length}/${total} frames @${FPS}fps`);

const t0 = Date.now();
for (let k = 0; k < idx.length; k++) {
  const i = idx[k];
  await page.evaluate((t) => window.__render(t), i / FPS);
  await page.screenshot({ path: `${root}frames/f${String(i).padStart(5, "0")}.png` });
  if ((k + 1) % 200 === 0) console.log(`${k + 1}/${idx.length} (${((Date.now() - t0) / (k + 1)) | 0} ms/frame)`);
}
await browser.close();
server.stop();
console.log("capture done");
