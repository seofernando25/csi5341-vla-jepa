// Headless capture for the world3d scenes: serves the project over HTTP,
// launches /usr/bin/chromium, calls window.__render3d(scene, t) once per frame
// and writes RGBA PNGs read back straight from the WebGL canvas to
// build/3d/<scene>/. Every frame is decoded and sanity-checked
// (1280x720 RGBA, > 1500 non-transparent pixels); any failure aborts loudly.
//   bun tools/render3d.mjs            full render of the four windows
//   bun tools/render3d.mjs --probe    3 probe frames per scene (build/3d/<s>/probe-fNNNNNN.png)
//   bun tools/render3d.mjs --scene=arm --frames=10,20,30   explicit frame indices
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { inflateSync } from "node:zlib";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
process.chdir(root);

const FPS = 30, W = 1280, H = 720;
const SCENES = [
  { name: "arm", t0: 10.634, t1: 21.727 },
  { name: "policy", t0: 22.277, t1: 33.715 },
  { name: "success", t0: 34.265, t1: 43.404 },
  { name: "episode", t0: 43.954, t1: 61.394 },
];
const frameCount = (s) => Math.ceil((s.t1 - s.t0) * FPS);

const args = new Map(process.argv.slice(2).map((a) => {
  const m = a.match(/^--([\w-]+)(?:=(.*))?$/);
  return m ? [m[1], m[2] ?? true] : [a, true];
}));
const PROBE = args.has("probe");
const SCENE_FILTER = args.get("scene") ? String(args.get("scene")).split(",") : null;
// FRAMES=a,b,c (env) or --frames=a,b,c spot-renders specific frames, no manifest.
const ENV_FRAMES = process.env.FRAMES ? process.env.FRAMES.split(",").map(Number) : null;
const FRAME_FILTER = ENV_FRAMES ?? (args.get("frames") ? String(args.get("frames")).split(",").map(Number) : null);
const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".css": "text/css", ".png": "image/png" };

// ------------------------------------------------------------ tiny PNG decoder (RGBA8, non-interlaced)
function decodePng(buf) {
  const sig = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  if (buf.length < 8 || !buf.subarray(0, 8).equals(sig)) throw new Error("not a PNG");
  let pos = 8, width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
  const idat = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos); const type = buf.toString("ascii", pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len); pos += 12 + len;
    if (type === "IHDR") { width = data.readUInt32BE(0); height = data.readUInt32BE(4); bitDepth = data[8]; colorType = data[9]; interlace = data[12]; }
    else if (type === "IDAT") idat.push(data);
    else if (type === "IEND") break;
  }
  if (bitDepth !== 8 || colorType !== 6 || interlace !== 0) throw new Error(`unexpected PNG format depth=${bitDepth} type=${colorType} interlace=${interlace} (need RGBA8 non-interlaced)`);
  const raw = inflateSync(Buffer.concat(idat));
  const bpp = 4, stride = width * bpp;
  const out = Buffer.alloc(height * stride);
  let prev = null, rp = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[rp++];
    const line = raw.subarray(rp, rp + stride); rp += stride;
    const cur = out.subarray(y * stride, (y + 1) * stride);
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0;
      const b = prev ? prev[i] : 0;
      const c = prev && i >= bpp ? prev[i - bpp] : 0;
      let v = line[i];
      if (filter === 1) v = (v + a) & 0xff;
      else if (filter === 2) v = (v + b) & 0xff;
      else if (filter === 3) v = (v + ((a + b) >> 1)) & 0xff;
      else if (filter === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v = (v + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c)) & 0xff;
      }
      cur[i] = v;
    }
    prev = cur;
  }
  return { width, height, data: out };
}
function countNonTransparent(png) {
  let n = 0;
  const d = png.data;
  for (let i = 3; i < d.length; i += 4) if (d[i] > 16) n++;
  return n;
}

// ------------------------------------------------------------ http static server
function serve() {
  const server = createServer((req, res) => {
    const urlPath = decodeURIComponent(new URL(req.url, "http://x").pathname);
    const file = path.join(root, urlPath === "/" ? "/src/world3d/page.html" : urlPath);
    if (!file.startsWith(root) || !existsSync(file)) { res.writeHead(404); res.end("nope"); return; }
    try {
      const body = readFileSync(file);
      res.writeHead(200, { "content-type": MIME[path.extname(file)] ?? "application/octet-stream", "cache-control": "no-store" });
      res.end(body);
    } catch { res.writeHead(500); res.end(); }
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port })));
}

// ------------------------------------------------------------ chromium / CDP
function launchChromium(url, extraArgs) {
  const flags = ["--headless=new", "--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=1", `--window-size=${W},${H}`, "--remote-debugging-port=0", ...extraArgs, url];
  const proc = spawn("/usr/bin/chromium", flags, { stdio: ["ignore", "ignore", "pipe"] });
  return new Promise((resolve, reject) => {
    let stderr = "";
    const onData = (chunk) => {
      stderr += chunk;
      const m = stderr.match(/DevTools listening on (ws:\/\/\S+)/);
      if (m) { cleanup(); resolve({ proc, wsUrl: m[1] }); }
    };
    const onExit = (code) => { cleanup(); reject(new Error(`chromium exited early (code ${code}): ${stderr.slice(-2000)}`)); };
    const cleanup = () => { proc.stderr.off("data", onData); proc.off("exit", onExit); };
    proc.stderr.on("data", onData);
    proc.on("exit", onExit);
    setTimeout(() => { cleanup(); proc.kill("SIGKILL"); reject(new Error("timed out waiting for DevTools ws url")); }, 30000);
  });
}

class CDP {
  static async connect(wsUrl) {
    const ws = new WebSocket(wsUrl);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = () => rej(new Error("ws open failed")); });
    const cdp = new CDP(ws);
    return cdp;
  }
  constructor(ws) {
    this.ws = ws; this.id = 0; this.pending = new Map();
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { res, rej } = this.pending.get(msg.id); this.pending.delete(msg.id);
        if (msg.error) rej(new Error(`${msg.error.message} (code ${msg.error.code})`)); else res(msg.result ?? {});
      }
    };
  }
  send(method, params = {}, timeoutMs = 35000) {
    const id = ++this.id;
    return new Promise((res, rej) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        rej(new Error(`CDP ${method} timed out after ${timeoutMs} ms (page hung or crashed?)`));
      }, timeoutMs);
      this.pending.set(id, {
        res: (v) => { clearTimeout(timer); res(v); },
        rej: (e) => { clearTimeout(timer); rej(e); },
      });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  close() { try { this.ws.close(); } catch { /* already closed */ } }
}

async function evaluate(cdp, expression, awaitPromise = true) {
  const r = await cdp.send("Runtime.evaluate", { expression, awaitPromise, returnByValue: true });
  if (r.exceptionDetails) throw new Error(`page exception: ${JSON.stringify(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text)}`);
  return r.result?.value;
}

async function session(url, extraArgs) {
  const { proc, wsUrl } = await launchChromium(url, extraArgs);
  const browserPort = new URL(wsUrl).port;
  // find the page target
  let pageTarget = null;
  for (let i = 0; i < 50 && !pageTarget; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${browserPort}/json/list`)).json();
      pageTarget = list.find((t) => t.type === "page");
    } catch { /* not up yet */ }
    if (!pageTarget) await new Promise((r) => setImmediate(r));
  }
  if (!pageTarget) { proc.kill("SIGKILL"); throw new Error("no page target found"); }
  const cdp = await CDP.connect(pageTarget.webSocketDebuggerUrl);
  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");
  // --window-size is the outer window; pin the layout viewport exactly.
  await cdp.send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
  return { proc, cdp };
}

async function waitForPage(cdp, url) {
  await cdp.send("Page.navigate", { url });
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    const ok = await evaluate(cdp, "typeof window.__webglOK", false).catch(() => undefined);
    if (ok === "boolean") {
      const webglOK = await evaluate(cdp, "window.__webglOK", false);
      const err = await evaluate(cdp, "window.__render3dError", false);
      return { webglOK, err };
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  const err = await evaluate(cdp, "window.__render3dError", false).catch(() => "unknown");
  return { webglOK: false, err: `timed out (30 s) waiting for page init; __render3dError: ${err}` };
}

async function openSession(pageUrl) {
  let attempt = await session(pageUrl, ["--use-angle=vulkan"]);
  let state = await waitForPage(attempt.cdp, pageUrl);
  if (!state.webglOK) {
    console.error(`[render3d] WebGL with --use-angle=vulkan failed: ${state.err}\n[render3d] retrying with default SwiftShader launch`);
    attempt.cdp.close(); attempt.proc.kill("SIGKILL");
    attempt = await session(pageUrl, []);
    state = await waitForPage(attempt.cdp, pageUrl);
  }
  if (!state.webglOK) {
    console.error(`[render3d] FAIL: WebGL unavailable even with SwiftShader: ${state.err}`);
    attempt.cdp.close(); attempt.proc.kill("SIGKILL");
    process.exit(1);
  }
  return attempt;
}

// ------------------------------------------------------------ main
const { server, port } = await serve();
const pageUrl = `http://127.0.0.1:${port}/src/world3d/page.html`;

let { cdp, proc } = await openSession(pageUrl);
console.error(`[render3d] WebGL ready (pid ${proc.pid})`);

async function restartSession() {
  cdp.close(); proc.kill("SIGKILL");
  const fresh = await openSession(pageUrl);
  cdp = fresh.cdp; proc = fresh.proc;
}

// Long capture runs the GPU/compositor into a hang after a few hundred
// frames; frames are pure functions of t, so periodic reloads are safe.
const CHUNK = 128;
let framesSinceRestart = 0;

const summary = [];
const selected = SCENES.filter((s) => !SCENE_FILTER || SCENE_FILTER.includes(s.name));
for (const scene of selected) {
  const total = frameCount(scene);
  const indices = PROBE
    ? [0, Math.floor(total / 2), total - 1]
    : FRAME_FILTER ?? Array.from({ length: total }, (_, i) => i);
  const dir = `build/3d/${scene.name}`;
  mkdirSync(dir, { recursive: true });
  let written = 0, bytes = 0, minNonTransp = Infinity;
  const t0 = Date.now();
  for (const i of indices) {
    if (framesSinceRestart >= CHUNK) {
      await restartSession();
      framesSinceRestart = 0;
      console.error(`[render3d] session restart before frame ${i + 1} (chunk boundary)`);
    }
    if (!Number.isInteger(i) || i < 0 || i >= total) throw new Error(`scene ${scene.name}: frame index ${i} out of range [0, ${total})`);
    const t = scene.t0 + i / FPS;
    const r = await evaluate(cdp, `window.__render3d(${JSON.stringify(scene.name)}, ${t})`);
    if (!r || !r.ok || !r.dataURL) throw new Error(`scene ${scene.name} frame ${i + 1}: __render3d failed: ${r?.error ?? "no dataURL"}`);
    const png = Buffer.from(r.dataURL.slice(r.dataURL.indexOf(",") + 1), "base64");
    const decoded = decodePng(png);
    if (decoded.width !== W || decoded.height !== H) throw new Error(`scene ${scene.name} frame ${i + 1}: ${decoded.width}x${decoded.height}, expected ${W}x${H}`);
    const nonTransparent = countNonTransparent(decoded);
    if (nonTransparent <= 1500) throw new Error(`scene ${scene.name} frame ${i + 1}: only ${nonTransparent} non-transparent pixels (need > 1500)`);
    const err = await evaluate(cdp, "window.__render3dError", false).catch(() => null);
    if (err) throw new Error(`scene ${scene.name} frame ${i + 1}: __render3dError set: ${err}`);
    const file = (PROBE || FRAME_FILTER) ? `${dir}/probe-f${String(i + 1).padStart(6, "0")}.png` : `${dir}/f${String(i + 1).padStart(6, "0")}.png`;
    writeFileSync(file, png);
    written++; bytes += png.length; minNonTransp = Math.min(minNonTransp, nonTransparent);
    framesSinceRestart++;
    if (!PROBE && !FRAME_FILTER && (i % 100 === 99 || i === total - 1)) console.error(`[render3d] ${scene.name}: ${i + 1}/${total} frames`);
  }
  if (!PROBE && !FRAME_FILTER && written !== total) {
    throw new Error(`scene ${scene.name}: wrote ${written} frames, expected ${total}`);
  }
  console.error(`[render3d] ${scene.name}: ${written} frames (min non-transparent px ${minNonTransp}) in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  summary.push({ name: scene.name, t0: scene.t0, t1: scene.t1, fps: FPS, frames: written, expected: total, dir, bytes });
}

if (!PROBE && !FRAME_FILTER) {
  const scenesOut = {};
  for (const s of summary) scenesOut[s.name] = { frames: s.frames, t0: s.t0, t1: s.t1 };
  writeFileSync("build/3d/manifest.json", JSON.stringify({
    fps: FPS,
    width: W,
    height: H,
    scenes: scenesOut,
  }, null, 2));
  console.error("[render3d] wrote build/3d/manifest.json");
}
cdp.close(); proc.kill("SIGKILL"); server.close();
console.error("[render3d] done");
