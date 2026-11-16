// 3D window frames rendered by tools/render3d.mjs from src/world3d/. The Skia
// compositor draws them straight over the paper background at the four scene
// windows; the painters fall back to the flat diagrams when the build is absent.
import { Image } from "skia-canvas";
import { readFileSync, existsSync } from "node:fs";

const MANIFEST = "build/3d/manifest.json";
export const HAS_3D = existsSync(MANIFEST);
const manifest = HAS_3D ? JSON.parse(readFileSync(MANIFEST, "utf8")) : null;

const frames = new Map();
const promiseCache = new Map();

export async function loadScene(scene) {
  if (frames.has(scene) || promiseCache.has(scene)) return promiseCache.get(scene);
  const meta = manifest.scenes[scene];
  const task = (async () => {
    const imgs = [];
    for (let i = 1; i <= meta.frames; i++) {
      const img = new Image();
      img.src = `build/3d/${scene}/f${String(i).padStart(6, "0")}.png`;
      await img.decode();
      imgs.push(img);
    }
    frames.set(scene, { meta, imgs, fps: manifest.fps });
    return frames.get(scene);
  })();
  promiseCache.set(scene, task);
  return task;
}

export async function preload3d() {
  if (!HAS_3D) return;
  await Promise.all(["arm", "policy", "success", "episode"].map(loadScene));
}

// Returns the RGBA Image for video time t in this scene's window, else null.
export function get3dFrame(scene, t) {
  if (!HAS_3D) return null;
  const entry = frames.get(scene);
  if (!entry) return null;
  // Pre-roll during the entrance wipe (t slightly before t0): hold the first frame
  // so the scene never falls back to the flat diagram.
  if (t < entry.meta.t0) return entry.imgs[0];
  const idx = Math.floor((t - entry.meta.t0) * entry.fps);
  if (idx >= entry.imgs.length) return entry.imgs[entry.imgs.length - 1];
  return entry.imgs[idx];
}
