// Headless capture page. Exposes window.__render3d(scene, t); the driver in
// tools/render3d.mjs calls it once per frame and screenshots with CDP
// omitBackground, so the PNGs carry alpha and the Skia layer composites them
// over the paper background.
import * as THREE from "three";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { buildSceneByName, WINDOWS } from "./scenes.js";

window.__webglOK = false;
window.__render3dError = null;

try {
  const canvas = document.getElementById("c");
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
    preserveDrawingBuffer: true, // let __render3d read the pixels back synchronously
  });
  renderer.setPixelRatio(1);
  renderer.setSize(1280, 720, false);
  renderer.setClearColor(0x000000, 0);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;

  const pmrem = new THREE.PMREMGenerator(renderer);
  const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();

  const cache = new Map();

  window.__render3d = async (sceneName, t) => {
    try {
      let entry = cache.get(sceneName);
      if (!entry) {
        if (!WINDOWS[sceneName]) throw new Error(`unknown scene ${sceneName}`);
        entry = buildSceneByName(sceneName, envTex);
        entry.update(t);
        cache.set(sceneName, entry);
      }
      entry.update(t);
      renderer.render(entry.scene, entry.camera);
      // Return the canvas pixels directly: Page.captureScreenshot stalls
      // intermittently on this GPU stack after a few hundred captures.
      return { ok: true, dataURL: canvas.toDataURL("image/png") };
    } catch (e) {
      window.__render3dError = String(e && e.stack ? e.stack : e);
      return { ok: false, error: window.__render3dError };
    }
  };

  window.__webglOK = true;
} catch (e) {
  window.__render3dError = String(e && e.stack ? e.stack : e);
}
