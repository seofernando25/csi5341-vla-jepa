// Scene definitions for the world3d pages. Each scene is built once and then
// driven purely by t (seconds, video timeline). No text is drawn here; the
// Skia 2D layer composites labels on top.
import * as THREE from "three";
import { createBench, createBowl, createPlate, createWorkshopClutter, createShadowCatcher, PLATE_DISH_Y } from "./props.js";
import { createArm, BOWL_GRIP_DROP } from "./arm.js";
import { sampleKeys, smootherstep, clamp01 } from "./motion.js";

export const WINDOWS = {
  arm: { t0: 10.634, t1: 21.727 },
  policy: { t0: 22.277, t1: 33.715 },
  success: { t0: 34.265, t1: 43.404 },
  episode: { t0: 43.954, t1: 61.394 },
};

const BOWL_START = new THREE.Vector3(0.02, 0, 0.55);
const PLATE_POS = new THREE.Vector3(1.28, 0, 0.62);
const PI = Math.PI;
const HOME = { x: 1.55, y: 2.15, z: 0.3, lean: 2.98, grip: 0.1 };

function addLights(scene) {
  const key = new THREE.DirectionalLight(0xfff0dd, 2.1);
  key.position.set(4.6, 6.4, 4.2);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -7;
  key.shadow.camera.right = 7;
  key.shadow.camera.top = 7;
  key.shadow.camera.bottom = -7;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 22;
  key.shadow.bias = -0.0003;
  key.shadow.normalBias = 0.02;
  key.shadow.radius = 5;
  const target = new THREE.Object3D();
  target.position.set(0.9, 0, 0.3);
  scene.add(target);
  key.target = target;
  scene.add(key);
  scene.add(new THREE.HemisphereLight(0xfff6ea, 0x6b5b4e, 0.5));
}

function makeCamera(fov = 36) {
  return new THREE.PerspectiveCamera(fov, 1280 / 720, 0.1, 60);
}

// Wrist-target keyframes for one pick-and-place pass, relative to window start.
function pickKeys(w0, slow = 1) {
  const k = (tau, v) => ({ t: w0 + tau * slow, v });
  return [
    k(0.0, HOME),
    k(0.55, HOME),
    k(1.35, { x: 1.35, y: 1.35, z: 0.5, lean: 2.75, grip: 0.0 }),
    k(2.15, { x: BOWL_START.x, y: 1.5, z: BOWL_START.z, lean: PI, grip: 0.0 }),
    k(2.95, { x: BOWL_START.x, y: 0.85, z: BOWL_START.z, lean: PI, grip: 0.0 }),
    k(3.35, { x: BOWL_START.x, y: 0.85, z: BOWL_START.z, lean: PI, grip: 1.0 }),
    k(3.95, { x: BOWL_START.x, y: 1.7, z: BOWL_START.z, lean: 2.95, grip: 1.0 }),
    k(5.45, { x: PLATE_POS.x, y: 1.72, z: PLATE_POS.z, lean: 2.9, grip: 1.0 }),
    k(6.35, { x: PLATE_POS.x, y: 0.886, z: PLATE_POS.z, lean: PI, grip: 1.0 }),
    k(6.9, { x: PLATE_POS.x, y: 0.886, z: PLATE_POS.z, lean: PI, grip: 0.0 }),
    k(7.35, { x: PLATE_POS.x, y: 0.95, z: PLATE_POS.z, lean: 2.8, grip: 0.0 }),
    k(8.25, { x: 1.7, y: 1.25, z: 0.5, lean: 2.7, grip: 0.0 }),
    k(9.3, { ...HOME, grip: 0.05 }),
    k(11.2, { ...HOME, grip: 0.05 }),
  ];
}

// Episode scene: the same pick-and-place spread over the longer window with
// four readable beats — settle (RESET), a slow lean toward the bowl
// (OBSERVE), the pick itself (ACT), hold with the bowl on the plate (DONE).
function episodeKeys(w0) {
  const k = (tau, v) => ({ t: w0 + tau, v });
  return [
    k(0.0, HOME),
    k(2.4, { ...HOME, grip: 0.02 }),
    k(3.5, { x: 1.55, y: 1.47, z: 0.45, lean: 2.75, grip: 0.02 }), // OBSERVE lean
    k(4.4, { x: 1.5, y: 1.38, z: 0.48, lean: 2.75, grip: 0.0 }),
    k(5.6, { x: 1.3, y: 1.32, z: 0.5, lean: 2.75, grip: 0.0 }), // ACT: reach
    k(6.9, { x: BOWL_START.x, y: 1.45, z: BOWL_START.z, lean: PI, grip: 0.0 }),
    k(8.1, { x: BOWL_START.x, y: 0.85, z: BOWL_START.z, lean: PI, grip: 0.0 }),
    k(8.6, { x: BOWL_START.x, y: 0.85, z: BOWL_START.z, lean: PI, grip: 1.0 }), // grasp
    k(9.5, { x: BOWL_START.x, y: 1.68, z: BOWL_START.z, lean: 2.95, grip: 1.0 }), // lift
    k(11.2, { x: PLATE_POS.x, y: 1.7, z: PLATE_POS.z, lean: 2.9, grip: 1.0 }), // swing
    k(12.0, { x: PLATE_POS.x, y: 0.886, z: PLATE_POS.z, lean: PI, grip: 1.0 }), // lower
    k(12.6, { x: PLATE_POS.x, y: 0.886, z: PLATE_POS.z, lean: PI, grip: 0.0 }), // release
    k(13.2, { x: PLATE_POS.x, y: 0.95, z: PLATE_POS.z, lean: 2.75, grip: 0.0 }),
    k(14.2, { x: 1.72, y: 1.3, z: 0.5, lean: 2.7, grip: 0.0 }), // retract
    k(15.3, { ...HOME, grip: 0.05 }), // DONE: back to idle, bowl on plate
    k(17.44, { ...HOME, grip: 0.05 }),
  ];
}

function settleBounce(dt) {
  if (dt <= 0) return 0;
  return 0.028 * Math.exp(-6.5 * dt) * Math.abs(Math.sin(11 * dt));
}

// Shared task world: bench slab, ShadowMaterial catcher, plate, bowl, arm.
function buildTaskWorld(scene, { withArm = true } = {}) {
  const bench = createBench();
  scene.add(bench);
  scene.add(createShadowCatcher({ width: 12.6, depth: 6.9, y: 0.006 }));

  const plate = createPlate();
  plate.position.copy(PLATE_POS);
  scene.add(plate);

  const bowl = createBowl();
  bowl.position.copy(BOWL_START);
  scene.add(bowl);

  let arm = null;
  if (withArm) {
    arm = createArm();
    scene.add(arm.root);
  }
  return { arm, bowl, plate };
}

function updateArmBowl(arm, bowl, t, keys, graspT, releaseT) {
  const v = sampleKeys(keys, t);
  arm.setPose(new THREE.Vector3(v.x, v.y, v.z), v.lean, v.grip, t);
  if (t < graspT) {
    bowl.position.set(BOWL_START.x, 0, BOWL_START.z);
  } else if (t <= releaseT) {
    const p = arm.anchorWorld();
    bowl.position.set(p.x, p.y - BOWL_GRIP_DROP, p.z);
  } else {
    bowl.position.set(PLATE_POS.x, PLATE_DISH_Y + settleBounce(t - releaseT), PLATE_POS.z);
  }
}

function pushInCamera(camera, t, w0, w1, from, to, lookFrom, lookTo) {
  const u = smootherstep(clamp01((t - w0) / (w1 - w0)));
  camera.position.set(
    from.x + (to.x - from.x) * u,
    from.y + (to.y - from.y) * u,
    from.z + (to.z - from.z) * u,
  );
  camera.lookAt(
    lookFrom.x + (lookTo.x - lookFrom.x) * u,
    lookFrom.y + (lookTo.y - lookFrom.y) * u,
    lookFrom.z + (lookTo.z - lookFrom.z) * u,
  );
}

// ------------------------------------------------------------- the four scenes
function buildScene(name, envTex) {
  const scene = new THREE.Scene();
  scene.environment = envTex;
  scene.background = null;
  scene.environmentIntensity = 0.85;
  addLights(scene);

  if (name === "arm" || name === "episode") {
    const { t0, t1 } = WINDOWS[name];
    const { arm, bowl } = buildTaskWorld(scene, { withArm: true });
    const keys = name === "arm" ? pickKeys(t0, 1.0) : episodeKeys(t0);
    // Grasp = first key with grip >= 1; release = first grip-open key after it.
    const graspKey = keys.find((k) => k.v.grip >= 1).t;
    const releaseKey = keys.find((k) => k.v.grip === 0 && k.t > graspKey).t;
    const camera = makeCamera(36);
    const v = (x, y, z) => new THREE.Vector3(x, y, z);
    return {
      scene,
      camera,
      update(t) {
        updateArmBowl(arm, bowl, t, keys, graspKey, releaseKey);
        if (name === "arm") {
          pushInCamera(camera, t, t0, t1, v(1.05, 2.4, 7.1), v(1.05, 2.18, 6.15), v(1.18, 0.55, 0.05), v(1.18, 0.62, 0.05));
        } else {
          pushInCamera(camera, t, t0, t1, v(0.95, 2.45, 7.4), v(0.95, 2.2, 6.1), v(1.2, 0.55, 0.05), v(1.2, 0.62, 0.05));
        }
      },
    };
  }

  if (name === "policy") {
    const { t0, t1 } = WINDOWS.policy;
    buildTaskWorld(scene, { withArm: false });
    scene.add(createWorkshopClutter());
    // Warm fill low over the bench to sell the workshop vibe.
    const warm = new THREE.PointLight(0xffd9b0, 14, 10, 2);
    warm.position.set(0.6, 3.1, 1.4);
    scene.add(warm);
    const camera = makeCamera(40);
    const v = (x, y, z) => new THREE.Vector3(x, y, z);
    return {
      scene,
      camera,
      update(t) {
        pushInCamera(camera, t, t0, t1, v(2.4, 2.7, 4.9), v(1.95, 2.15, 4.05), v(0.63, 0.14, 0.57), v(0.63, 0.1, 0.57));
        camera.position.y += 0.015 * Math.sin(t * 0.8);
      },
    };
  }

  if (name === "success") {
    const { t0, t1 } = WINDOWS.success;
    // Diptych: two mini benches. Left: bowl ON plate. Right: bowl BESIDE plate.
    const left = createBench({ width: 3.5, depth: 2.9, thickness: 0.34 });
    left.position.set(-1.5, 0, 0);
    scene.add(left);
    const right = createBench({ width: 3.5, depth: 2.9, thickness: 0.34 });
    right.position.set(2.2, 0, 0);
    scene.add(right);
    scene.add(createShadowCatcher({ width: 9.2, depth: 3.4, y: 0.006 }));

    const plateL = createPlate();
    plateL.position.set(-1.5, 0, 0.18);
    scene.add(plateL);
    const bowlL = createBowl();
    bowlL.position.set(-1.5, PLATE_DISH_Y, 0.18);
    scene.add(bowlL);

    const plateR = createPlate();
    plateR.position.set(1.68, 0, -0.12);
    scene.add(plateR);
    const bowlR = createBowl();
    bowlR.position.set(2.95, 0, 0.62);
    bowlR.rotation.y = 0.6;
    scene.add(bowlR);

    const camera = makeCamera(36);
    return {
      scene,
      camera,
      update(t) {
        const sway = 0.16 * Math.sin(t * 0.55 + 0.8);
        camera.position.set(0.35 + sway, 2.75, 6.2);
        camera.lookAt(0.35 + sway * 0.7, 0.26, 0.05);
      },
    };
  }

  throw new Error(`unknown scene: ${name}`);
}

export function buildSceneByName(name, envTex) {
  return buildScene(name, envTex);
}
