// Six-axis industrial arm built from primitives: base yaw, shoulder, elbow,
// wrist (brushed steel), two-finger gripper with rubber pads. Charcoal paint
// with one brown accent ring per joint. Local tool axis is +y; a lean of 0
// points the tool straight up, PI points it straight down. Planar 2-link IK
// places the wrist; yaw aims the arm plane at the target. Forward kinematics
// via the scene graph give the gripper anchor for the bowl.
import * as THREE from "three";
import { smootherstep } from "./motion.js";

export const ARM_BASE = { x: 2.42, z: -0.28 };
export const L1 = 1.5;
export const L2 = 1.4;
export const SHOULDER_Y = 0.89;
export const ANCHOR_LOCAL = 0.33; // gripper anchor (between fingers) above tool base
export const BOWL_GRIP_DROP = 0.27; // anchor → bowl base offset when grasping
const OPEN_OFF = 0.42;
const CLOSED_OFF = 0.3;

const CHARCOAL = 0x2b2824;
const BROWN = 0x604737;
const STEEL = 0xb9b5ab;
const RUBBER = 0x14120e;

function std(params) {
  return new THREE.MeshStandardMaterial(params);
}

function mesh(geo, mat, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(geo, mat);
  m.position.set(x, y, z);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

export function createArm() {
  const charcoal = std({ color: CHARCOAL, roughness: 0.55, metalness: 0.28 });
  const brown = std({ color: BROWN, roughness: 0.5, metalness: 0.22 });
  const steel = std({ color: STEEL, roughness: 0.3, metalness: 0.9 });
  const rubber = std({ color: RUBBER, roughness: 0.95, metalness: 0.0 });

  const root = new THREE.Group();
  root.name = "arm";
  root.position.set(ARM_BASE.x, 0, ARM_BASE.z);

  // --- static base
  root.add(mesh(new THREE.CylinderGeometry(0.68, 0.72, 0.09, 48), charcoal, 0, 0.045, 0));
  root.add(mesh(new THREE.CylinderGeometry(0.54, 0.6, 0.3, 48), charcoal, 0, 0.24, 0));
  root.add(mesh(new THREE.CylinderGeometry(0.555, 0.555, 0.065, 48), brown, 0, 0.415, 0));

  // --- yaw stage
  const yawGroup = new THREE.Group();
  yawGroup.position.y = 0.45;
  root.add(yawGroup);
  yawGroup.add(mesh(new THREE.CylinderGeometry(0.375, 0.43, 0.3, 40), charcoal, 0, 0.15, 0));
  yawGroup.add(mesh(new THREE.CylinderGeometry(0.31, 0.31, 0.06, 40), brown, 0, 0.33, 0));

  // --- shoulder
  const shoulderGroup = new THREE.Group();
  shoulderGroup.position.y = SHOULDER_Y - 0.45; // local to yawGroup
  yawGroup.add(shoulderGroup);
  const shoulderHousing = mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.52, 36), charcoal);
  shoulderHousing.rotation.x = Math.PI / 2;
  shoulderGroup.add(shoulderHousing);
  for (const s of [1, -1]) {
    const cap = mesh(new THREE.CylinderGeometry(0.305, 0.305, 0.07, 36), brown, 0, 0, s * 0.285);
    cap.rotation.x = Math.PI / 2;
    shoulderGroup.add(cap);
    shoulderGroup.add(mesh(new THREE.SphereGeometry(0.068, 16, 12), steel, 0, 0, s * 0.335));
  }
  shoulderGroup.add(mesh(new THREE.CylinderGeometry(0.16, 0.235, L1, 28), charcoal, 0, L1 / 2, 0));
  shoulderGroup.add(mesh(new THREE.CylinderGeometry(0.205, 0.215, 0.12, 28), brown, 0, 1.02, 0));

  // --- elbow
  const elbowGroup = new THREE.Group();
  elbowGroup.position.y = L1;
  shoulderGroup.add(elbowGroup);
  const elbowHousing = mesh(new THREE.CylinderGeometry(0.235, 0.235, 0.4, 32), charcoal);
  elbowHousing.rotation.x = Math.PI / 2;
  elbowGroup.add(elbowHousing);
  for (const s of [1, -1]) {
    const cap = mesh(new THREE.CylinderGeometry(0.24, 0.24, 0.06, 32), brown, 0, 0, s * 0.22);
    cap.rotation.x = Math.PI / 2;
    elbowGroup.add(cap);
    elbowGroup.add(mesh(new THREE.SphereGeometry(0.055, 16, 12), steel, 0, 0, s * 0.26));
  }
  elbowGroup.add(mesh(new THREE.CylinderGeometry(0.115, 0.18, L2, 24), charcoal, 0, L2 / 2, 0));
  elbowGroup.add(mesh(new THREE.CylinderGeometry(0.165, 0.172, 0.09, 24), steel, 0, 0.2, 0));

  // --- wrist (brushed steel)
  const wristGroup = new THREE.Group();
  wristGroup.position.y = L2;
  elbowGroup.add(wristGroup);
  wristGroup.add(mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.19, 32), steel, 0, 0.095, 0));
  wristGroup.add(mesh(new THREE.CylinderGeometry(0.152, 0.152, 0.05, 32), brown, 0, 0.205, 0));

  // --- two-finger parallel-jaw gripper (rail-mounted on a wide palm)
  const gripperGroup = new THREE.Group();
  gripperGroup.position.y = 0.23;
  wristGroup.add(gripperGroup);
  gripperGroup.add(mesh(new THREE.BoxGeometry(0.8, 0.17, 0.28), charcoal, 0, 0.085, 0));
  gripperGroup.add(mesh(new THREE.BoxGeometry(0.76, 0.05, 0.24), steel, 0, 0.185, 0));

  const fingers = [];
  for (const s of [1, -1]) {
    const finger = new THREE.Group();
    finger.position.set(s * OPEN_OFF, 0.17, 0);
    finger.add(mesh(new THREE.BoxGeometry(0.12, 0.55, 0.15), charcoal, 0, 0.3, 0));
    const pin = mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.15, 18), steel, 0, 0.08, 0);
    pin.rotation.x = Math.PI / 2;
    finger.add(pin);
    // Rubber pad on the inner face.
    finger.add(mesh(new THREE.BoxGeometry(0.035, 0.36, 0.115), rubber, -s * 0.0775, 0.33, 0));
    gripperGroup.add(finger);
    fingers.push(finger);
  }

  // Gripper anchor between the fingers; the bowl hangs from it.
  const anchor = new THREE.Object3D();
  anchor.position.set(0, ANCHOR_LOCAL, 0);
  gripperGroup.add(anchor);

  function setGrip(g) {
    const off = OPEN_OFF + (CLOSED_OFF - OPEN_OFF) * smootherstep(g);
    fingers[0].position.x = off;
    fingers[1].position.x = -off;
  }

  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

  // Pose the whole chain. target is the WRIST position in world space;
  // toolLean is the absolute tool-axis lean from vertical (PI = straight down).
  // Returns nothing; call anchorWorld() for the grip point.
  function setPose(target, toolLean, grip, t = 0) {
    const dx = target.x - ARM_BASE.x;
    const dz = target.z - ARM_BASE.z;
    const r = Math.hypot(dx, dz);
    const dy = target.y - SHOULDER_Y;
    const reach = L1 + L2 - 1e-3;
    const D = clamp(Math.hypot(r, dy), 0.35, reach);
    const beta = Math.atan2(r, dy);
    const alpha = Math.acos(clamp((L1 * L1 + D * D - L2 * L2) / (2 * L1 * D), -1, 1));
    const phi1 = beta - alpha; // elbow-up posture
    const gamma = Math.acos(clamp((L1 * L1 + L2 * L2 - D * D) / (2 * L1 * L2), -1, 1));
    const phi2 = phi1 + (Math.PI - gamma);

    // Subtle idle breathing so the arm never feels frozen.
    const sway = 0.008 * Math.sin(t * 1.35 + 0.7);
    yawGroup.rotation.y = Math.atan2(-dz, dx) + 0.01 * Math.sin(t * 1.05);
    shoulderGroup.rotation.z = -(phi1 + sway);
    elbowGroup.rotation.z = -(phi2 - phi1);
    wristGroup.rotation.z = -(toolLean - phi2 + sway * 0.5);
    setGrip(grip);
  }

  const _v = new THREE.Vector3();
  function anchorWorld() {
    root.updateMatrixWorld(true);
    return anchor.getWorldPosition(_v);
  }

  setGrip(0);
  root.updateMatrixWorld(true);
  return { root, yawGroup, shoulderGroup, elbowGroup, wristGroup, gripperGroup, anchor, anchorWorld, setPose, setGrip };
}
