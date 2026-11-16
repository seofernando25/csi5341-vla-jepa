// Shared props for the world3d scenes: bench slab, black bowl and ceramic
// plate (both LatheGeometry), workshop clutter for the warm counter shot, and
// the ShadowMaterial catcher that grounds everything on the transparent paper.
// One root Group per prop; no text anywhere.
import * as THREE from "three";

const V2 = (x, y) => new THREE.Vector2(x, y);

export const BENCH_COLOR = 0x77604a; // warm muted wood-brown
export const BENCH_COLOR_DEEP = 0x5d4a38;
export const PLATE_COLOR = 0xf1eee6; // light ceramic
export const BOWL_COLOR = 0x17150f; // near-black ceramic

function std(params) {
  return new THREE.MeshStandardMaterial(params);
}

export function createBench({ width = 12.6, depth = 6.9, thickness = 0.38, color = BENCH_COLOR } = {}) {
  const root = new THREE.Group();
  root.name = "bench";
  const slab = new THREE.Mesh(
    new THREE.BoxGeometry(width, thickness, depth),
    std({ color, roughness: 0.82, metalness: 0.04 }),
  );
  slab.position.y = -thickness / 2;
  slab.receiveShadow = true;
  slab.castShadow = true;
  root.add(slab);
  // Thin darker front edge so the slab reads as a crafted tabletop, not a box.
  const edge = new THREE.Mesh(
    new THREE.BoxGeometry(width, 0.07, 0.05),
    std({ color: BENCH_COLOR_DEEP, roughness: 0.85, metalness: 0.02 }),
  );
  edge.position.set(0, -0.035, depth / 2 + 0.012);
  edge.receiveShadow = true;
  root.add(edge);
  return root;
}

export function createBowl() {
  const root = new THREE.Group();
  root.name = "bowl";
  // Outer shell up to the rim, then back down the inside: one closed lathe.
  const pts = [
    V2(0.0, 0.012), V2(0.19, 0.012), V2(0.26, 0.045), V2(0.315, 0.09),
    V2(0.375, 0.165), V2(0.425, 0.27), V2(0.445, 0.36), V2(0.45, 0.425),
    V2(0.435, 0.438), V2(0.415, 0.365), V2(0.38, 0.255), V2(0.325, 0.15),
    V2(0.215, 0.088), V2(0.0, 0.062),
  ];
  const geo = new THREE.LatheGeometry(pts, 48);
  const mesh = new THREE.Mesh(
    geo,
    std({ color: BOWL_COLOR, roughness: 0.36, metalness: 0.08, side: THREE.DoubleSide }),
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  root.add(mesh);
  return root;
}

// Interior dish height; the bowl rests on the plate at this y.
export const PLATE_DISH_Y = 0.056;

export function createPlate() {
  const root = new THREE.Group();
  root.name = "plate";
  const pts = [
    V2(0.0, 0.0), V2(0.26, 0.0), V2(0.305, 0.018), V2(0.35, 0.033),
    V2(0.56, 0.047), V2(0.79, 0.072), V2(0.9, 0.108), V2(0.96, 0.148),
    V2(0.965, 0.158), V2(0.925, 0.158), V2(0.86, 0.118), V2(0.62, 0.076),
    V2(0.35, 0.059), V2(0.0, 0.054),
  ];
  const geo = new THREE.LatheGeometry(pts, 56);
  const mesh = new THREE.Mesh(
    geo,
    std({ color: PLATE_COLOR, roughness: 0.34, metalness: 0.0, side: THREE.DoubleSide }),
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  root.add(mesh);
  return root;
}

// Workshop clutter, kept to the right side so the left third stays quiet.
export function createWorkshopClutter() {
  const root = new THREE.Group();
  root.name = "clutter";

  // Wooden crate, back right.
  const crate = new THREE.Group();
  const crateMat = std({ color: 0x6e5238, roughness: 0.85, metalness: 0.02 });
  const crateDark = std({ color: 0x54402c, roughness: 0.85, metalness: 0.02 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.78, 0.8), crateMat);
  body.position.y = 0.39;
  body.castShadow = true;
  body.receiveShadow = true;
  crate.add(body);
  for (const dy of [0.12, 0.66]) {
    const rail = new THREE.Mesh(new THREE.BoxGeometry(1.09, 0.09, 0.84), crateDark);
    rail.position.y = dy;
    rail.castShadow = true;
    crate.add(rail);
  }
  crate.position.set(3.15, 0, -1.55);
  crate.rotation.y = -0.28;
  root.add(crate);

  // Two muted tins beside the crate.
  const tinA = new THREE.Mesh(
    new THREE.CylinderGeometry(0.21, 0.21, 0.52, 28),
    std({ color: 0x8a5a3a, roughness: 0.5, metalness: 0.35 }),
  );
  tinA.position.set(2.15, 0.26, -1.85);
  tinA.castShadow = true;
  tinA.receiveShadow = true;
  root.add(tinA);

  const tinB = new THREE.Mesh(
    new THREE.CylinderGeometry(0.16, 0.16, 0.36, 28),
    std({ color: 0x7a7568, roughness: 0.55, metalness: 0.3 }),
  );
  tinB.position.set(1.55, 0.18, -2.05);
  tinB.castShadow = true;
  tinB.receiveShadow = true;
  root.add(tinB);

  // Low stack of blocks, mid-back, keeps the eye on the task but adds depth.
  const block = new THREE.Mesh(
    new THREE.BoxGeometry(0.62, 0.3, 0.5),
    std({ color: 0x8f8574, roughness: 0.9, metalness: 0.0 }),
  );
  block.position.set(0.9, 0.15, -2.15);
  block.rotation.y = 0.22;
  block.castShadow = true;
  block.receiveShadow = true;
  root.add(block);

  return root;
}

// ShadowMaterial plane just above the bench top: renders only the shadow, so
// it works on the transparent paper backdrop.
export function createShadowCatcher({ width = 12.6, depth = 6.9, y = 0.006 } = {}) {
  const mat = new THREE.ShadowMaterial({ color: 0x1f1c17, opacity: 0.3, depthWrite: false });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, depth), mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = y;
  mesh.receiveShadow = true;
  mesh.name = "shadowCatcher";
  return mesh;
}

export function addShadows(root) {
  root.traverse((o) => {
    if (o.isMesh && o.material && o.material.isShadowMaterial !== true) {
      o.castShadow = true;
    }
  });
}
