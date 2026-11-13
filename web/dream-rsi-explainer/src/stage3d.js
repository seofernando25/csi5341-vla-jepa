// three.js layer: starfield, grid floor, particle brain with orbit rings, and the real Dream-RSI search tree.
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

function rng(seed) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function layoutTree(nodes) {
  const byId = new Map(nodes.map((n, i) => [n.id, { ...n, i, kids: [] }]));
  for (const n of byId.values()) if (n.parent) byId.get(n.parent).kids.push(n);
  const leaves = (n) => (n.kids.length ? n.kids.reduce((s, k) => s + leaves(k), 0) : 1);
  const rand = rng(7);
  const place = (n, depth, a0, a1) => {
    const a = (a0 + a1) / 2;
    const r = depth * 1.0;
    n.pos = new THREE.Vector3(Math.cos(a) * r, -depth * 0.35 + (rand() - 0.5) * 0.25, Math.sin(a) * r);
    n.depth = depth;
    let cur = a0;
    const total = leaves(n);
    for (const k of n.kids) {
      const span = ((a1 - a0) * leaves(k)) / total;
      place(k, depth + 1, cur, cur + span);
      cur += span;
    }
  };
  const root = byId.get("root");
  place(root, 0, 0, Math.PI * 2);
  return { byId, list: nodes.map((n) => byId.get(n.id)) };
}

export function createStage(container, W, H, treeNodes) {
  const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(W, H);
  renderer.setClearColor(0x06070d, 1);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x06070d, 0.028);
  const cam = new THREE.PerspectiveCamera(50, W / H, 0.1, 200);

  const target = new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType, samples: 4 });
  const composer = new EffectComposer(renderer, target);
  composer.setSize(W, H);
  composer.addPass(new RenderPass(scene, cam));
  const bloom = new UnrealBloomPass(new THREE.Vector2(W, H), 0.85, 0.7, 0.12);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  const rand = rng(1337);

  // starfield
  const SN = 1800;
  const sp = new Float32Array(SN * 3);
  for (let i = 0; i < SN; i++) {
    const r = 14 + rand() * 40;
    const th = rand() * Math.PI * 2;
    const ph = Math.acos(2 * rand() - 1);
    sp[i * 3] = r * Math.sin(ph) * Math.cos(th);
    sp[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
    sp[i * 3 + 2] = r * Math.cos(ph);
  }
  const sg = new THREE.BufferGeometry();
  sg.setAttribute("position", new THREE.BufferAttribute(sp, 3));
  const stars = new THREE.Points(
    sg,
    new THREE.PointsMaterial({ size: 0.09, color: 0x6aa8ff, transparent: true, opacity: 0.65, depthWrite: false }),
  );
  scene.add(stars);

  const grid = new THREE.GridHelper(80, 80, 0x1f4a7a, 0x0f2440);
  grid.position.y = -4;
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  scene.add(grid);

  // particle brain
  const BN = 3400;
  const base = new Float32Array(BN * 3);
  const ph = new Float32Array(BN);
  const dd = new Float32Array(BN);
  for (let i = 0; i < BN; i++) {
    const u = rand();
    const v = rand();
    const th = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);
    const lobe = 1 + 0.18 * Math.sin(3 * th) * Math.sin(2 * phi);
    const r = 2.1 * lobe * (0.9 + 0.12 * rand());
    base[i * 3] = r * Math.sin(phi) * Math.cos(th) * 1.15;
    base[i * 3 + 1] = r * Math.cos(phi) * 0.85;
    base[i * 3 + 2] = r * Math.sin(phi) * Math.sin(th);
    ph[i] = rand() * Math.PI * 2;
    dd[i] = rand();
  }
  const bgeo = new THREE.BufferGeometry();
  const bpos = new Float32Array(base);
  bgeo.setAttribute("position", new THREE.BufferAttribute(bpos, 3));
  const bmat = new THREE.PointsMaterial({
    size: 0.05,
    transparent: true,
    opacity: 1,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const brain = new THREE.Group();
  const bpts = new THREE.Points(bgeo, bmat);
  brain.add(bpts);
  const rings = [];
  for (let k = 0; k < 3; k++) {
    const pts = [];
    for (let i = 0; i <= 128; i++) {
      const a = (i / 128) * Math.PI * 2;
      pts.push(new THREE.Vector3(Math.cos(a) * (3 + k * 0.45), 0, Math.sin(a) * (3 + k * 0.45)));
    }
    const m = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending });
    const ring = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), m);
    ring.rotation.set(0.5 + k * 0.9, k * 0.7, k * 0.4);
    brain.add(ring);
    rings.push(ring);
  }
  scene.add(brain);

  // search tree
  const L = layoutTree(treeNodes);
  const NN = L.list.length;
  const treeG = new THREE.Group();
  const nodeMesh = new THREE.InstancedMesh(
    new THREE.IcosahedronGeometry(0.17, 1),
    new THREE.MeshBasicMaterial({ transparent: true }),
    NN,
  );
  const baseCol = L.list.map((n) => {
    if (n.id === "root") return new THREE.Color(0xffffff);
    if (n.id === "n0008") return new THREE.Color(0xffe14d);
    if (n.status === "ok") return new THREE.Color(0x3df0ff);
    if (n.status === "interrupted") return new THREE.Color(0x7a8199);
    return new THREE.Color(0xff4d4d);
  });
  baseCol.forEach((c, i) => nodeMesh.setColorAt(i, c));
  treeG.add(nodeMesh);
  const edgeIdx = L.list.filter((n) => n.parent).map((n) => [L.byId.get(n.parent), n]);
  const epos = new Float32Array(edgeIdx.length * 6);
  const egeo = new THREE.BufferGeometry();
  egeo.setAttribute("position", new THREE.BufferAttribute(epos, 3));
  const emat = new THREE.LineBasicMaterial({ color: 0x3d8fff, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending });
  treeG.add(new THREE.LineSegments(egeo, emat));
  const runners = [];
  const runnerMat = new THREE.MeshBasicMaterial({ color: 0xffe14d });
  for (let k = 0; k < 4; k++) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.13, 10, 10), runnerMat);
    m.visible = false;
    treeG.add(m);
    runners.push(m);
  }
  scene.add(treeG);
  const chain = (id) => {
    const out = [];
    for (let n = L.byId.get(id); n; n = n.parent ? L.byId.get(n.parent) : null) out.unshift(n.pos);
    return out;
  };
  const runnerPaths = ["n0020", "n0008", "n0024", "n0016"].map(chain);

  const dummy = new THREE.Object3D();
  const tmp = new THREE.Color();

  function render(t, s) {
    // s: {bv,bx,bs,tv,tx,ts,cz,cy, brainColor, shatter, pulse, grow(i), runT, runK, shake, edgeColor}
    cam.position.set(Math.sin(t * 0.17) * 1.4 + s.shake[0] * 0.25, s.cy + s.shake[1] * 0.25, s.cz);
    cam.lookAt(0, 0, 0);
    stars.rotation.y = t * 0.012;
    stars.rotation.x = t * 0.004;
    grid.position.z = (t * 0.6) % 1;
    grid.material.opacity = 0.35 + 0.2 * s.pulse;

    // brain
    brain.visible = s.bv > 0.01;
    if (brain.visible) {
      const k = s.shatter;
      const br = 1 + 0.045 * Math.sin(t * 3.2) + 0.06 * s.pulse;
      for (let i = 0; i < BN; i++) {
        const w = (1 + 0.06 * Math.sin(t * 3.4 + ph[i])) * br * (1 + k * (0.6 + 2.6 * dd[i]));
        bpos[i * 3] = base[i * 3] * w;
        bpos[i * 3 + 1] = base[i * 3 + 1] * w - k * k * dd[i] * 2.2;
        bpos[i * 3 + 2] = base[i * 3 + 2] * w;
      }
      bgeo.attributes.position.needsUpdate = true;
      bmat.color.setHex(s.brainColor);
      bmat.opacity = s.bv * (1 - 0.5 * k);
      brain.position.set(s.bx, 0, 0);
      brain.scale.setScalar(s.bs);
      brain.rotation.y = t * 0.35;
      brain.rotation.x = 0.18 + 0.06 * Math.sin(t * 0.5);
      rings.forEach((r, i) => {
        r.material.color.setHex(s.brainColor);
        r.material.opacity = 0.55 * s.bv * (1 - k);
        r.rotation.y += 0.004 * (i + 1);
        r.scale.setScalar(1 + 0.04 * s.pulse);
      });
    }

    // tree
    treeG.visible = s.tv > 0.01;
    if (treeG.visible) {
      treeG.position.set(s.tx, 0.2, 0);
      treeG.scale.setScalar(s.ts);
      treeG.rotation.set(-0.75, t * 0.12, 0);
      emat.opacity = 0.85 * s.tv;
      emat.color.setHex(s.edgeColor);
      nodeMesh.material.opacity = s.tv;
      for (let i = 0; i < NN; i++) {
        const n = L.list[i];
        const g = s.grow(i);
        const sc = g * (n.id === "root" ? 1.5 : n.id === "n0008" ? 1.5 : 1) * (1 + s.nodePulse(i));
        dummy.position.copy(n.pos);
        dummy.scale.setScalar(Math.max(0.0001, sc));
        dummy.updateMatrix();
        nodeMesh.setMatrixAt(i, dummy.matrix);
        tmp.copy(baseCol[i]);
        if (s.ghost > 0) tmp.lerp(new THREE.Color(0x6a7cff), s.ghost * 0.7);
        nodeMesh.setColorAt(i, tmp);
      }
      nodeMesh.instanceMatrix.needsUpdate = true;
      nodeMesh.instanceColor.needsUpdate = true;
      edgeIdx.forEach(([p, c], e) => {
        const g = s.grow(c.i);
        epos[e * 6] = p.pos.x;
        epos[e * 6 + 1] = p.pos.y;
        epos[e * 6 + 2] = p.pos.z;
        epos[e * 6 + 3] = p.pos.x + (c.pos.x - p.pos.x) * g;
        epos[e * 6 + 4] = p.pos.y + (c.pos.y - p.pos.y) * g;
        epos[e * 6 + 5] = p.pos.z + (c.pos.z - p.pos.z) * g;
      });
      egeo.attributes.position.needsUpdate = true;
      runners.forEach((m, k) => {
        m.visible = s.runK > 0 && k < s.runK;
        if (!m.visible) return;
        const path = runnerPaths[k];
        const u = (((s.runT * (0.9 + 0.35 * k) + k * 0.23) % 1) + 1) % 1;
        const f = u * (path.length - 1);
        const a = Math.min(path.length - 2, Math.floor(f));
        m.position.lerpVectors(path[a], path[a + 1], f - a);
      });
    }
    bloom.strength = 0.8 + 0.5 * s.pulse;
    composer.render();
  }
  return { render, layout: L };
}
