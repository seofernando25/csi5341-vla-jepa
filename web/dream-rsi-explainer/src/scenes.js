// Scene infographics + HUD drawn onto the 2D canvas over the 3D stage.
import { appear, clamp, kf, lerp, outBack, outCubic } from "./ease.js";

const F = (w, sz, mono = false) => `${w} ${sz}px ${mono ? "'JetBrains Mono'" : "Inter"}, sans-serif`;

function rrect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function card(ctx, x, y, w, h, stroke = "rgba(61,240,255,0.45)", r = 18, fill = "rgba(10,14,28,0.82)") {
  rrect(ctx, x, y, w, h, r);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = stroke;
  ctx.stroke();
}

function txt(ctx, s, x, y, sz = 24, col = "#fff", align = "left", w = 800, mono = false) {
  ctx.font = F(w, sz, mono);
  ctx.fillStyle = col;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  ctx.fillText(s, x, y);
}

function pill(ctx, s, x, y, bg, fg = "#06070d", sz = 18) {
  ctx.font = F(800, sz, true);
  const w = ctx.measureText(s).width + 26;
  rrect(ctx, x, y - 16, w, 32, 16);
  ctx.fillStyle = bg;
  ctx.fill();
  txt(ctx, s, x + w / 2, y, sz, fg, "center", 800, true);
  return w;
}

export function sceneIndexAt(t, scenes) {
  for (let i = scenes.length - 1; i >= 0; i--) if (t >= scenes[i].start - 0.3) return i;
  return 0;
}

export function drawHUD(ctx, t, tl, cues, treeNodes, W, H) {
  const idx = sceneIndexAt(t, tl.scenes);
  const sc = tl.scenes[idx];
  const s0 = idx === 0 ? 0 : sc.start - 0.3;

  // top bar: scene chip + brand + progress
  const ce = outBack(appear(t, s0, 0.35));
  ctx.save();
  ctx.translate(48, 38);
  ctx.scale(ce, ce);
  pill(ctx, `0${idx + 1} · ${sc.chip}`, 0, 0, "#3df0ff", "#06070d", 16);
  ctx.restore();
  txt(ctx, "CSI 5341 · VLA-JEPA", W - 48, 38, 16, "rgba(255,255,255,0.55)", "right", 700, true);
  ctx.fillStyle = "rgba(255,255,255,0.12)";
  ctx.fillRect(0, 0, W, 4);
  const g = ctx.createLinearGradient(0, 0, W, 0);
  g.addColorStop(0, "#3df0ff");
  g.addColorStop(1, "#ff3d9a");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W * clamp(t / tl.total), 4);

  const id = sc.id;
  if (id === "hook") drawHook(ctx, t, cues, W, H);
  else if (id === "idea") drawIdea(ctx, t, s0, cues, W);
  else if (id === "dream") drawDream(ctx, t, s0, cues, W);
  else if (id === "loop") drawLoop(ctx, t, s0, cues, W);
  else if (id === "code") drawCode(ctx, t, s0, cues, W);
  else if (id === "arms") drawArms(ctx, t, s0, cues, W);
  else if (id === "project") drawProject(ctx, t, s0, cues, treeNodes, W);
  else if (id === "fail") drawFail(ctx, t, s0, cues, W, H);
  else if (id === "audit") drawAudit(ctx, t, s0, cues, W);
  else if (id === "fix") drawFix(ctx, t, s0, cues, W);
  else if (id === "live") drawLive(ctx, t, s0, cues, W);
  else if (id === "jump") drawJump(ctx, t, s0, cues, W);
  else if (id === "tasks") drawTasks(ctx, t, s0, cues, W);
  else if (id === "close") drawClose(ctx, t, s0, cues, tl.total, W, H);
}

function drawHook(ctx, t, c, W, H) {
  const k = outBack(appear(t, 0.3, 0.45));
  const kb = outBack(appear(t, c.better - 0.15, 0.35));
  const kz = appear(t, c.zero - 0.05, 0.22);
  ctx.save();
  ctx.translate(64, 150);
  ctx.scale(k, k);
  card(ctx, 0, 0, 470, 290, kb > 0 ? "#5dff8f" : "rgba(61,240,255,0.5)");
  pill(ctx, "CANDIDATE n0008", 28, 42, "#3df0ff");
  txt(ctx, "HELD-OUT ACTION LOSS", 28, 92, 18, "#8fa3bf", "left", 700, true);
  const v = lerp(0.32, 0.28, clamp((t - c.better) / 1.5));
  txt(ctx, v.toFixed(2), 28, 165, 86, kb > 0 ? "#5dff8f" : "#ffffff", "left", 900, true);
  txt(ctx, "baseline 0.32", 260, 180, 22, "#8fa3bf", "left", 700, true);
  if (kb > 0) {
    ctx.save();
    ctx.translate(28, 240);
    ctx.scale(kb, kb);
    pill(ctx, "✓ −13.6% LOSS · RANK #1 OF 27", 0, 0, "#5dff8f", "#06070d", 17);
    ctx.restore();
  }
  ctx.restore();

  if (kz > 0) {
    const flash = Math.max(0, 1 - (t - c.zero) * 2.5);
    ctx.fillStyle = `rgba(255,45,75,${flash * 0.42})`;
    ctx.fillRect(0, 0, W, H);
    const s = outBack(kz, 2.8);
    ctx.save();
    ctx.translate(880, 295);
    ctx.rotate(-0.05);
    ctx.scale(s, s);
    card(ctx, -290, -145, 580, 290, "#ff4d4d", 24, "rgba(30,6,14,0.92)");
    ctx.shadowColor = "rgba(255,77,77,0.85)";
    ctx.shadowBlur = 36;
    txt(ctx, "0 / 10", 0, -22, 138, "#ff4d4d", "center", 900, true);
    ctx.shadowBlur = 0;
    txt(ctx, "REAL ROBOT TRIALS PASSED", 0, 88, 22, "#ffd6d6", "center", 800, true);
    ctx.restore();
  }
}

function drawIdea(ctx, t, s0, c, W) {
  const kt = outBack(appear(t, c.title - 0.1, 0.4));
  const lift = smoothMove(t, c.title, c.gpu);
  ctx.save();
  ctx.translate(lerp(W / 2, 290, lift), lerp(250, 135, lift));
  const sc = lerp(1, 0.62, lift) * kt;
  ctx.scale(sc, sc);
  ctx.shadowColor = "rgba(61,240,255,0.6)";
  ctx.shadowBlur = 30;
  txt(ctx, "DREAM-RSI", 0, 0, 104, "#ffffff", "center", 900);
  ctx.shadowBlur = 0;
  txt(ctx, "RECURSIVE SELF-IMPROVING EXPLORATION · arXiv:2609.14858", 0, 68, 20, "#3df0ff", "center", 700, true);
  ctx.restore();

  const kg = outBack(appear(t, c.gpu - 0.3, 0.4));
  if (kg > 0) {
    const n = Math.min(27, Math.max(1, Math.floor((t - (c.gpu - 0.6)) * 6.2)));
    const hot = t >= c.gpu - 0.1;
    ctx.save();
    ctx.translate(64, 210);
    ctx.scale(kg, kg);
    card(ctx, 0, 0, 440, 250, hot ? "#ff4d4d" : "rgba(61,240,255,0.45)");
    txt(ctx, "LIVE GPU SEARCH", 28, 42, 18, "#8fa3bf", "left", 700, true);
    txt(ctx, `${n}`, 28, 125, 92, hot ? "#ff4d4d" : "#3df0ff", "left", 900, true);
    txt(ctx, "CANDIDATES TESTED", 165, 112, 22, "#ffffff", "left", 800);
    txt(ctx, "500–1,500 GPU steps each", 165, 146, 18, "#8fa3bf", "left", 700, true);
    pill(ctx, hot ? "🔥 EVERY BRANCH BURNS GPU" : "1 NODE = 1 TRAINING RUN", 28, 210, hot ? "#ff4d4d" : "#ffe14d");
    ctx.restore();
  }
  const kw = outBack(appear(t, c.who - 0.1, 0.35), 2.5);
  if (kw > 0) {
    ctx.save();
    ctx.translate(880, 290);
    ctx.scale(kw, kw);
    pill(ctx, "❓ EXPLORATION POLICY π: WHICH BRANCH NEXT?", -260, 0, "#ff3d9a", "#ffffff", 20);
    ctx.restore();
  }
}

function drawDream(ctx, t, s0, c, W) {
  const k0 = outBack(appear(t, s0 + 0.3, 0.4));
  ctx.save();
  ctx.translate(860, 125);
  ctx.scale(k0, k0);
  card(ctx, -310, 0, 620, 118, "rgba(122,140,255,0.6)");
  pill(ctx, "REPLAY SIMULATOR", -282, 34, "#7a8cff", "#06070d", 16);
  txt(ctx, "27 recorded attempts → offline search tree", -282, 80, 22, "#ffffff", "left", 700);
  ctx.restore();

  const kw = outBack(appear(t, c.walk - 0.1, 0.35));
  if (kw > 0) {
    const ms = Math.min(16, ((t - c.walk) * 14).toFixed(0));
    const ng = t >= c.nogpu - 0.1;
    ctx.save();
    ctx.translate(860, 265);
    ctx.scale(kw, kw);
    card(ctx, -310, 0, 620, 118, ng ? "#5dff8f" : "#ffe14d");
    txt(ctx, "OFFLINE POLICY REPLAY", -282, 34, 17, "#8fa3bf", "left", 700, true);
    txt(ctx, `${ms} ms`, -282, 80, 48, "#ffe14d", "left", 900, true);
    if (ng) pill(ctx, "0 GPU RUNS · FREE FEEDBACK", -60, 80, "#5dff8f", "#06070d", 18);
    ctx.restore();
  }

  const kk = outBack(appear(t, c.keep - 0.15, 0.38));
  if (kk > 0) {
    const nv = outBack(appear(t, c.never - 0.1, 0.35), 2.4);
    ctx.save();
    ctx.translate(860, 405);
    ctx.scale(kk, kk);
    card(ctx, -310, 0, 620, 118, "#3df0ff");
    txt(ctx, "MONOTONIC SELECTION · argmax V(π)", -282, 32, 17, "#3df0ff", "left", 800, true);
    txt(ctx, "π₀ serial  0.19", -282, 78, 22, "#8fa3bf", "left", 700, true);
    txt(ctx, "→  π₁ portfolio  0.34  ✓", -70, 78, 24, "#5dff8f", "left", 800, true);
    if (nv > 0) {
      ctx.save();
      ctx.translate(210, 32);
      ctx.scale(nv, nv);
      pill(ctx, "LOCKED ≥", -70, 0, "#ffe14d", "#06070d", 16);
      ctx.restore();
    }
    ctx.restore();
  }
}

function drawLoop(ctx, t, s0, c, W) {
  const cx = W / 2, cy = 295, R = 165;
  const nodes = [
    ["01 · SEARCH", "run policy online", c.l1, "#3df0ff", -Math.PI / 2],
    ["02 · REMEMBER", "append tree to H", c.l2, "#7a8cff", 0],
    ["03 · DREAM", "replay policies offline", c.l3, "#ff3d9a", Math.PI / 2],
    ["04 · SELECT", "keep argmax V", c.l4, "#5dff8f", Math.PI],
  ];
  ctx.save();
  ctx.strokeStyle = "rgba(61,240,255,0.28)";
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, Math.PI * 2 * clamp((t - s0) / 1.2));
  ctx.stroke();
  const ang = t * 1.9;
  ctx.fillStyle = "#ffe14d";
  ctx.beginPath();
  ctx.arc(cx + Math.cos(ang) * R, cy + Math.sin(ang) * R, 9, 0, Math.PI * 2);
  ctx.fill();
  txt(ctx, "DREAM-RSI", cx, cy - 12, 30, "#ffffff", "center", 900);
  txt(ctx, "SELF-IMPROVING LOOP", cx, cy + 20, 15, "#3df0ff", "center", 700, true);

  for (const [title, sub, cue, col, a] of nodes) {
    const x = cx + Math.cos(a) * (R + 150) * (a === 0 || a === Math.PI ? 1.15 : 0.0);
    const y = cy + Math.sin(a) * (R + 30);
    const on = t >= cue - 0.12;
    const sc = on ? outBack(appear(t, cue - 0.12, 0.32), 2.2) : 0.88 * outCubic(appear(t, s0 + 0.2, 0.4));
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(sc, sc);
    card(ctx, -155, -48, 310, 96, on ? col : "rgba(255,255,255,0.18)", 16, on ? "rgba(14,20,40,0.95)" : "rgba(10,14,28,0.65)");
    txt(ctx, title, 0, -14, 22, on ? col : "#8fa3bf", "center", 900, true);
    txt(ctx, sub, 0, 20, 17, "#ffffff", "center", 600);
    ctx.restore();
  }
  ctx.restore();
}

function drawProject(ctx, t, s0, c, treeNodes, W) {
  const k0 = outBack(appear(t, s0 + 0.3, 0.4));
  const smol = appear(t, c.smol - 0.1, 0.45);
  ctx.save();
  ctx.translate(64, 115);
  ctx.scale(k0, k0);
  card(ctx, 0, 0, 560, 185, smol > 0 ? "#ff3d9a" : "#3df0ff");
  pill(ctx, smol > 0 ? "ARM 1 · BACKBONE SWAP" : "RELEASED BASELINE · B16", 24, 36, smol > 0 ? "#ff3d9a" : "#3df0ff", "#06070d", 15);
  txt(ctx, smol > 0 ? "Qwen3-VL  →  SmolVLM2-500M" : "VLA-JEPA (Qwen3-VL + Action Head)", 24, 82, 26, "#ffffff", "left", 800);
  const mb = Math.round(lerp(5532, 3041, outCubic(smol)));
  ctx.fillStyle = "rgba(255,255,255,0.14)";
  rrect(ctx, 24, 118, 512, 42, 10);
  ctx.fill();
  ctx.fillStyle = smol > 0 ? "#ff3d9a" : "#3df0ff";
  rrect(ctx, 24, 118, 512 * lerp(1, 3041 / 5532, outCubic(smol)), 42, 10);
  ctx.fill();
  txt(ctx, smol > 0 ? "500M params · 2.97 GiB peak" : `${mb} MB resident (5.5 GB)`, 40, 139, 19, "#ffffff", "left", 900, true);
  ctx.restore();

  const kl = outBack(appear(t, c.llm - 0.15, 0.4));
  if (kl > 0) {
    ctx.save();
    ctx.translate(64, 325);
    ctx.scale(kl, kl);
    card(ctx, 0, 0, 560, 190, "rgba(255,225,77,0.55)");
    txt(ctx, "LLM-PROPOSED CONDITIONING ADAPTERS · 27 RUNS", 24, 32, 16, "#ffe14d", "left", 800, true);
    const real = treeNodes.slice(1);
    const rev = Math.floor(clamp((t - c.llm) / 2.2) * 27);
    real.forEach((n, i) => {
      const gx = 38 + (i % 9) * 36;
      const gy = 76 + Math.floor(i / 9) * 38;
      ctx.beginPath();
      ctx.arc(gx, gy, i < rev ? (n.id === "n0008" ? 13 : 10) : 5, 0, Math.PI * 2);
      ctx.fillStyle = i >= rev ? "rgba(255,255,255,0.18)" : n.id === "n0008" ? "#ffe14d" : n.status === "ok" ? "#3df0ff" : "#ff4d4d";
      ctx.fill();
    });
    const kloss = appear(t, c.loss - 0.15, 0.45);
    if (kloss > 0) {
      const v = lerp(0.3197, 0.2762, outCubic(kloss));
      txt(ctx, "BEST n0008 LOSS", 365, 76, 15, "#8fa3bf", "left", 700, true);
      txt(ctx, v.toFixed(2), 365, 130, 58, "#5dff8f", "left", 900, true);
      txt(ctx, "from 0.32", 495, 138, 17, "#8fa3bf", "left", 700, true);
    }
    ctx.restore();
  }
}

function drawTabletop(ctx, x, y, w, h, crush) {
  ctx.save();
  rrect(ctx, x, y, w, h, 14);
  ctx.clip();
  // normal vibrant tabletop scene
  ctx.fillStyle = "#1c2538";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = "#b9895b";
  ctx.fillRect(x, y + h * 0.42, w, h * 0.58);
  // target plate
  ctx.fillStyle = "#f2f4f8";
  ctx.beginPath();
  ctx.ellipse(x + w * 0.68, y + h * 0.68, 42, 22, 0, 0, Math.PI * 2);
  ctx.fill();
  // red bowl
  ctx.fillStyle = "#e63946";
  ctx.beginPath();
  ctx.ellipse(x + w * 0.32, y + h * 0.66, 32, 18, 0, 0, Math.PI * 2);
  ctx.fill();
  // robot gripper
  ctx.strokeStyle = "#3df0ff";
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.moveTo(x + w * 0.35, y);
  ctx.lineTo(x + w * 0.33, y + h * 0.42);
  ctx.stroke();
  if (crush > 0) {
    ctx.fillStyle = `rgba(8,10,16,${0.975 * crush})`;
    ctx.fillRect(x, y, w, h);
  }
  ctx.restore();
  rrect(ctx, x, y, w, h, 14);
  ctx.strokeStyle = crush > 0.5 ? "#ff4d4d" : "#3df0ff";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawFail(ctx, t, s0, c, W, H) {
  const k0 = outBack(appear(t, s0 + 0.25, 0.35));
  const steps = Math.min(10000, Math.round(clamp((t - s0) / 1.7) * 10000));
  const kz = appear(t, c.zero2 - 0.08, 0.22);
  ctx.save();
  ctx.translate(64, 110);
  ctx.scale(k0, k0);
  card(ctx, 0, 0, 460, 175, kz > 0 ? "#ff4d4d" : "#3df0ff");
  txt(ctx, "CLOUD CONFIRMATION · RTX 5090", 24, 34, 16, "#8fa3bf", "left", 700, true);
  txt(ctx, `${steps.toLocaleString()} steps`, 24, 82, 36, "#ffffff", "left", 900, true);
  txt(ctx, "loss 0.265 · $2.17", 24, 134, 20, "#5dff8f", "left", 700, true);
  if (kz > 0) {
    const s = outBack(kz, 2.6);
    ctx.save();
    ctx.translate(362, 110);
    ctx.scale(s, s);
    pill(ctx, "0 / 10", -65, 0, "#ff4d4d", "#ffffff", 32);
    ctx.restore();
  }
  ctx.restore();

  const kt = outBack(appear(t, c.twice - 0.2, 0.4));
  if (kt > 0) {
    const kx = outCubic(appear(t, c.x255 - 0.15, 0.5));
    ctx.save();
    ctx.translate(555, 110);
    ctx.scale(kt, kt);
    card(ctx, 0, 0, 660, 395, "#ff4d4d");
    pill(ctx, "ROOT CAUSE · DOUBLE ÷255 RESCALE", 24, 36, "#ff4d4d", "#ffffff", 16);
    drawTabletop(ctx, 28, 74, 260, 175, 0);
    txt(ctx, "CAMERA [0, 1]", 158, 272, 17, "#3df0ff", "center", 800, true);
    txt(ctx, "÷ 255", 330, 145, 22, "#ffe14d", "center", 900, true);
    txt(ctx, "÷ 255", 330, 180, 22, "#ff4d4d", "center", 900, true);
    drawTabletop(ctx, 372, 74, 260, 175, kx);
    txt(ctx, kx > 0.3 ? "MODEL INPUT [0, 0.004] · MUSH" : "EXPECTED INPUT", 502, 272, 17, "#ff4d4d", "center", 800, true);
    if (kx > 0) {
      txt(ctx, "255× LOWER CONTRAST — RANKED BLIND BRAINS", 330, 338, 22, "#ffe14d", "center", 900, true);
    }
    ctx.restore();
  }

  const kl = appear(t, c.lossnot - 0.08, 0.25);
  if (kl > 0) {
    const s = outBack(kl, 2.6);
    ctx.save();
    ctx.translate(295, 405);
    ctx.rotate(-0.04);
    ctx.scale(s, s);
    card(ctx, -230, -65, 460, 130, "#ffe14d", 20, "rgba(25,18,6,0.95)");
    txt(ctx, "LOSS ≠ SUCCESS", 0, 0, 48, "#ffe14d", "center", 900, true);
    ctx.restore();
  }
}

function drawAudit(ctx, t, s0, c, W) {
  const k0 = outBack(appear(t, s0 + 0.25, 0.4));
  ctx.save();
  ctx.translate(W / 2, 300);
  ctx.scale(k0, k0);
  card(ctx, -520, -185, 1040, 380, "#3df0ff");
  pill(ctx, "FIDELITY AUDIT vs arXiv:2609.14858v2 · RTX 5060 Ti, commit ef1535e", -485, -142, "#3df0ff", "#06070d", 15);
  txt(ctx, "Stage 2: post-training Qwen compression, 8 smoke probes", -485, -100, 20, "#8fa3bf", "left", 600);
  const rows = [
    ["Eq.(1) SCORING", "FAITHFUL ✓", "#5dff8f", c.faithful],
    ["REPLAY COVERAGE", "IN-SAMPLE ✗", "#ff4d4d", c.insample],
    ["BASE ARM", "T2 compact, NOT report C2 ✗", "#ff4d4d", c.t2],
    ["8/10 SMOKE GATE", "13.08% FALSE KILLS ✗", "#ff4d4d", c.fkill],
    ["M=4 REVISION + FROZEN CONTROL", "MISSING ✗", "#ffe14d", c.missing],
  ];
  rows.forEach(([label, verdict, col, cue], i) => {
    const kr = outBack(appear(t, cue - 0.1, 0.32));
    const y = -52 + i * 56;
    ctx.save();
    ctx.globalAlpha = kr > 0 ? 1 : 0.25;
    const sc = kr > 0 ? kr : 0.9;
    ctx.translate(-485, y);
    ctx.scale(sc, sc);
    txt(ctx, label, 0, 0, 20, "#ffffff", "left", 800, true);
    txt(ctx, verdict, 430, 0, 21, kr > 0 ? col : "#56637a", "left", 900, true);
    ctx.restore();
  });
  ctx.restore();
}

const ROUNDS = [
  ["R1", "r1-t20-v45-g32", 83, 853.1, "124.7 MB under T2", "3 cells pass, control 0", false],
  ["R2", "pruning cliff", 0, 832.5, "820-845 MB", "0/2 passes both arms", true],
  ["R3", "r3-t15-v40-g32", 88, 885.1, "CI [80.2, 93.0]", "promoted", false],
];

function drawLive(ctx, t, s0, c, W) {
  const kp = outBack(appear(t, s0 + 0.25, 0.35));
  ctx.save();
  ctx.translate(54, 100);
  ctx.scale(kp, kp);
  card(ctx, 0, 0, 610, 415, "rgba(61,240,255,0.45)");
  pill(ctx, "LIVE SEARCH - RTX 3090 - M=4 + FROZEN CONTROL", 22, 34, "#3df0ff", "#06070d", 15);

  const x0 = 32, bw = 150, gap = 32, base = 330, maxH = 200;
  ROUNDS.forEach(([tag, name, sc, mb, note, sub, cliff], i) => {
    const rev = clamp((t - (s0 + 0.5 + i * 1.1)) / 0.4);
    const h = (cliff ? 0.06 : sc / 100) * maxH * outBack(rev);
    const x = x0 + i * (bw + gap);
    const col = cliff ? "#ff4d4d" : sc >= 88 ? "#5dff8f" : "#3df0ff";
    ctx.fillStyle = !rev ? "rgba(255,255,255,0.1)" : col;
    rrect(ctx, x, base - h, bw, Math.max(6, h), 8);
    ctx.fill();
    if (rev > 0.3) {
      const ly = cliff ? base - h - 18 : base - h + 30;
      txt(ctx, cliff ? "0/2" : `${sc}/100`, x + bw / 2, ly, 24, cliff ? "#ff4d4d" : "#ffffff", "center", 900, true);
      txt(ctx, tag, x + bw / 2, base + 22, 15, "#ffffff", "center", 900, true);
      txt(ctx, name, x + bw / 2, base + 42, 12, "#8fa3bf", "center", 700, true);
      txt(ctx, cliff ? note : `${mb} MB`, x + bw / 2, base + 60, 12, "#56637a", "center", 700, true);
    }
  });
  const gy = base - 0.8 * maxH;
  ctx.strokeStyle = "#ffe14d";
  ctx.setLineDash([8, 6]);
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(22, gy);
  ctx.lineTo(588, gy);
  ctx.stroke();
  ctx.setLineDash([]);
  pill(ctx, "GATE 8/10 + EXACT-BINOMIAL", 340, gy - 26, "#ffe14d", "#06070d", 14);
  ctx.restore();

  const kr3 = appear(t, c.r3 - 0.1, 0.4);
  if (kr3 > 0) {
    const kw = outBack(kr3);
    ctx.save();
    ctx.translate(690, 100);
    ctx.scale(kw, kw);
    card(ctx, 0, 0, 536, 175, "#5dff8f");
    pill(ctx, "PROMOTED - r3-t15-v40-g32", 22, 34, "#5dff8f", "#06070d", 16);
    pill(ctx, "TEXT -15% - VISION -40%", 22, 82, "#3df0ff", "#06070d", 15);
    pill(ctx, "GROUP SIZE 32", 300, 82, "#3df0ff", "#06070d", 15);
    txt(ctx, "88 / 100 dev - 885.1 MB", 22, 140, 28, "#ffffff", "left", 900, true);
    ctx.restore();
  }

  const kc = appear(t, c.cliff - 0.1, 0.35);
  if (kc > 0) {
    const k = outBack(kc);
    ctx.save();
    ctx.translate(690, 295);
    ctx.scale(k, k);
    card(ctx, 0, 0, 536, 220, "#ff4d4d");
    txt(ctx, "PRUNING CLIFF", 24, 50, 30, "#ff4d4d", "left", 900, true);
    txt(ctx, "820-845 MB - both arms fail", 24, 100, 22, "#ffffff", "left", 700);
    txt(ctx, "R1: 3 cells pass vs control 0", 24, 140, 20, "#ffe14d", "left", 700);
    txt(ctx, "83/100 - 853.1 MB - 124.7 vs T2", 24, 176, 20, "#8fa3bf", "left", 700);
    ctx.restore();
  }
}

function drawCode(ctx, t, s0, c, W) {
  const k0 = outBack(appear(t, s0 + 0.25, 0.4));
  ctx.save();
  ctx.translate(64, 110);
  ctx.scale(k0, k0);
  card(ctx, 0, 0, 560, 200, "#3df0ff");
  pill(ctx, "VLA-JEPA COMPRESSION CODEBASE", 24, 34, "#3df0ff", "#06070d", 15);
  txt(ctx, "released LIBERO policy - post-training compression", 24, 78, 20, "#ffffff", "left", 700);
  const mods = [
    ["evaluation/", "rollouts + timing", c.eval, "#3df0ff"],
    ["fastpath.py", "CUDA-graph engine", c.fast, "#7a8cff"],
    ["nf4.py", "4-bit weights", c.nf, "#ff3d9a"],
    ["compression/", "recipes to measures", c.comp, "#5dff8f"],
    ["rsi/", "tonight's search", c.rsimod, "#ffe14d"],
    ["studies/", "evidence", c.studies, "#8fa3bf"],
  ];
  mods.forEach(([name, desc, cue, col], i) => {
    const kr = outBack(appear(t, cue - 0.1, 0.32));
    const gx = 24 + (i % 2) * 270, gy = 108 + Math.floor(i / 2) * 32;
    ctx.save();
    ctx.globalAlpha = kr > 0 ? 1 : 0.3;
    txt(ctx, name, gx, gy, 17, kr > 0 ? col : "#56637a", "left", 900, true);
    txt(ctx, desc, gx + 128, gy, 14, "#8fa3bf", "left", 600);
    ctx.restore();
  });
  ctx.restore();

  const kv = appear(t, c.vla - 0.1, 0.4);
  if (kv > 0) {
    const s = outBack(kv);
    ctx.save();
    ctx.translate(660, 110);
    ctx.scale(s, s);
    card(ctx, 0, 0, 556, 200, "#ff3d9a");
    pill(ctx, "VLA-JEPA - QWEN3-VL + ACTION HEAD", 22, 34, "#ff3d9a", "#ffffff", 15);
    txt(ctx, "5.5 GB released policy", 22, 84, 30, "#ffffff", "left", 900, true);
    txt(ctx, "LIBERO-Spatial - 10 tasks", 22, 126, 20, "#8fa3bf", "left", 700);
    txt(ctx, "dev-100 guides, reserved-500 decides", 22, 160, 18, "#ffe14d", "left", 700);
    ctx.restore();
  }
}

function drawArms(ctx, t, s0, c, W) {
  const k0 = outBack(appear(t, s0 + 0.25, 0.4));
  ctx.save();
  ctx.translate(W / 2, 290);
  ctx.scale(k0, k0);
  card(ctx, -520, -180, 1040, 370, "#5dff8f");
  pill(ctx, "RESERVED-500 SCOREBOARD - NOBODY TUNES AGAINST THIS", -485, -137, "#5dff8f", "#06070d", 15);
  const arms = [
    ["B16 baseline", "475/500", "95.0%", "-", c.b16, "#8fa3bf"],
    ["C1 full-vocab NF4/HQQ2", "484/500", "96.8%", "1.38 GB", c.c1, "#3df0ff"],
    ["C2 CUDA-graph NF4", "490/500", "98.0%", "1.51 GB - 31.6 ms", c.c2, "#ffe14d"],
  ];
  arms.forEach(([label, frac, pct, extra, cue, col], i) => {
    const kr = outBack(appear(t, cue - 0.12, 0.35));
    const y = -70 + i * 96;
    const w = (i === 0 ? 0.95 : i === 1 ? 0.968 : 0.98) * 560 * outCubic(kr);
    ctx.save();
    ctx.globalAlpha = kr > 0 ? 1 : 0.25;
    txt(ctx, label, -485, y, 21, "#ffffff", "left", 800);
    rrect(ctx, -185, y - 22, Math.max(10, w), 44, 10);
    ctx.fillStyle = kr > 0 ? col : "#56637a";
    ctx.fill();
    if (kr > 0.4) {
      txt(ctx, `${frac}  ${pct}`, -165 + w, y, 24, "#ffffff", "left", 900, true);
      txt(ctx, extra, 330, y, 18, "#8fa3bf", "left", 700, true);
    }
    ctx.restore();
  });
  ctx.restore();
}

function drawFix(ctx, t, s0, c, W) {
  const k0 = outBack(appear(t, s0 + 0.25, 0.4));
  ctx.save();
  ctx.translate(64, 110);
  ctx.scale(k0, k0);
  card(ctx, 0, 0, 1152, 400, "#3df0ff");
  pill(ctx, "TONIGHT - RTX 3090 - ALL FOUR REPAIRS", 24, 36, "#3df0ff", "#06070d", 15);
  const items = [
    ["LEAVE-ONE-WORLD-OUT REPLAY", "33+ worlds", c.worlds, "#3df0ff"],
    ["10/10 COARSE FOLDS WON", "0.3854 vs 0.2192", c.coarse, "#5dff8f"],
    ["5/5 LIVE FOLDS WON", "0.2047 vs 0.0964", c.live5, "#5dff8f"],
    ["M=4 REVISION GENERATOR", "rsi/develop.py", c.m4gen, "#7a8cff"],
    ["EXACT-BINOMIAL GATE", "false kills 13.08 to 4.43%", c.fk, "#ffe14d"],
    ["PARALLEL ROLLOUT HARNESS", "evaluation/parallel_rollout", c.par, "#ff3d9a"],
  ];
  items.forEach(([label, val, cue, col], i) => {
    const kr = outBack(appear(t, cue - 0.1, 0.32));
    const gx = 24 + (i % 2) * 576, gy = 92 + Math.floor(i / 2) * 100;
    ctx.save();
    ctx.globalAlpha = kr > 0 ? 1 : 0.22;
    const sc = kr > 0 ? kr : 0.92;
    ctx.translate(gx, gy);
    ctx.scale(sc, sc);
    card(ctx, 0, 0, 548, 84, kr > 0 ? col : "rgba(255,255,255,0.18)", 12, "rgba(10,14,28,0.9)");
    txt(ctx, label, 18, 30, 19, kr > 0 ? col : "#56637a", "left", 900, true);
    txt(ctx, val, 18, 60, 17, "#ffffff", "left", 700, true);
    ctx.restore();
  });
  ctx.restore();
}

function drawJump(ctx, t, s0, c, W) {
  const k0 = outBack(appear(t, s0 + 0.25, 0.4));
  ctx.save();
  ctx.translate(64, 100);
  ctx.scale(k0, k0);
  card(ctx, 0, 0, 560, 420, "#3df0ff");
  pill(ctx, "ROUND 4 - FULL-VOCAB C1 BASE", 22, 34, "#3df0ff", "#06070d", 15);
  txt(ctx, "int8_embedding replaces compact_embedding", 22, 78, 18, "#8fa3bf", "left", 700);
  const kf = outBack(appear(t, c.five5 - 0.1, 0.35));
  txt(ctx, "smoke: 5/5 probes pass 8/10", 22, 118, 22, kf > 0 ? "#5dff8f" : "#56637a", "left", 800);
  const kr4 = outBack(appear(t, c.r4 - 0.1, 0.35));
  if (kr4 > 0) {
    ctx.save();
    ctx.translate(22, 150);
    ctx.scale(kr4, kr4);
    card(ctx, 0, 0, 516, 130, "#5dff8f");
    txt(ctx, "r4-c1-t15-v35-g32", 18, 34, 24, "#5dff8f", "left", 900, true);
    txt(ctx, "86/100 dev [77.9, 91.5] - 1199.6 MB", 18, 74, 20, "#ffffff", "left", 800);
    txt(ctx, "88.8 MB under C1 (1288.4 MB, 87/100)", 18, 104, 17, "#8fa3bf", "left", 700);
    ctx.restore();
  }
  const ks1 = outBack(appear(t, c.s1 - 0.1, 0.35));
  if (ks1 > 0) {
    ctx.save();
    ctx.translate(22, 295);
    ctx.scale(ks1, ks1);
    pill(ctx, "solver_steps=1 PASSES 8/10", 0, 0, "#ffe14d", "#06070d", 17);
    ctx.restore();
  }
  ctx.restore();

  const kh = appear(t, c.ctrl - 0.12, 0.4);
  if (kh > 0) {
    const k = outBack(kh);
    const kf6 = appear(t, c.fail6 - 0.1, 0.35);
    ctx.save();
    ctx.translate(656, 100);
    ctx.scale(k, k);
    card(ctx, 0, 0, 560, 420, "#ff3d9a");
    pill(ctx, "ROUND 5 - UNTOUCHED AXES", 22, 34, "#ff3d9a", "#ffffff", 15);
    txt(ctx, "HQQ 3-bit vision: 85/100 dev", 22, 92, 24, "#ffffff", "left", 900, true);
    txt(ctx, "876.0 MB", 22, 130, 24, "#5dff8f", "left", 800);
    txt(ctx, "tasks 0, 5, 8: 5, 7, 4 /10", 22, 172, 20, "#ffe14d", "left", 800);
    txt(ctx, "control-bottlenecked, not bitwidth", 22, 206, 18, "#ffffff", "left", 700);
    if (kf6 > 0) {
      txt(ctx, "12.5% head pruning: 6/10 FAILS", 22, 252, 20, "#ff4d4d", "left", 900, true);
      txt(ctx, "attention heads are load-bearing", 22, 284, 18, "#8fa3bf", "left", 700);
    }
    ctx.restore();
  }
}

function drawTasks(ctx, t, s0, c, W) {
  const k0 = outBack(appear(t, s0 + 0.25, 0.4));
  ctx.save();
  ctx.translate(64, 110);
  ctx.scale(k0, k0);
  card(ctx, 0, 0, 1152, 400, "#ffe14d");
  pill(ctx, "PER-TASK DIAGNOSIS - WHY DEV RUNS STALL AT 83-88", 24, 36, "#ffe14d", "#06070d", 15);
  const scores = [55, 95, 95, 95, 92, 60, 95, 97, 45, 93];
  const bad = { 0: c.t0, 5: c.t5, 8: c.t8 };
  const x0 = 40, bw = 92, gap = 16, base = 420, maxH = 220;
  scores.forEach((sc, i) => {
    const cue = bad[i] !== undefined ? bad[i] : s0 + 1.2;
    const rev = clamp((t - (cue - 0.15)) / 0.35);
    const h = (sc / 100) * maxH * outBack(Math.max(0, rev));
    const isBad = sc < 80;
    const x = x0 + i * (bw + gap);
    ctx.fillStyle = !rev ? "rgba(255,255,255,0.12)" : isBad ? "#ff4d4d" : "#5dff8f";
    rrect(ctx, x, base - h, bw, Math.max(6, h), 8);
    ctx.fill();
    if (rev > 0.3) {
      txt(ctx, isBad ? "40-70" : "90+", x + bw / 2, base - h - 16, isBad ? 18 : 20, isBad ? "#ff4d4d" : "#ffffff", "center", 900, true);
      txt(ctx, `T${i}`, x + bw / 2, base + 24, 15, isBad ? "#ff4d4d" : "#8fa3bf", "center", 900, true);
    }
  });
  const gy = base - 0.9 * maxH;
  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(24, gy);
  ctx.lineTo(1128, gy);
  ctx.stroke();
  ctx.setLineDash([]);
  txt(ctx, "the gripper, not the quantization, is the ceiling", 576, 462, 20, "#ffffff", "center", 700);
  ctx.restore();
}

function drawClose(ctx, t, s0, c, total, W, H) {
  const k1 = outBack(appear(t, s0 + 0.25, 0.4));
  const k2 = outBack(appear(t, s0 + 1.6, 0.4));
  const k3 = outBack(appear(t, s0 + 2.8, 0.4));
  ctx.save();
  ctx.translate(150, 175);
  ctx.scale(k1, k1);
  card(ctx, -130, -55, 260, 110, "#7a8cff");
  txt(ctx, "ROUND 6 STAGED", 0, -14, 22, "#7a8cff", "center", 900, true);
  txt(ctx, "C2 + full-precision grid", 0, 22, 16, "#ffffff", "center", 600);
  ctx.restore();

  ctx.save();
  ctx.translate(640, 175);
  ctx.scale(k2, k2);
  card(ctx, -170, -55, 340, 110, "#ffe14d");
  txt(ctx, "500 RESERVED: 100% UNTOUCHED", 0, -14, 18, "#ffe14d", "center", 900, true);
  txt(ctx, "experiments freeze here", 0, 22, 17, "#ffffff", "center", 600);
  ctx.restore();

  ctx.save();
  ctx.translate(1070, 175);
  ctx.scale(k3, k3);
  card(ctx, -130, -55, 260, 110, "#3df0ff");
  txt(ctx, "FINALIZE", 0, -14, 22, "#3df0ff", "center", 900, true);
  txt(ctx, "5-page report + talk", 0, 22, 16, "#ffffff", "center", 600);
  ctx.restore();

  const kr = appear(t, c.robot - 0.15, 0.4);
  if (kr > 0) {
    const s = outBack(kr, 2.3);
    ctx.save();
    ctx.translate(W / 2, 370);
    ctx.scale(s, s);
    card(ctx, -470, -95, 940, 190, "#5dff8f", 24, "rgba(8,14,26,0.92)");
    txt(ctx, "DREAM-RSI MAKES SEARCH CHEAP.", 0, -34, 32, "#3df0ff", "center", 800);
    txt(ctx, "ONLY THE ROBOT TELLS THE TRUTH.", 0, 26, 46, "#ffe14d", "center", 900);
    ctx.restore();
  }
}

function smoothMove(t, a, b) {
  const x = clamp((t - a) / (b - a));
  return x * x * (3 - 2 * x);
}
