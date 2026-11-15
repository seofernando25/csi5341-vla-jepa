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
  pill(ctx, `${String(idx + 1).padStart(2, "0")} · ${sc.chip}`, 0, 0, "#3df0ff", "#06070d", 16);
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
  else if (id === "protocol") drawProtocol(ctx, t, s0, cues, W);
  else if (id === "project") drawProject(ctx, t, s0, cues, W);
  else if (id === "fail") drawFail(ctx, t, s0, cues, W, H);
  else if (id === "audit") drawAudit(ctx, t, s0, cues, W);
  else if (id === "harness") drawHarness(ctx, t, s0, cues, W);
  else if (id === "live") drawLive(ctx, t, s0, cues, W);
  else if (id === "axes") drawAxes(ctx, t, s0, cues, W);
  else if (id === "loss") drawLoss(ctx, t, s0, cues, W);
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
  const v = lerp(0.31, 0.27, clamp((t - c.better) / 1.5));
  txt(ctx, v.toFixed(2), 28, 165, 86, kb > 0 ? "#5dff8f" : "#ffffff", "left", 900, true);
  txt(ctx, "from 0.31", 260, 180, 22, "#8fa3bf", "left", 700, true);
  if (kb > 0) {
    ctx.save();
    ctx.translate(28, 240);
    ctx.scale(kb, kb);
    pill(ctx, "✓ LOSS DOWN · BEST CANDIDATE", 0, 0, "#5dff8f", "#06070d", 17);
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
  txt(ctx, "recorded attempts → offline search tree", -282, 80, 22, "#ffffff", "left", 700);
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

function drawProject(ctx, t, s0, c, W) {
  const k0 = outBack(appear(t, s0 + 0.3, 0.4));
  const smol = appear(t, c.smol - 0.1, 0.45);
  ctx.save();
  ctx.translate(64, 115);
  ctx.scale(k0, k0);
  card(ctx, 0, 0, 560, 185, smol > 0 ? "#ff3d9a" : "#3df0ff");
  pill(ctx, smol > 0 ? "ARM 1 · BACKBONE SWAP" : "RELEASED BASELINE · B16", 24, 36, smol > 0 ? "#ff3d9a" : "#3df0ff", "#06070d", 15);
  txt(ctx, smol > 0 ? "Qwen3-VL  →  SmolVLM2-500M" : "VLA-JEPA (Qwen3-VL + Action Head)", 24, 82, 26, "#ffffff", "left", 800);
  ctx.fillStyle = "rgba(255,255,255,0.14)";
  rrect(ctx, 24, 118, 512, 42, 10);
  ctx.fill();
  ctx.fillStyle = smol > 0 ? "#ff3d9a" : "#3df0ff";
  rrect(ctx, 24, 118, 512 * lerp(1, 3041 / 5532, outCubic(smol)), 42, 10);
  ctx.fill();
  txt(ctx, smol > 0 ? "500M · 2.97 GiB peak" : "5.5 GB resident", 40, 139, 19, "#ffffff", "left", 900, true);
  ctx.restore();

  const kl = outBack(appear(t, c.llm - 0.15, 0.4));
  if (kl > 0) {
    ctx.save();
    ctx.translate(64, 325);
    ctx.scale(kl, kl);
    card(ctx, 0, 0, 560, 190, "rgba(255,225,77,0.55)");
    txt(ctx, "LLM-PROPOSED RESIDUAL ADAPTERS", 24, 32, 16, "#ffe14d", "left", 800, true);
    const rev = clamp((t - c.llm) / 1.6);
    for (let i = 0; i < 3; i++) {
      const k = clamp(rev * 3 - i);
      if (k <= 0) continue;
      ctx.fillStyle = "rgba(255,255,255,0.14)";
      rrect(ctx, 38, 62 + i * 34, 300, 22, 8);
      ctx.fill();
      ctx.fillStyle = i === 2 ? "#ffe14d" : "#3df0ff";
      rrect(ctx, 38, 62 + i * 34, 300 * outCubic(k), 22, 8);
      ctx.fill();
    }
    const kloss = appear(t, c.loss - 0.15, 0.45);
    if (kloss > 0) {
      const v = lerp(0.31, 0.27, outCubic(kloss));
      txt(ctx, "BEST HELD-OUT LOSS", 365, 76, 15, "#8fa3bf", "left", 700, true);
      txt(ctx, v.toFixed(2), 365, 130, 58, "#5dff8f", "left", 900, true);
      txt(ctx, "from 0.31", 495, 138, 17, "#8fa3bf", "left", 700, true);
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
  card(ctx, -520, -190, 1040, 420, "#3df0ff");
  pill(ctx, "FOUR WAYS A SEARCH MISLEADS · THE RULE THAT BLOCKS EACH", -485, -147, "#3df0ff", "#06070d", 15);
  txt(ctx, "Audited against arXiv:2609.14858v2 before the clean run", -485, -104, 20, "#8fa3bf", "left", 600);
  const rows = [
    ["REPLAY COVERAGE", "OUT-OF-SAMPLE ONLY", c.replay2],
    ["BASE ARM", "FROZEN IN THE RECIPE", c.base2],
    ["PROMOTION GATE", "WRITTEN BEFORE THE PHASE", c.gatefix],
    ["M=4 REVISION", "IN THE PLAN", c.rev],
    ["FROZEN CONTROL", "PAIRED ON THE SAME STATES", c.control],
  ];
  rows.forEach(([label, verdict, cue], i) => {
    const kr = outBack(appear(t, cue - 0.1, 0.32));
    const y = -50 + i * 56;
    ctx.save();
    ctx.globalAlpha = kr > 0.001 ? 1 : 0.55;
    const sc = kr > 0.001 ? kr : 0.9;
    ctx.translate(-485, y);
    ctx.scale(sc, sc);
    txt(ctx, label, 0, 0, 20, "#ffffff", "left", 800, true);
    txt(ctx, verdict, 975, 0, 18, kr > 0.001 ? "#5dff8f" : "#6f7d95", "right", 900, true);
    ctx.restore();
  });
  const kf = appear(t, c.fixed - 0.1, 0.35);
  if (kf > 0) {
    txt(ctx, "written before the first run", 0, 208, 19, `rgba(255,225,77,${kf})`, "center", 800);
  }
  ctx.restore();
}

function drawLive(ctx, t, s0, c, W) {
  const kp = outBack(appear(t, s0 + 0.25, 0.35));
  ctx.save();
  ctx.translate(54, 110);
  ctx.scale(kp, kp);
  card(ctx, 0, 0, 1170, 300, "rgba(61,240,255,0.45)");
  pill(ctx, "CLEAN LOOP · NOT LAUNCHED · METHOD ONLY", 22, 34, "#3df0ff", "#06070d", 15);

  const steps = [
    ["01 · ONLINE", "run the current policy", c.online, "#3df0ff"],
    ["02 · GATE", "smoke, then development", c.promote, "#ffe14d"],
    ["03 · PROMOTE", "keep only what clears", c.promote, "#5dff8f"],
    ["04 · RECORD", "every attempt stays", c.record, "#7a8cff"],
  ];
  const bw = 250, gap = 40, x0 = 25, y0 = 96;
  steps.forEach(([label, sub, cue, col], i) => {
    const kr = outBack(appear(t, cue - 0.12, 0.34));
    const x = x0 + i * (bw + gap);
    ctx.save();
    ctx.globalAlpha = kr > 0.001 ? 1 : 0.55;
    const sc = kr > 0.001 ? kr : 0.92;
    ctx.translate(x, y0);
    ctx.scale(sc, sc);
    card(ctx, 0, 0, bw, 150, kr > 0.001 ? col : "rgba(255,255,255,0.18)", 14, "rgba(10,14,28,0.9)");
    txt(ctx, label, 18, 40, 20, kr > 0.001 ? col : "#6f7d95", "left", 900, true);
    txt(ctx, sub, 18, 92, 18, "#ffffff", "left", 600);
    ctx.restore();
    if (i < steps.length - 1) {
      const ka = appear(t, cue + 0.25, 0.3);
      if (ka > 0) txt(ctx, "→", x + bw + gap / 2, y0 + 75, 30, `rgba(255,255,255,${0.7 * ka})`, "center", 900, true);
    }
  });

  const kf = outBack(appear(t, c.lfreeze - 0.1, 0.35));
  if (kf > 0) {
    ctx.save();
    ctx.translate(25, 270);
    ctx.scale(kf, kf);
    pill(ctx, "SELECTION FREEZES BEFORE THE RESERVED READ", 0, 0, "#ffe14d", "#06070d", 17);
    ctx.restore();
  }
  ctx.restore();
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
    ctx.globalAlpha = kr > 0.001 ? 1 : 0.3;
    txt(ctx, name, gx, gy, 17, kr > 0.001 ? col : "#6f7d95", "left", 900, true);
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

function drawProtocol(ctx, t, s0, c, W) {
  const k0 = outBack(appear(t, s0 + 0.25, 0.4));
  ctx.save();
  ctx.translate(W / 2, 300);
  ctx.scale(k0, k0);
  card(ctx, -520, -190, 1040, 390, "#5dff8f");
  pill(ctx, "FOUR PHASES · NO PHASE IS POOLED", -485, -147, "#5dff8f", "#06070d", 15);
  const rows = [
    ["SCREEN", "80 fixed frames, no simulator", c.screen, "#8fa3bf"],
    ["SMOKE", "10 trials · gate 8/10", c.smoke, "#3df0ff"],
    ["DEVELOPMENT", "100 trials · guides selection", c.dev, "#ffe14d"],
    ["RESERVED", "500 trials · final claim only", c.reserved, "#5dff8f"],
  ];
  rows.forEach(([label, note, cue, col], i) => {
    const kr = outBack(appear(t, cue - 0.12, 0.35));
    const gate = i === 1 && t >= c.gate8 - 0.1;
    const y = -85 + i * 72;
    ctx.save();
    ctx.globalAlpha = kr > 0.001 ? 1 : 0.55;
    const sc = kr > 0.001 ? kr : 0.92;
    ctx.translate(-485, y);
    ctx.scale(sc, sc);
    txt(ctx, label, 0, 0, 24, kr > 0.001 ? col : "#6f7d95", "left", 900, true);
    txt(ctx, note, 250, 0, 21, gate ? "#5dff8f" : kr > 0.001 ? "#ffffff" : "#6f7d95", "left", gate ? 900 : 700);
    ctx.restore();
  });
  const kn = appear(t, c.nopool - 0.1, 0.35);
  if (kn > 0) {
    ctx.save();
    ctx.globalAlpha = kn;
    txt(ctx, "reserved never guides selection", 0, 172, 20, "#ffe14d", "center", 800);
    ctx.restore();
  }
  ctx.restore();
}

function drawHarness(ctx, t, s0, c, W) {
  const k0 = outBack(appear(t, s0 + 0.25, 0.4));
  ctx.save();
  ctx.translate(64, 110);
  ctx.scale(k0, k0);
  card(ctx, 0, 0, 1152, 400, "#3df0ff");
  pill(ctx, "CLEAN HARNESS · ONE NODE AT A TIME", 24, 36, "#3df0ff", "#06070d", 15);
  const items = [
    ["EXPANSION", "rsi/expansion.py · fork, agent, freeze", c.exp, "#3df0ff"],
    ["REPLAY", "rsi/replay_worker.py · offline, no GPU", c.rp, "#5dff8f"],
    ["POLICY DEV", "rsi/policy_dev.py · replay-gated", c.pdev, "#7a8cff"],
    ["GATES", "rsi/gate.py · 8/10, 77/100, Wilson", c.gates, "#ffe14d"],
    ["PARALLEL ROLLOUT", "evaluation/parallel_rollout.py", c.par, "#ff3d9a"],
    ["CYCLE RECORD", "tree, manifest, policy hash", c.rec, "#8fa3bf"],
  ];
  items.forEach(([label, val, cue, col], i) => {
    const kr = outBack(appear(t, cue - 0.1, 0.32));
    const gx = 24 + (i % 2) * 576, gy = 92 + Math.floor(i / 2) * 100;
    ctx.save();
    ctx.globalAlpha = kr > 0.001 ? 1 : 0.55;
    const sc = kr > 0.001 ? kr : 0.92;
    ctx.translate(gx, gy);
    ctx.scale(sc, sc);
    card(ctx, 0, 0, 548, 84, kr > 0.001 ? col : "rgba(255,255,255,0.18)", 12, "rgba(10,14,28,0.9)");
    txt(ctx, label, 18, 30, 19, kr > 0.001 ? col : "#6f7d95", "left", 900, true);
    txt(ctx, val, 18, 60, 16, "#ffffff", "left", 700, true);
    ctx.restore();
  });
  ctx.restore();
}

function drawAxes(ctx, t, s0, c, W) {
  const k0 = outBack(appear(t, s0 + 0.25, 0.4));
  ctx.save();
  ctx.translate(64, 100);
  ctx.scale(k0, k0);
  card(ctx, 0, 0, 1152, 420, "#3df0ff");
  pill(ctx, "PRECISION SURFACE · PRE-REGISTERED ARMS", 22, 34, "#3df0ff", "#06070d", 15);
  const cols = [
    ["LANGUAGE", c.lang],
    ["VISION", c.vision],
    ["ACTION", c.action],
  ];
  const fmts = ["INT8", "NF4", "HQQ3", "HQQ2"];
  cols.forEach(([label, cue], i) => {
    const kr = outBack(appear(t, cue - 0.1, 0.35));
    const x = 22 + i * 376;
    ctx.save();
    ctx.globalAlpha = kr > 0.001 ? 1 : 0.55;
    card(ctx, x, 66, 352, 186, kr > 0.001 ? "#3df0ff" : "rgba(255,255,255,0.18)", 12, "rgba(10,14,28,0.9)");
    txt(ctx, label, x + 18, 96, 20, kr > 0.001 ? "#3df0ff" : "#6f7d95", "left", 900, true);
    fmts.forEach((f, j) => {
      const kk = outBack(appear(t, cue - 0.1 + j * 0.12, 0.3));
      if (kk <= 0) return;
      ctx.save();
      ctx.translate(x + 18 + (j % 2) * 168, 146 + Math.floor(j / 2) * 54);
      ctx.scale(kk, kk);
      pill(ctx, f, 0, 0, kr > 0.001 ? "#7a8cff" : "#6f7d95", "#06070d", 16);
      ctx.restore();
    });
    ctx.restore();
  });

  const kj = outBack(appear(t, c.joint - 0.1, 0.35));
  if (kj > 0) {
    ctx.save();
    ctx.translate(22, 284);
    ctx.scale(kj, kj);
    pill(ctx, "JOINT VISION + ACTION 2-BIT CELL", 0, 0, "#ff3d9a", "#ffffff", 17);
    ctx.restore();
  }
  const kc = appear(t, c.curve - 0.1, 0.35);
  if (kc > 0) txt(ctx, "success against effective bits, not nominal bits", 22, 336, 20, `rgba(255,255,255,${kc})`, "left", 700);
  const ke = appear(t, c.exec - 0.1, 0.35);
  if (ke > 0) {
    txt(ctx, "EXECUTION LADDER", 22, 382, 17, `rgba(255,225,77,${ke})`, "left", 800, true);
    txt(ctx, "stock / CUDA graph · dense / packed · 1 / 4 flow steps", 250, 382, 18, `rgba(255,255,255,${ke})`, "left", 700);
  }
  ctx.restore();
}

function drawLoss(ctx, t, s0, c, W) {
  const k0 = outBack(appear(t, s0 + 0.25, 0.4));
  ctx.save();
  ctx.translate(64, 110);
  ctx.scale(k0, k0);
  card(ctx, 0, 0, 1152, 400, "#ffe14d");
  pill(ctx, "THREE SUSPECTS · SEPARATED BY THE PROTOCOL", 24, 36, "#ffe14d", "#06070d", 15);
  const items = [
    ["WEIGHT PRECISION", "does the drop track effective bits?", "measured by the surface", c.w, "#3df0ff"],
    ["EXECUTION PATH", "does it track the runtime path?", "measured by the ladder", c.e, "#7a8cff"],
    ["SEARCH PROCEDURE", "does it track the selected policy?", "measured by the cycle", c.s, "#ff3d9a"],
  ];
  items.forEach(([label, sub, note, cue, col], i) => {
    const kr = outBack(appear(t, cue - 0.1, 0.35));
    const x = 24 + i * 376;
    ctx.save();
    ctx.globalAlpha = kr > 0.001 ? 1 : 0.55;
    const sc = kr > 0.001 ? kr : 0.92;
    ctx.translate(x, 92);
    ctx.scale(sc, sc);
    card(ctx, 0, 0, 352, 180, kr > 0.001 ? col : "rgba(255,255,255,0.18)", 14, "rgba(10,14,28,0.9)");
    txt(ctx, label, 18, 48, 21, kr > 0.001 ? col : "#6f7d95", "left", 900, true);
    txt(ctx, sub, 18, 100, 17, "#ffffff", "left", 600);
    txt(ctx, note, 18, 146, 15, "#8fa3bf", "left", 700, true);
    ctx.restore();
  });
  const kq = appear(t, c.q - 0.1, 0.35);
  if (kq > 0) {
    txt(ctx, "per-task results name the ceiling: the gripper or the quantization", 24, 320, 21, `rgba(255,255,255,${kq})`, "left", 700);
  }
  const kr2 = outBack(appear(t, c.robot2 - 0.1, 0.35));
  if (kr2 > 0) {
    ctx.save();
    ctx.translate(24, 366);
    ctx.scale(kr2, kr2);
    pill(ctx, "ONLY THE ROBOT DECIDES", 0, 0, "#5dff8f", "#06070d", 17);
    ctx.restore();
  }
  ctx.restore();
}

function drawClose(ctx, t, s0, c, total, W, H) {
  const k1 = outBack(appear(t, c.locked - 0.1, 0.4));
  const k2 = outBack(appear(t, c.untouched - 0.1, 0.4));
  const k3 = outBack(appear(t, c.freeze + 0.4, 0.4));
  ctx.save();
  ctx.translate(150, 175);
  ctx.scale(k1, k1);
  card(ctx, -130, -55, 260, 110, "#7a8cff");
  txt(ctx, "SELECTION FROZEN", 0, -14, 22, "#7a8cff", "center", 900, true);
  txt(ctx, "before the reserved read", 0, 22, 16, "#ffffff", "center", 600);
  ctx.restore();

  ctx.save();
  ctx.translate(640, 175);
  ctx.scale(k2, k2);
  card(ctx, -170, -55, 340, 110, "#ffe14d");
  txt(ctx, "500 RESERVED · LOCKED", 0, -14, 18, "#ffe14d", "center", 900, true);
  txt(ctx, "experiments stop here", 0, 22, 17, "#ffffff", "center", 600);
  ctx.restore();

  ctx.save();
  ctx.translate(1070, 175);
  ctx.scale(k3, k3);
  card(ctx, -130, -55, 260, 110, "#3df0ff");
  txt(ctx, "FINALIZE", 0, -14, 22, "#3df0ff", "center", 900, true);
  txt(ctx, "5-page report + talk", 0, 22, 16, "#ffffff", "center", 600);
  ctx.restore();

  const kr = appear(t, c.robot - 0.15, 0.4);
  if (kr > 0.001) {
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
