// Scene painters. Minimal text; the narrator carries the content. Figures, diagrams, glyphs only.
import {
  PAPER, INK, INK_SOFT, MUTED, FAINT, HAIRLINE, BROWN, BROWN_DEEP, TERRA,
  CARD, mono, card, chip, checkMark, crossMark, hairline, appear, outBack, smooth, lerp,
} from "./lib.js";
import { clamp, outCubic } from "./ease.js";

const ENTER = 0.42;

function env(t, s0) {
  const a = outCubic(appear(t, s0 + 0.04, ENTER));
  return { a, dy: (1 - a) * 16 };
}

function bar(ctx, x, y, w, h, fill, r = 4) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.fillStyle = fill;
  ctx.fill();
}

function sunkenBar(ctx, x, y, wMax, h) {
  ctx.beginPath();
  ctx.roundRect(x, y, wMax, h, 4);
  ctx.strokeStyle = HAIRLINE;
  ctx.lineWidth = 1;
  ctx.stroke();
}

// ---------------------------------------------------------------- hook
function drawHook(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  const pulse = 0.62 + 0.38 * Math.sin(t * 1.5);
  chip(ctx, 300, 392, 250, { lit: pulse });

  const k = appear(t, s0 + 0.15, 0.9);
  const size = 210;
  ctx.font = `800 ${size}px Inter, sans-serif`;
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  ctx.fillStyle = INK;
  const shown = (5.5 * outCubic(k)).toFixed(1);
  ctx.fillText(shown, 560, 400);
  const w5 = ctx.measureText("5.5").width;
  ctx.font = `500 54px Inter, sans-serif`;
  ctx.fillStyle = MUTED;
  ctx.fillText("GB", 560 + w5 + 22, 400);
  mono(ctx, "R E S I D E N T   ·   R E L E A S E D", 560, 452, { size: 14, color: MUTED });
  ctx.restore();
}

// ---------------------------------------------------------------- why
function drawWhy(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  // workstation: monitor with chip
  ctx.strokeStyle = INK_SOFT;
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.roundRect(140, 285, 220, 150, 12);
  ctx.stroke();
  chip(ctx, 250, 355, 64, { lit: 1 });
  // stand
  ctx.beginPath();
  ctx.moveTo(250, 435);
  ctx.lineTo(250, 462);
  ctx.moveTo(190, 468);
  ctx.lineTo(310, 468);
  ctx.stroke();
  mono(ctx, "W O R K S T A T I O N", 250, 520, { size: 12, color: MUTED, align: "center" });
  // arrow
  const ax0 = 420, ax1 = 610, ay = 360;
  const ak = appear(t, s0 + 0.3, 0.55);
  ctx.strokeStyle = BROWN;
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(ax0, ay);
  ctx.lineTo(lerp(ax0, ax1, ak), ay);
  ctx.stroke();
  if (ak > 0.9) {
    ctx.beginPath();
    ctx.moveTo(ax1, ay);
    ctx.lineTo(ax1 - 26, ay - 14);
    ctx.lineTo(ax1 - 26, ay + 14);
    ctx.closePath();
    ctx.fillStyle = BROWN;
    ctx.fill();
  }
  ctx.font = `800 52px Inter, sans-serif`;
  ctx.fillStyle = BROWN;
  ctx.textAlign = "center";
  ctx.fillText("6×", (ax0 + ax1) / 2, ay - 34);
  // robot arm (line art): base, pedestal, forearm, wrist, gripper
  ctx.strokeStyle = INK;
  ctx.lineWidth = 7;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.ellipse(880, 500, 96, 17, 0, Math.PI, 0);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(880, 492);
  ctx.lineTo(880, 356);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(880, 356);
  ctx.lineTo(1012, 262);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(1012, 262);
  ctx.lineTo(1012, 380);
  ctx.stroke();
  // wrist + two gripper fingers
  ctx.beginPath();
  ctx.moveTo(1012, 380);
  ctx.lineTo(1012, 424);
  ctx.moveTo(1012, 424);
  ctx.lineTo(984, 452);
  ctx.moveTo(1012, 424);
  ctx.lineTo(1040, 452);
  ctx.lineWidth = 5;
  ctx.stroke();
  // joints
  ctx.fillStyle = BROWN;
  for (const [jx, jy, jr] of [[880, 356, 15], [1012, 262, 13], [1012, 380, 11]]) {
    ctx.beginPath();
    ctx.arc(jx, jy, jr, 0, Math.PI * 2);
    ctx.fill();
  }
  mono(ctx, "A R M   ·   P L A T E", 880, 530, { size: 12, color: MUTED, align: "center" });
  ctx.restore();
}

// ---------------------------------------------------------------- policy
function drawPolicy(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  // camera frame
  card(ctx, 96, 152, 560, 348, 16);
  ctx.fillStyle = BROWN;
  ctx.beginPath();
  ctx.arc(130, 186, 7, 0, Math.PI * 2);
  ctx.fill();
  mono(ctx, "L I B E R O", 148, 186, { size: 13, weight: 700, color: MUTED });
  // table line + bowl (on counter, left) + plate (right)
  hairline(ctx, 150, 430, 610, INK_SOFT, 2.5);
  ctx.strokeStyle = INK_SOFT;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.ellipse(235, 430, 46, 7.5, 0, Math.PI, 0);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(455, 430, 52, 9, 0, Math.PI, 0);
  ctx.stroke();
  const drop = outBack(clamp(appear(t, s0 + 0.1, 0.6)), 1.8);
  const yBowl = 424 - 44 * (1 - drop);
  ctx.beginPath();
  ctx.arc(235, yBowl, 24, Math.PI, 0);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(235, yBowl, 24, 7, 0, 0, Math.PI * 2);
  ctx.stroke();
  mono(ctx, "p i c k   u p   t h e   b l a c k   b o w l", 376, 474, { size: 12, color: MUTED, align: "center" });
  // right column: VISION / LANGUAGE / ACTION
  const items = ["VISION", "LANGUAGE", "ACTION"];
  items.forEach((s, i) => {
    const y = 232 + i * 92;
    const k = appear(t, s0 + 0.25 + i * 0.18, 0.4);
    ctx.globalAlpha = a * outCubic(clamp(k));
    hairline(ctx, 752, y + 24, 790, BROWN, 3);
    ctx.font = `800 40px Inter, sans-serif`;
    ctx.fillStyle = INK;
    ctx.fillText(s, 806, y + 8);
    ctx.globalAlpha = a;
  });
  mono(ctx, "V L A - J E P A", 752, 520, { size: 16, weight: 700, color: BROWN });
  ctx.restore();
}

// ---------------------------------------------------------------- success
function drawSuccess(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  const pop = appear(t, s0 + 0.4, 0.5);
  const cw = 390, ch = 300, gap = 90, x0 = (W - (2 * cw + gap)) / 2, y = 185;
  for (let i = 0; i < 2; i++) {
    const x = x0 + i * (cw + gap);
    card(ctx, x, y, cw, ch, 16);
    const cx = x + cw / 2;
    // verdict glyph, centred in the card
    if (i === 0) {
      const k = outBack(clamp(pop), 2.0);
      ctx.save();
      ctx.translate(cx, y + 118);
      ctx.globalAlpha = a * clamp(pop);
      ctx.scale(k, k);
      checkMark(ctx, 0, 0, 44, BROWN);
      ctx.restore();
      ctx.globalAlpha = a;
    } else {
      ctx.globalAlpha = a * 0.85;
      crossMark(ctx, cx, y + 118, 38, TERRA);
      ctx.globalAlpha = a;
    }
    // bowl vs plates, bottom area
    hairline(ctx, x + 40, y + 205, x + cw - 40, INK_SOFT, 2);
    ctx.strokeStyle = INK_SOFT;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(x + 100, y + 205, 40, 7, 0, Math.PI, 0);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(x + cw - 100, y + 205, 46, 8, 0, Math.PI, 0);
    ctx.stroke();
    const bx = i === 0 ? x + cw - 100 : x + 100;
    ctx.beginPath();
    ctx.arc(bx, y + 178, 19, Math.PI, 0);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(bx, y + 178, 19, 5.5, 0, 0, Math.PI * 2);
    ctx.stroke();
    mono(ctx, i === 0 ? "O N   T H E   P L A T E" : "B E S I D E   I T", cx, y + 252, { size: 13, weight: 700, color: i === 0 ? BROWN : MUTED, align: "center" });
  }
  ctx.restore();
}

// ---------------------------------------------------------------- episode
function drawEpisode(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  const y = 380;
  hairline(ctx, 130, y, 1160, HAIRLINE, 2);
  const stations = [
    { x: 200, label: "R E S E T", sub: "F I X E D   S T A T E" },
    { x: 500, label: "O B S E R V E", sub: "1   F R A M E" },
    { x: 800, label: "A C T", sub: "S T E P   B Y   S T E P" },
    { x: 1100, label: "D O N E", sub: "O R   T I M E O U T" },
  ];
  stations.forEach((st, i) => {
    const k = appear(t, s0 + 0.2 + i * 0.16, 0.35);
    const e = outBack(clamp(k), 2.2);
    ctx.fillStyle = i === 0 ? CARD : BROWN;
    ctx.beginPath();
    ctx.arc(st.x, y, 15 * e, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = BROWN;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.globalAlpha = a * clamp(k);
    mono(ctx, st.label, st.x, y + 46, { size: 15, weight: 700, color: INK, align: "center" });
    mono(ctx, st.sub, st.x, y + 74, { size: 10, color: MUTED, align: "center" });
    ctx.globalAlpha = a;
  });
  // playhead
  const loopT = (t - s0) % 4.6;
  const px = lerp(130, 1160, clamp(loopT / 4.2));
  ctx.fillStyle = BROWN;
  ctx.beginPath();
  ctx.arc(px, y, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = a * 0.35;
  ctx.beginPath();
  ctx.arc(px, y, 16, 0, Math.PI * 2);
  ctx.strokeStyle = BROWN;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.globalAlpha = a;
  // example state/seed (data, not decoration)
  mono(ctx, "s t a t e  4 e 3 e a f 7 a …   ·   s e e d  1 1 0 0 0", W / 2, 168, { size: 12, color: FAINT, align: "center" });
  ctx.restore();
}

// ---------------------------------------------------------------- rule
function drawRule(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  const bars = [
    { label: "T E N S O R   W A L K", frac: 1.0, value: "5.5", tag: "MAX" },
    { label: "L I V E   C U D A", frac: 0.48, value: null, tag: null },
    { label: "P E A K   C U D A", frac: 0.74, value: null, tag: null },
  ];
  const x0 = 190, wMax = 800, h = 44, gap = 46;
  bars.forEach((b, i) => {
    const y = 210 + i * (h + gap);
    const k = appear(t, s0 + 0.2 + i * 0.22, 0.7);
    const w = wMax * b.frac * outCubic(clamp(k));
    mono(ctx, b.label, x0, y - 16, { size: 12, color: MUTED });
    sunkenBar(ctx, x0, y, wMax, h);
    bar(ctx, x0, y, Math.max(0.001, w), h, i === 0 ? BROWN : "#c9c3b2");
    if (b.tag) {
      ctx.globalAlpha = a * clamp(k);
      mono(ctx, "M A X", x0 + wMax - 74, y + h / 2 + 1, { size: 14, weight: 700, color: PAPER, spacing: 3 });
      ctx.globalAlpha = a;
    }
    if (b.value) {
      ctx.font = `800 44px Inter, sans-serif`;
      ctx.fillStyle = BROWN;
      ctx.textBaseline = "middle";
      ctx.fillText(b.value, x0 + wMax + 26, y + h / 2);
      const wv = ctx.measureText(b.value).width;
      ctx.font = `400 22px Inter, sans-serif`;
      ctx.fillStyle = MUTED;
      ctx.fillText("GB", x0 + wMax + 26 + wv + 8, y + h / 2 + 4);
    }
  });
  ctx.restore();
}

// ---------------------------------------------------------------- footprint
function drawFootprint(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  const x0 = 140, y0 = 190, wMax = 560, h = 64, gap = 52;
  const segs = [
    { label: "W E I G H T S", frac: 1.0, col: BROWN },
    { label: "C A C H E S", frac: 0.24, col: "#a08a76" },
    { label: "B U F F E R S", frac: 0.13, col: "#c4bca8" },
  ];
  segs.forEach((s, i) => {
    const y = y0 + i * (h + gap);
    const k = appear(t, s0 + 0.18 + i * 0.25, 0.7);
    const w = wMax * s.frac * outCubic(clamp(k));
    mono(ctx, s.label, x0, y - 16, { size: 13, color: MUTED });
    sunkenBar(ctx, x0, y, wMax, h);
    bar(ctx, x0, y, Math.max(0.001, w), h, s.col);
  });
  ctx.font = `800 62px Inter, sans-serif`;
  ctx.fillStyle = INK;
  ctx.textBaseline = "middle";
  ctx.fillText("5.5 GB", x0 + wMax + 60, y0 + (h + gap) + h / 2);
  ctx.restore();
}

// ---------------------------------------------------------------- phases
function drawPhases(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  const cards = [
    { title: "SMOKE", n: "10", sub: "T R I A L S", tag: "G A T E   8 / 1 0", lock: false },
    { title: "DEVELOPMENT", n: "100", sub: "T R I A L S", tag: "S E L E C T I O N", lock: false },
    { title: "RESERVED", n: "500", sub: "T R I A L S", tag: "R E A D   O N C E", lock: true },
  ];
  const cw = 330, gap = 50, x0 = (W - (3 * cw + 2 * gap)) / 2;
  cards.forEach((c, i) => {
    const x = x0 + i * (cw + gap);
    const k = appear(t, s0 + 0.15 + i * 0.22, 0.5);
    ctx.globalAlpha = a * outCubic(clamp(k));
    card(ctx, x, 205, cw, 330, 14);
    if (c.lock) {
      ctx.strokeStyle = BROWN;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.roundRect(x + 6, 205 + 6, cw - 12, 330 - 12, 10);
      ctx.stroke();
    }
    mono(ctx, c.title.split("").join(" "), x + 34, 258, { size: 15, weight: 700, color: BROWN });
    ctx.font = `800 120px Inter, sans-serif`;
    ctx.fillStyle = INK;
    ctx.textBaseline = "middle";
    ctx.fillText(c.n, x + 34, 385);
    mono(ctx, c.sub, x + 34, 452, { size: 12, color: MUTED });
    hairline(ctx, x + 34, 482, x + cw - 34, HAIRLINE, 1);
    mono(ctx, c.tag, x + 34, 505, { size: 11, weight: 700, color: c.lock ? BROWN : MUTED });
    if (c.lock) {
      const lk = outBack(clamp(appear(t, s0 + 0.7, 0.45)), 2.2);
      ctx.save();
      ctx.translate(x + cw - 60, 240);
      ctx.scale(lk, lk);
      ctx.strokeStyle = BROWN;
      ctx.lineWidth = 3.2;
      ctx.beginPath();
      ctx.arc(0, -2, 9, Math.PI, 0);
      ctx.stroke();
      ctx.beginPath();
      ctx.roundRect(-13, -2, 26, 19, 4);
      ctx.fillStyle = BROWN;
      ctx.fill();
      ctx.restore();
    }
    ctx.globalAlpha = a;
  });
  // chevrons in the gaps between cards
  ctx.strokeStyle = FAINT;
  ctx.lineWidth = 3;
  for (const cx of [x0 + cw + gap / 2, x0 + 2 * cw + gap + gap / 2]) {
    ctx.beginPath();
    ctx.moveTo(cx - 11, 360);
    ctx.lineTo(cx, 378);
    ctx.lineTo(cx - 11, 396);
    ctx.stroke();
  }
  ctx.restore();
}

// ---------------------------------------------------------------- budget
function drawBudget(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  const cx = 360, cy = 360, r = 96;
  ctx.strokeStyle = INK;
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  const sweep = (t - s0) * 1.4;
  ctx.strokeStyle = BROWN;
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx + Math.sin(sweep) * r * 0.62, cy - Math.cos(sweep) * r * 0.62);
  ctx.stroke();
  for (let i = 0; i < 12; i++) {
    const ang = (i / 12) * Math.PI * 2;
    ctx.strokeStyle = i % 3 === 0 ? INK : FAINT;
    ctx.lineWidth = i % 3 === 0 ? 3 : 1.5;
    ctx.beginPath();
    ctx.moveTo(cx + Math.sin(ang) * (r - 14), cy - Math.cos(ang) * (r - 14));
    ctx.lineTo(cx + Math.sin(ang) * (r - 4), cy - Math.cos(ang) * (r - 4));
    ctx.stroke();
  }
  ctx.font = `800 74px Inter, sans-serif`;
  ctx.fillStyle = INK;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText("≈ 10 s", 620, 348);
  mono(ctx, "P E R   E P I S O D E", 620, 412, { size: 13, color: MUTED });
  ctx.restore();
}

// ---------------------------------------------------------------- paired
function drawPaired(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  const cols = 50, rows = 10, dot = 6;
  const gridW = 840, gapX = gridW / cols, rowGap = 15, groupGap = 62;
  const x0 = (W - gridW) / 2, y0 = 200;
  const sweep = clamp((t - s0 - 0.25) / 3.1);
  const labels = ["R E F E R E N C E", "W I N N E R"];
  for (let gIdx = 0; gIdx < 2; gIdx++) {
    const gy = y0 + gIdx * (rows * rowGap + groupGap);
    mono(ctx, labels[gIdx], x0, gy - 24, { size: 12, weight: 700, color: BROWN });
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const frac = (r * cols + c) / (rows * cols);
        const lit = frac <= sweep;
        ctx.fillStyle = lit ? (gIdx === 0 ? BROWN : BROWN_DEEP) : "rgba(31,28,23,0.15)";
        ctx.beginPath();
        ctx.arc(x0 + c * gapX + gapX / 2, gy + r * rowGap, dot / 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  mono(ctx, "T H E   S A M E   5 0 0   S T A T E S   ·   T H E   S A M E   S E E D S", W / 2, 566, { size: 12, color: MUTED, align: "center" });
  ctx.restore();
}

// ---------------------------------------------------------------- audit
function drawAudit(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  const x = 300, y = 140, w = 640, h = 330;
  card(ctx, x, y, w, h, 12);
  mono(ctx, "s o l u t i o n . p y", x + 30, y + 40, { size: 13, color: MUTED });
  // fake diff lines
  const lines = [
    { w: 0.86, flag: false },
    { w: 0.62, flag: false },
    { w: 0.94, flag: false },
    { w: 0.5, flag: true },
    { w: 0.7, flag: false },
    { w: 0.4, flag: false },
  ];
  lines.forEach((l, i) => {
    const ly = y + 84 + i * 40;
    if (l.flag) {
      ctx.fillStyle = "rgba(124,63,42,0.10)";
      ctx.fillRect(x + 24, ly - 13, w - 48, 26);
    }
    ctx.fillStyle = l.flag ? TERRA : "#c9c3b2";
    ctx.beginPath();
    ctx.roundRect(x + 30, ly - 6, (w - 130) * l.w, 12, 6);
    ctx.fill();
    if (l.flag) {
      crossMark(ctx, x + w - 40, ly, 9, TERRA);
    } else {
      checkMark(ctx, x + w - 40, ly, 7, BROWN);
    }
  });
  // magnifier
  const k = appear(t, s0 + 0.3, 0.6);
  const magn = 1;
  ctx.strokeStyle = INK;
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.arc(x + w - 64, y + h - 44 - (1 - k) * 30, 34, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x + w - 42, y + h - 21 - (1 - k) * 30);
  ctx.lineTo(x + w + 12, y + h + 36);
  ctx.lineWidth = 8;
  ctx.stroke();
  ctx.restore();
}

// ---------------------------------------------------------------- gate
function drawGate(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  const cx = W / 2, cy = 356, r = 150;
  ctx.strokeStyle = INK;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = HAIRLINE;
  ctx.beginPath();
  ctx.arc(cx, cy, r - 16, 0, Math.PI * 2);
  ctx.stroke();
  ctx.font = `800 84px Inter, sans-serif`;
  ctx.fillStyle = INK;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("8/10", cx, cy + 4);
  mono(ctx, "P R O M O T E S", cx, cy + r + 38, { size: 13, weight: 700, color: BROWN, align: "center" });
  mono(ctx, "E V A L U A T O R   ·   F I X E D", cx, 138, { size: 12, color: MUTED, align: "center" });
  ctx.restore();
}

// ---------------------------------------------------------------- search
function drawSearch(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  // replay card feeding the tree
  card(ctx, 150, 138, 250, 84, 10);
  mono(ctx, "H I S T O R Y", 275, 168, { size: 13, weight: 700, color: MUTED, align: "center" });
  mono(ctx, "R E P L A Y", 275, 194, { size: 13, weight: 700, color: MUTED, align: "center" });
  const arrowK = appear(t, s0 + 0.3, 0.5);
  ctx.strokeStyle = BROWN;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(275, 222);
  ctx.lineTo(275, lerp(222, 262, arrowK));
  ctx.stroke();
  if (arrowK > 0.85) {
    mono(ctx, "π", 300, 246, { size: 20, weight: 700, color: BROWN });
  }
  // tree
  const root = { x: 275, y: 300 };
  const levels = [
    [root],
    [{ x: 190, y: 408 }, { x: 372, y: 408 }],
    [{ x: 112, y: 508 }, { x: 232, y: 508 }, { x: 322, y: 508 }, { x: 446, y: 508 }],
  ];
  ctx.strokeStyle = INK_SOFT;
  ctx.lineWidth = 2.5;
  const edges = [
    [root, levels[1][0]],
    [root, levels[1][1]],
    [levels[1][0], levels[2][0]],
    [levels[1][0], levels[2][1]],
    [levels[1][1], levels[2][2]],
    [levels[1][1], levels[2][3]],
  ];
  edges.forEach(([p, q], i) => {
    const k = appear(t, s0 + 0.4 + i * 0.13, 0.3);
    ctx.globalAlpha = a * clamp(k);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(lerp(p.x, q.x, k), lerp(p.y, q.y, k));
    ctx.stroke();
    ctx.globalAlpha = a;
  });
  const nodes = [levels[1][0], levels[1][1], ...levels[2]];
  nodes.forEach((n, i) => {
    const k = appear(t, s0 + 0.45 + i * 0.13, 0.35);
    const e = outBack(clamp(k), 2.0);
    ctx.fillStyle = i === 5 ? BROWN : CARD;
    ctx.beginPath();
    ctx.arc(n.x, n.y, 11 * e, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2;
    ctx.stroke();
  });
  ctx.beginPath();
  ctx.arc(root.x, root.y, 13, 0, Math.PI * 2);
  ctx.fillStyle = BROWN;
  ctx.fill();
  // right column: propose -> score
  const rx = 640;
  card(ctx, rx, 150, 440, 120, 12);
  mono(ctx, "A G E N T", rx + 40, 190, { size: 14, weight: 700, color: BROWN });
  mono(ctx, "P R O P O S E S   A   S O L U T I O N", rx + 40, 224, { size: 12, color: MUTED });
  ctx.strokeStyle = BROWN;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(rx + 220, 282);
  ctx.lineTo(rx + 220, 314);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(rx + 220, 322);
  ctx.lineTo(rx + 220 - 11, 322 - 17);
  ctx.moveTo(rx + 220, 322);
  ctx.lineTo(rx + 220 + 11, 322 - 17);
  ctx.stroke();
  card(ctx, rx, 330, 440, 120, 12);
  mono(ctx, "E V A L U A T O R", rx + 40, 370, { size: 14, weight: 700, color: BROWN });
  mono(ctx, "S C O R E S   I T   O N   T H E   G A T E", rx + 40, 404, { size: 12, color: MUTED });
  ctx.restore();
}

// ---------------------------------------------------------------- exploit
function drawExploit(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  ctx.font = `800 175px Inter, sans-serif`;
  ctx.fillStyle = INK;
  ctx.fillText("82", W / 2 - 165, 330);
  ctx.fillText("82", W / 2 + 165, 330);
  ctx.font = `400 64px Inter, sans-serif`;
  ctx.fillStyle = MUTED;
  ctx.fillText("=", W / 2, 322);
  mono(ctx, "T H E   S A M E", W / 2, 452, { size: 14, weight: 700, color: MUTED, align: "center" });
  // stamp
  const stampT = cues?.d82 ? cues.d82 + 1.15 : s0 + 4.5;
  const k = appear(t, stampT, 0.35);
  if (k > 0) {
    const e = outBack(clamp(k), 2.4);
    ctx.save();
    ctx.translate(W / 2, 210);
    ctx.rotate(-0.07);
    ctx.globalAlpha = a * clamp(k);
    ctx.scale(e, e);
    ctx.strokeStyle = TERRA;
    ctx.lineWidth = 4;
    const tw = 480, th = 74;
    ctx.beginPath();
    ctx.roundRect(-tw / 2, -th / 2, tw, th, 8);
    ctx.stroke();
    mono(ctx, "D I S Q U A L I F I E D", 0, 1, { size: 30, weight: 700, color: TERRA, align: "center", spacing: 6 });
    ctx.restore();
    ctx.globalAlpha = a;
  }
  ctx.restore();
}

// ---------------------------------------------------------------- results
function drawResults(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  const x0 = 170, wMax = 640, h = 54, gap = 26;
  const rows = [
    { label: "R E F E R E N C E", frac: 0.82, value: "82" },
    { label: "W I N N E R", frac: 0.84, value: "84" },
  ];
  mono(ctx, "D E V E L O P M E N T   ·   1 0 0", x0, 158, { size: 12, weight: 700, color: MUTED });
  rows.forEach((r, i) => {
    const y = 192 + i * (h + gap);
    const k = appear(t, s0 + 0.2 + i * 0.25, 0.8);
    const w = wMax * r.frac * outCubic(clamp(k));
    sunkenBar(ctx, x0, y, wMax, h);
    bar(ctx, x0, y, Math.max(0.001, w), h, i === 0 ? "#c9c3b2" : BROWN);
    ctx.font = `800 42px Inter, sans-serif`;
    ctx.fillStyle = INK;
    ctx.textBaseline = "middle";
    ctx.fillText(r.value, x0 + wMax + 28, y + h / 2);
    mono(ctx, r.label, x0, y - 13, { size: 11, color: MUTED });
  });
  hairline(ctx, x0, 394, x0 + wMax + 130, HAIRLINE, 1);
  mono(ctx, "M E M O R Y   R E M O V E D", x0, 424, { size: 12, weight: 700, color: MUTED });
  const k2 = appear(t, s0 + 0.7, 0.9);
  const v1 = lerp(0, 87, outCubic(clamp(k2))).toFixed(0);
  const v2 = lerp(0, 84, outCubic(clamp(k2))).toFixed(0);
  ctx.textBaseline = "middle";
  ctx.font = `800 96px Inter, sans-serif`;
  ctx.fillStyle = BROWN;
  ctx.fillText(`−${v1}%`, x0, 498);
  ctx.fillText(`−${v2}%`, x0 + 450, 498);
  mono(ctx, "C Y C L E   1", x0, 556, { size: 11, color: MUTED });
  mono(ctx, "C Y C L E   2", x0 + 450, 556, { size: 11, color: MUTED });
  ctx.restore();
}

// ---------------------------------------------------------------- reserved
function drawReserved(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  // padlock, closes on the cue
  const lk = outBack(clamp(appear(t, s0 + 0.12, 0.6)), 2.0);
  const cx = W / 2, ly = 205, r = 58;
  ctx.save();
  ctx.translate(cx, ly);
  ctx.globalAlpha = a * lk;
  ctx.strokeStyle = BROWN;
  ctx.lineWidth = 8;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.arc(0, 6, r * 0.62, Math.PI + lerp(1.6, 0.001, outCubic(clamp(appear(t, s0 + 0.12, 0.75)))), -lerp(1.6, 0.001, outCubic(clamp(appear(t, s0 + 0.12, 0.75)))));
  ctx.stroke();
  ctx.beginPath();
  ctx.roundRect(-r * 0.78, 2, r * 1.56, r, 12);
  ctx.fillStyle = BROWN;
  ctx.fill();
  ctx.beginPath();
  ctx.arc(0, 2 + r * 0.42, 5.5, 0, Math.PI * 2);
  ctx.fillStyle = PAPER;
  ctx.fill();
  ctx.restore();
  ctx.globalAlpha = a;
  // counters
  const cols = [
    { n: 475, label: "R E F E R E N C E", x: W / 2 - 330 },
    { n: 478, label: "F I R S T   W I N N E R", x: W / 2 + 330 },
  ];
  for (const c of cols) {
    const k = appear(t, s0 + 0.55, 1.0);
    const shown = Math.round(lerp(0, c.n, outCubic(clamp(k))));
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.font = `800 128px Inter, sans-serif`;
    const wnum = ctx.measureText(String(shown)).width;
    ctx.font = `400 40px Inter, sans-serif`;
    const wsub = ctx.measureText(" /500").width;
    const x = c.x - (wnum + wsub) / 2;
    ctx.font = `800 128px Inter, sans-serif`;
    ctx.fillStyle = INK;
    ctx.fillText(String(shown), x, 420);
    ctx.font = `400 40px Inter, sans-serif`;
    ctx.fillStyle = MUTED;
    ctx.fillText(" /500", x + wnum, 438);
    mono(ctx, c.label, c.x, 516, { size: 12, weight: 700, color: MUTED, align: "center" });
  }
  ctx.restore();
}

// ---------------------------------------------------------------- honest
function drawHonest(ctx, t, s0, cues, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  const items = [
    { label: "L O S S", sub: "I S   N O T   T H E   R O B O T" },
    { label: "E R R O R", sub: "D O E S N ' T   R A N K   W I N N E R S" },
    { label: "S M O K E", sub: "I S   N O T   S K I L L" },
  ];
  const cw = 330, gap = 50, x0 = (W - (3 * cw + 2 * gap)) / 2;
  items.forEach((item, i) => {
    const x = x0 + i * (cw + gap);
    const k = appear(t, s0 + 0.18 + i * 0.2, 0.4);
    ctx.globalAlpha = a * outCubic(clamp(k));
    card(ctx, x, 210, cw, 230, 14);
    ctx.globalAlpha = a * outCubic(clamp(k)) * 0.9;
    crossMark(ctx, x + cw - 72, 258, 24, TERRA);
    ctx.globalAlpha = a * outCubic(clamp(k));
    mono(ctx, item.label, x + cw / 2, 330, { size: 30, weight: 700, color: INK, align: "center", spacing: 4 });
    const sub = item.sub;
    mono(ctx, sub, x + cw / 2, 382, { size: 10, color: MUTED, align: "center", spacing: 1.2 });
    ctx.globalAlpha = a;
  });
  ctx.restore();
}

// ---------------------------------------------------------------- close
function drawClose(ctx, t, s0, cues, total, W, H) {
  const { a, dy } = env(t, s0);
  ctx.save();
  ctx.translate(0, dy);
  ctx.globalAlpha = a;
  mono(ctx, "A   R E L E A S E D   P O L I C Y", W / 2, 180, { size: 13, weight: 700, color: MUTED, align: "center" });
  // shrinking bar
  const y = 262, h = 96, x0 = 140;
  const full = 660;
  const k = appear(t, s0 + 0.25, 1.25);
  const kk = outCubic(clamp(k));
  const w = lerp(full, full / 6, kk);
  sunkenBar(ctx, x0, y, full, h);
  bar(ctx, x0, y, w, h, BROWN);
  ctx.font = `800 120px Inter, sans-serif`;
  ctx.fillStyle = INK;
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  ctx.fillText("1/6", x0 + full + 44, y + h / 2);
  mono(ctx, "S A M E   S T A T E S   ·   S A M E   S E E D S", W / 2, 442, { size: 13, weight: 700, color: BROWN, align: "center" });
  mono(ctx, "S A M E   A N S W E R", W / 2, 474, { size: 13, weight: 700, color: BROWN, align: "center" });
  ctx.restore();
}

export const DRAW = {
  hook: drawHook,
  why: drawWhy,
  policy: drawPolicy,
  success: drawSuccess,
  episode: drawEpisode,
  rule: drawRule,
  footprint: drawFootprint,
  phases: drawPhases,
  budget: drawBudget,
  paired: drawPaired,
  audit: drawAudit,
  gate: drawGate,
  search: drawSearch,
  exploit: drawExploit,
  results: drawResults,
  reserved: drawReserved,
  honest: drawHonest,
  close: drawClose,
};
