// OhRats paper design system for the project explainer: warm paper, hairlines, one brown accent.
export const PAPER = "#efece5";
export const PAPER_DEEP = "#e8e5d9";
export const CARD = "#ffffff";
export const INK = "#1f1c17";
export const INK_SOFT = "#3a3730";
export const MUTED = "#6f6b5e";
export const FAINT = "#a8a492";
export const HAIRLINE = "#d8d3c4";
export const BROWN = "#604737";
export const BROWN_DEEP = "#463325";
export const TERRA = "#7c3f2a";

export const MONO = (w, sz) => `${w} ${sz}px 'JetBrains Mono', monospace`;
export const INTER = (w, sz) => `${w} ${sz}px Inter, sans-serif`;

export function hairline(ctx, x0, y, x1, col = HAIRLINE, lw = 1) {
  ctx.strokeStyle = col;
  ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.moveTo(x0, y);
  ctx.lineTo(x1, y);
  ctx.stroke();
}

export function card(ctx, x, y, w, h, r = 10) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.fillStyle = CARD;
  ctx.fill();
  ctx.strokeStyle = HAIRLINE;
  ctx.lineWidth = 1;
  ctx.stroke();
}

// Small mono caps label with letterspacing, centred on x or drawn from x.
export function mono(ctx, text, x, y, { size = 13, weight = 400, color = MUTED, align = "left", spacing = 2.2 } = {}) {
  ctx.font = MONO(weight, size);
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  try { ctx.letterSpacing = `${spacing}px`; } catch { /* noop */ }
  ctx.fillText(text, x, y);
  try { ctx.letterSpacing = "0px"; } catch { /* noop */ }
}

// The two-tone headline from the OhRats site: dark clause, gray clause.
export function statement(ctx, strong, soft, x, y, { size = 84, align = "left", softColor = MUTED } = {}) {
  ctx.textAlign = align;
  ctx.textBaseline = "alphabetic";
  ctx.font = INTER(750, size);
  ctx.fillStyle = INK;
  ctx.fillText(strong, x, y);
  const w = ctx.measureText(strong).width;
  ctx.font = INTER(400, size);
  ctx.fillStyle = softColor;
  ctx.fillText(" " + soft, x + w, y);
}

// Chip "01 — LABEL" in the site's eyebrow style: brown numeral, dash, ink label.
export function eyebrow(ctx, num, label, x, y) {
  ctx.textBaseline = "middle";
  ctx.font = MONO(700, 13);
  ctx.fillStyle = BROWN;
  ctx.textAlign = "left";
  try { ctx.letterSpacing = "1.8px"; } catch { /* noop */ }
  ctx.fillText(num, x, y);
  const nw = ctx.measureText(num).width;
  hairline(ctx, x + nw + 8, y, x + nw + 26, BROWN, 1.5);
  ctx.fillStyle = INK;
  ctx.font = MONO(400, 13);
  ctx.fillText(label, x + nw + 34, y);
  try { ctx.letterSpacing = "0px"; } catch { /* noop */ }
}

export function checkMark(ctx, cx, cy, r, color = BROWN) {
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(2.5, r * 0.16);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(cx - r * 0.55, cy + r * 0.02);
  ctx.lineTo(cx - r * 0.12, cy + r * 0.45);
  ctx.lineTo(cx + r * 0.65, cy - r * 0.45);
  ctx.stroke();
}

export function crossMark(ctx, cx, cy, r, color = TERRA) {
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(2.5, r * 0.16);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(cx - r * 0.45, cy - r * 0.45);
  ctx.lineTo(cx + r * 0.45, cy + r * 0.45);
  ctx.moveTo(cx + r * 0.45, cy - r * 0.45);
  ctx.lineTo(cx - r * 0.45, cy + r * 0.45);
  ctx.stroke();
}

// Stylised memory-chip glyph, the "robot brain" motif. Draws centred at (cx, cy), size s.
export function chip(ctx, cx, cy, s, { lit = 1, accent = BROWN } = {}) {
  const w = s, h = s * 0.78;
  ctx.save();
  ctx.translate(cx, cy);
  // pins
  ctx.strokeStyle = INK_SOFT;
  ctx.lineWidth = s * 0.032;
  ctx.lineCap = "round";
  const pin = s * 0.085;
  for (const side of [-1, 1]) {
    for (let i = -2; i <= 2; i++) {
      const yy = (i / 2.4) * h * 0.62;
      ctx.beginPath();
      ctx.moveTo(side * (w / 2), yy);
      ctx.lineTo(side * (w / 2 + pin), yy);
      ctx.stroke();
    }
  }
  // body
  ctx.beginPath();
  ctx.roundRect(-w / 2, -h / 2, w, h, s * 0.09);
  ctx.fillStyle = CARD;
  ctx.fill();
  ctx.strokeStyle = INK;
  ctx.lineWidth = s * 0.026;
  ctx.stroke();
  // inner die
  ctx.beginPath();
  ctx.roundRect(-w * 0.22, -h * 0.24, w * 0.44, h * 0.48, 6);
  ctx.strokeStyle = HAIRLINE;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.font = MONO(700, s * 0.13);
  ctx.fillStyle = accent;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("V L A", 0, 0);
  // lit grid dots (memory bytes)
  ctx.fillStyle = `rgba(96,71,55,${0.25 + 0.75 * lit})`;
  const n = 5;
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < 7; c++) {
      const xx = -w * 0.36 + (c / 6) * w * 0.72;
      const yy = -h * 0.33 + (r / 4) * h * 0.66;
      ctx.beginPath();
      ctx.arc(xx, yy, s * 0.014, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

export function lerp(a, b, k) {
  return a + (b - a) * k;
}

export function smooth(x) {
  x = Math.min(1, Math.max(0, x));
  return x * x * (3 - 2 * x);
}

export function appear(t, t0, d = 0.4) {
  return Math.min(1, Math.max(0, (t - t0) / d));
}

export function outBack(x, s = 1.70158) {
  x = Math.min(1, Math.max(0, x)) - 1;
  return 1 + (s + 1) * x ** 3 + s * x ** 2;
}
