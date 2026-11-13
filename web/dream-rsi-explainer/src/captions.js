// Word-highlight subtitles: chunked single-line pages, active word pops in a highlight box.
import { clamp, inCubic, lerp, outBack, outCubic } from "./ease.js";

const FONT = (size) => `800 ${size}px Inter, sans-serif`;
const SIZE = 54;
const MAX_W = 1060;
const BASE_Y = 648;

export function buildPages(ctx, sentences) {
  ctx.font = FONT(SIZE);
  const pages = [];
  const sp = 30;
  for (const s of sentences) {
    let cur = [];
    let w = 0;
    const flush = () => {
      if (cur.length) pages.push({ words: cur, width: w - sp, sentence: s });
      cur = [];
      w = 0;
    };
    for (const tok of s.tokens) {
      const label = tok.shown + tok.punct;
      const ww = ctx.measureText(label).width;
      if (cur.length && w + ww > MAX_W) flush();
      cur.push({ ...tok, label, w: ww, x: w });
      w += ww + sp;
      if (cur.length >= 2 && /[.?!]$/.test(tok.punct)) flush();
      else if (cur.length >= 3 && /[,:;]$/.test(tok.punct)) flush();
      else if (cur.length >= 6) flush();
    }
    flush();
  }
  for (const pg of pages) {
    pg.start = pg.words[0].start;
    pg.end = pg.words[pg.words.length - 1].end;
  }
  pages.forEach((pg, i) => {
    const next = pages[i + 1];
    pg.show0 = pg.start - 0.12;
    pg.hide = Math.min(pg.end + 0.28, next ? Math.max(pg.end + 0.02, next.start - 0.08) : 1e9);
  });
  return pages;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export function drawCaptions(ctx, pages, t, W, H) {
  const g = ctx.createLinearGradient(0, 520, 0, H);
  g.addColorStop(0, "rgba(3,4,10,0)");
  g.addColorStop(1, "rgba(3,4,10,0.88)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 520, W, H - 520);

  const pg = pages.find((p) => t >= p.show0 && t < p.hide);
  if (!pg) return;
  const enter = clamp((t - pg.show0) / 0.2);
  const exit = clamp((t - (pg.hide - 0.08)) / 0.08);
  const e = outBack(enter, 2.2);
  const alpha = outCubic(enter) * (1 - exit);
  const sc = lerp(0.84, 1, e) * (1 - 0.04 * exit);
  const dy = lerp(24, 0, outCubic(enter)) + 10 * exit;
  const x0 = (W - pg.width) / 2;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(W / 2, BASE_Y - 18 + dy);
  ctx.scale(sc, sc);
  ctx.translate(-W / 2, -(BASE_Y - 18));
  ctx.font = FONT(SIZE);
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.lineJoin = "round";

  // Pass 1: active word highlight pills (behind all text)
  for (const w of pg.words) {
    const x = x0 + w.x;
    const active = t >= w.start && t < w.end;
    const p = outBack(clamp((t - w.start) / 0.13), 2.6);
    if (active || (t >= w.start && t < w.end + 0.08)) {
      const k = active ? 1 : 1 - clamp((t - w.end) / 0.08);
      const cx = x + w.w / 2;
      const cy = BASE_Y - 18;
      const s = lerp(0.55, 1, p) * (w.emph ? 1.08 : 1);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-0.03 * (1 - p) + (w.emph ? -0.015 : 0));
      ctx.scale(s, s);
      ctx.globalAlpha = alpha * (active ? 1 : k);
      ctx.shadowColor = w.emph ? "rgba(255,61,154,0.85)" : "rgba(255,225,77,0.75)";
      ctx.shadowBlur = 28;
      ctx.fillStyle = w.emph ? "#ff3d9a" : "#ffe14d";
      roundRect(ctx, -w.w / 2 - 12, -44, w.w + 24, 74, 16);
      ctx.fill();
      ctx.restore();
    }
  }
  // Pass 2: word text on top
  for (const w of pg.words) {
    const x = x0 + w.x;
    const active = t >= w.start && t < w.end;
    const spoken = t >= w.end;
    const p = outBack(clamp((t - w.start) / 0.13), 2.6);
    ctx.save();
    const lift = active ? -4 * p : 0;
    ctx.translate(x + w.w / 2, BASE_Y + lift);
    const ws = active ? lerp(0.9, 1, p) * (w.emph ? 1.08 : 1) : 1;
    ctx.scale(ws, ws);
    ctx.translate(-(x + w.w / 2), -BASE_Y);
    ctx.lineWidth = 9;
    ctx.strokeStyle = "rgba(0,0,0,0.88)";
    if (!active) ctx.strokeText(w.label, x, BASE_Y);
    if (active) ctx.fillStyle = w.emph ? "#ffffff" : "#0a0c14";
    else if (spoken) ctx.fillStyle = w.emph ? "#ffe14d" : "#ffffff";
    else ctx.fillStyle = "rgba(255,255,255,0.42)";
    ctx.fillText(w.label, x, BASE_Y);
    ctx.restore();
  }
  ctx.restore();
}

export function drawWipe(ctx, t, starts, W, H) {
  for (const s0 of starts) {
    const a = s0 - 0.25;
    const k = (t - a) / 0.45;
    if (k <= 0 || k >= 1) continue;
    const cover = k < 0.5 ? outCubic(k / 0.5) : 1 - inCubic((k - 0.5) / 0.5);
    const sk = 180;
    for (const [col, off] of [["#ff3d9a", 0], ["#3df0ff", 60], ["#0a0c14", 120]]) {
      const c = clamp(cover * 1.25 - (off / (W + sk)) * 1.25);
      const x = -sk + (W + 2 * sk) * c;
      ctx.fillStyle = col;
      ctx.beginPath();
      if (k < 0.5) {
        ctx.moveTo(-sk, 0);
        ctx.lineTo(x, 0);
        ctx.lineTo(x - sk, H);
        ctx.lineTo(-sk, H);
      } else {
        ctx.moveTo(x + sk, 0);
        ctx.lineTo(W + sk, 0);
        ctx.lineTo(W + sk, H);
        ctx.lineTo(x, H);
      }
      ctx.closePath();
      ctx.fill();
    }
  }
}
