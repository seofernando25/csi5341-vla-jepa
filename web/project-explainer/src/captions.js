// Word-highlighted subtitles on paper: faded ink, brown karaoke sweep revealing the active word.
import { PAPER, INK, MUTED, BROWN, BROWN_DEEP, HAIRLINE, INTER } from "./lib.js";
import { clamp, lerp, outBack, outCubic } from "./ease.js";

const SIZE = 44;
const MAX_W = 1140;
const FONT = INTER(600, SIZE);
const BASE_Y = 672;
const RULE_Y = 610;
const PAD = 72;

export function buildPages(ctx, sentences) {
  ctx.font = FONT;
  const pages = [];
  const sp = 15;
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
    }
    flush();
  }
  for (const pg of pages) {
    pg.start = pg.words[0].start;
    pg.end = pg.words[pg.words.length - 1].end;
  }
  pages.forEach((pg, i) => {
    const next = pages[i + 1];
    pg.show0 = pg.start - 0.14;
    pg.hide = Math.min(pg.end + 0.3, next ? Math.max(pg.end + 0.02, next.start - 0.1) : 1e9);
  });
  return pages;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

export function drawCaptions(ctx, pages, t, W, H) {
  // hairline separating content from captions + a soft seat for the text
  const g = ctx.createLinearGradient(0, RULE_Y - 30, 0, H);
  g.addColorStop(0, "rgba(232,229,217,0)");
  g.addColorStop(1, "rgba(232,229,217,0.92)");
  ctx.fillStyle = g;
  ctx.fillRect(0, RULE_Y - 30, W, H - (RULE_Y - 30));
  ctx.strokeStyle = HAIRLINE;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD, RULE_Y);
  ctx.lineTo(W - PAD, RULE_Y);
  ctx.stroke();

  const pg = pages.find((p) => t >= p.show0 && t < p.hide);
  if (!pg) return;
  const enter = clamp((t - pg.show0) / 0.22);
  const exit = clamp((t - (pg.hide - 0.09)) / 0.09);
  const e = outBack(enter, 2.0);
  const alpha = outCubic(enter) * (1 - exit);
  const dy = lerp(10, 0, outCubic(enter)) + 6 * exit;
  const x0 = (W - pg.width) / 2;

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(0, dy);
  ctx.font = FONT;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.lineJoin = "round";

  // Pass 1: faded or spoken base text; pass 2: brown sweep pill + paper text for the active word.
  for (const w of pg.words) {
    const x = x0 + w.x;
    const active = t >= w.start && t < w.end;
    const spoken = t >= w.end;
    const baseCol = spoken ? (w.emph ? BROWN : INK) : "rgba(31,28,23,0.30)";
    ctx.fillStyle = baseCol;
    ctx.fillText(w.label, x, BASE_Y);
    if (active) {
      const p = clamp((t - w.start) / Math.max(0.05, w.end - w.start));
      const pillW = w.w + 22;
      const pillH = 50;
      const py = BASE_Y - 38;
      const cx = x + w.w / 2;
      // Spring pop on word start, then a left-to-right reveal.
      const pop = outBack(clamp((t - w.start) / 0.16), 2.4);
      ctx.save();
      ctx.translate(cx, BASE_Y + dy - 8);
      ctx.scale(lerp(0.9, 1, pop), lerp(0.9, 1, pop));
      ctx.translate(-cx, -(BASE_Y + dy - 8));
      ctx.beginPath();
      ctx.rect(x - 11, py - 8, pillW * p, pillH + 16);
      ctx.clip();
      roundRect(ctx, x - 11, py, pillW, pillH, 12);
      ctx.fillStyle = w.emph ? BROWN_DEEP : BROWN;
      ctx.fill();
      ctx.fillStyle = PAPER;
      ctx.fillText(w.label, x, BASE_Y);
      ctx.restore();
    }
  }
  ctx.restore();
}

// Ink-and-brown diagonal slabs sweep left to right as a whip transition.
export function drawWipe(ctx, t, starts, W, H) {
  for (const s0 of starts) {
    const u = (t - s0) / 0.55;
    if (u <= 0 || u >= 1) continue;
    const e = u * u * (3 - 2 * u); // smoothstep across the whole sweep
    const base = lerp(-W - 340, W + 60, e);
    for (const [col, off, am] of [
      [BROWN, 0, 0.95],
      [INK, 92, 1],
      [BROWN, 184, 0.38],
    ]) {
      const x = base + off;
      ctx.globalAlpha = am;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x + W, 0);
      ctx.lineTo(x + W - 340, H);
      ctx.lineTo(x - 340, H);
      ctx.closePath();
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
}
