// The OhRats-style paper backdrop: warm gradient, faint blueprint grid, slow dust, vignette.
import { PAPER, PAPER_DEEP } from "./lib.js";

export function fillNoise(nctx) {
  const img = nctx.createImageData(128, 128);
  let seed = 7;
  const rnd = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
  for (let i = 0; i < img.data.length; i += 4) {
    const v = 200 + Math.floor(rnd() * 55);
    img.data[i] = v;
    img.data[i + 1] = v;
    img.data[i + 2] = v;
    img.data[i + 3] = 14 + Math.floor(rnd() * 14);
  }
  nctx.putImageData(img, 0, 0);
}

export function createBackground(W, H, noiseArg) {
  const noise = noiseArg || document.createElement("canvas");
  if (!noiseArg) {
    noise.width = noise.height = 128;
    fillNoise(noise.getContext("2d"));
  }

  let dustSeed = 1234;
  const rnd = () => {
    dustSeed = (dustSeed * 16807) % 2147483647;
    return dustSeed / 2147483647;
  };
  const dust = [];
  for (let i = 0; i < 46; i++) {
    dust.push({
      x: rnd() * W,
      y: rnd() * H,
      r: 0.7 + rnd() * 1.7,
      a: 0.12 + rnd() * 0.2,
      v: 2.2 + rnd() * 2.4,
      ph: rnd() * Math.PI * 2,
    });
  }

  function draw(ctx, t) {
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, PAPER);
    g.addColorStop(1, PAPER_DEEP);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    // faint blueprint grid, drifting slowly up-left
    const step = 44;
    const off = (t * 2.4) % step;
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(31,28,23,0.045)";
    ctx.beginPath();
    for (let x = -step; x < W + step; x += step) {
      ctx.moveTo(x - off, 0);
      ctx.lineTo(x - off, H);
    }
    for (let y = -step; y < H + step; y += step) {
      ctx.moveTo(0, y - off);
      ctx.lineTo(W, y - off);
    }
    ctx.stroke();
    ctx.strokeStyle = "rgba(31,28,23,0.07)";
    ctx.beginPath();
    for (let x = -step * 5; x < W + step * 5; x += step * 5) {
      ctx.moveTo(x - off, 0);
      ctx.lineTo(x - off, H);
    }
    ctx.stroke();

    // drifting dust
    ctx.save();
    for (const d of dust) {
      const yy = (d.y - t * d.v) % H;
      const y = yy < 0 ? yy + H : yy;
      const xx = d.x + Math.sin(t * 0.32 + d.ph) * 12;
      ctx.fillStyle = `rgba(31,28,23,${d.a})`;
      ctx.beginPath();
      ctx.arc(xx, y, d.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // vignette
    const v = ctx.createRadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, W * 0.85);
    v.addColorStop(0, "rgba(58,52,40,0)");
    v.addColorStop(1, "rgba(58,52,40,0.16)");
    ctx.fillStyle = v;
    ctx.fillRect(0, 0, W, H);

    // paper grain
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = ctx.createPattern(noise, "repeat");
    ctx.fillRect(0, 0, W, H);
    ctx.globalAlpha = 1;
  }

  return { draw };
}
