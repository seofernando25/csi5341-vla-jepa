/* Authored visual beats. Independent of audio callbacks; identical for browser and export. */
const CHOREOGRAPHY = {
  0: { length: 18, act: "WATCH → ACT" },
  2: { length: 10, act: "THE DATA GAP" },
  3: { length: 62, act: "THE TRAINING PATH" },
  4: { length: 40, act: "PREDICT THE OUTCOME" },
  5: { length: 19, act: "REPRESENT THE TRANSITION" },
  6: { length: 19, act: "PREDICT THE STATE" },
  7: { length: 21, act: "LEARN FROM THE ERROR" },
  8: { length: 33, act: "PROTECT THE BOUNDARY" },
  9: { length: 55, act: "LEARN A FIELD" },
  10: { length: 34, act: "FOLLOW THE FIELD" },
  11: { length: 43, act: "TWO SOURCES, ONE POLICY" },
  12: { length: 10, act: "A NEAR TIE" },
  13: { length: 17, act: "ROBUSTNESS" },
  14: { length: 16, act: "THE ABLATION" },
  15: { length: 14, act: "THE COUNTEREXAMPLE" },
  16: { length: 24, act: "OUR PROPOSAL" },
  17: { length: 23, act: "THE LIMITS" },
  18: { length: 15, act: "THE PAPER’S CONTRIBUTION" },
};
function motionLength(i = index) {
  return CHOREOGRAPHY[i]?.length || 8;
}
function sceneSeconds() {
  return p * motionLength();
}
const smoothOut = (x) => 1 - Math.pow(1 - clamp(x), 3);
const springOut = (x) => {
  x = clamp(x);
  return x === 1 ? 1 : 1 - Math.cos(x * Math.PI * 2.1) * Math.exp(-6 * x);
};
function emerge(c, seconds, start, draw, dy = 14) {
  let t = smoothOut((seconds - start) / 0.5);
  c.save();
  c.translate(0, dy * (1 - t));
  c.globalAlpha *= t;
  draw(t);
  c.restore();
}
function travellingSignal(c, pts, t, color = blue) {
  if (t <= 0 || t >= 1) return;
  const a = bez(t, pts);
  c.save();
  c.globalAlpha = 0.14;
  dotC(c, ...a, 10, color);
  c.globalAlpha = 1;
  dotC(c, ...a, 3, color);
  c.restore();
}
function frameToFeatures(c, im, from, to, t, color = blue) {
  if (t <= 0 || t >= 1) return;
  let e = smoothOut(t);
  const n = 12;
  for (let j = 0; j < n; j++) {
    let w = from[2] / n,
      xx = from[0] + j * w,
      yy = from[1] + from[3] / 2;
    const tx = to[0] - to[2] / 2 + (j * to[2]) / n,
      ty = to[1],
      height = mix(from[3], 18 + 25 * Math.abs(Math.sin(j * 1.91)), e);
    c.save();
    c.translate(
      mix(xx, tx, e),
      mix(yy, ty, e) - Math.sin(t * Math.PI) * (j % 2 ? 18 : -18),
    );
    c.rotate(Math.sin(t * Math.PI) * (j - 5.5) * 0.017);
    c.globalAlpha = Math.sin(t * Math.PI);
    if (t < 0.55 && im.complete) {
      c.drawImage(
        im,
        (j * im.naturalWidth) / n,
        0,
        im.naturalWidth / n,
        im.naturalHeight,
        0,
        -height / 2,
        mix(w, to[2] / n - 3, e),
        height,
      );
    } else {
      c.fillStyle = color;
      c.beginPath();
      c.roundRect(0, -height / 2, mix(w, to[2] / n - 3, e), height, 3);
      c.fill();
    }
    c.restore();
  }
}
