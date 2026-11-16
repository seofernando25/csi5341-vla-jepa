// Deterministic keyframe sampling. Values are numbers or flat objects of
// numbers; each segment interpolates with smootherstep so motion is eased
// without any per-scene bookkeeping.

export function clamp01(x) {
  return Math.min(1, Math.max(0, x));
}

export function smootherstep(x) {
  x = clamp01(x);
  return x * x * x * (x * (x * 6 - 15) + 10);
}

function lerpValue(a, b, u) {
  if (typeof a === "number") return a + (b - a) * u;
  const o = {};
  for (const k in a) o[k] = a[k] + (b[k] - a[k]) * u;
  return o;
}

// keys: [{ t, v }] sorted by t. Flat extrapolation at both ends.
export function sampleKeys(keys, t) {
  if (t <= keys[0].t) return keys[0].v;
  const last = keys[keys.length - 1];
  if (t >= last.t) return last.v;
  let i = 0;
  while (i < keys.length - 2 && keys[i + 1].t < t) i++;
  const a = keys[i];
  const b = keys[i + 1];
  const u = smootherstep((t - a.t) / (b.t - a.t));
  return lerpValue(a.v, b.v, u);
}

export function smoothstepRange(t, t0, t1) {
  return smootherstep((t - t0) / (t1 - t0));
}
