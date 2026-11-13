export const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
export const lerp = (a, b, k) => a + (b - a) * k;
export const smooth = (x) => {
  x = clamp(x);
  return x * x * (3 - 2 * x);
};
export const outCubic = (x) => 1 - (1 - clamp(x)) ** 3;
export const inCubic = (x) => clamp(x) ** 3;
export const outExpo = (x) => (x >= 1 ? 1 : 1 - 2 ** (-10 * clamp(x)));
export const outBack = (x, s = 1.70158) => {
  x = clamp(x) - 1;
  return 1 + (s + 1) * x ** 3 + s * x ** 2;
};
export const appear = (t, t0, d = 0.35) => clamp((t - t0) / d);

// Piecewise-linear keyframes: kf(t, [[t0, v0], [t1, v1], ...]) with smoothstep between keys.
export function kf(t, keys) {
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    if (t < keys[i][0]) {
      const [t0, v0] = keys[i - 1];
      const [t1, v1] = keys[i];
      return lerp(v0, v1, smooth((t - t0) / (t1 - t0)));
    }
  }
  return keys[keys.length - 1][1];
}

export function mixHex(a, b, k) {
  const ch = (c, s) => (c >> s) & 255;
  const m = (s) => Math.round(lerp(ch(a, s), ch(b, s), k));
  return (m(16) << 16) | (m(8) << 8) | m(0);
}
export const hexCss = (h, a = 1) => `rgba(${(h >> 16) & 255},${(h >> 8) & 255},${h & 255},${a})`;

export const COL = {
  bg: 0x06070d,
  cyan: 0x3df0ff,
  magenta: 0xff3d9a,
  yellow: 0xffe14d,
  green: 0x5dff8f,
  red: 0xff4d4d,
  gray: 0x7a8199,
  white: 0xffffff,
};
