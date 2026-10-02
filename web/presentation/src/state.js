const stage = document.getElementById("stage"),
  audio = document.getElementById("audio");
let index = 0,
  p = 0,
  animation = false;
const ink = "#142033",
  blue = "#0284c7",
  gray = "#94a3b8",
  muted = "#64748b",
  orange = "#bf531c",
  light = "#e8f5fb";
const clamp = (x) => Math.max(0, Math.min(1, x));
const ease = (x) => {
  x = clamp(x);
  return x * x * (3 - 2 * x);
};
const phase = (a, b) => ease((p - a) / (b - a));
