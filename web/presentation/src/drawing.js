let sketch,
  canvasCtx,
  framesSeen = 0,
  frameTimes = [],
  lastPaint = 0;
function label(
  c,
  t,
  x,
  y,
  size = 24,
  color = ink,
  bold = false,
  align = "left",
) {
  c.fillStyle = color;
  c.font = `${bold ? 700 : 500} ${size}px "Plus Jakarta Sans",Arial,sans-serif`;
  c.textAlign = align;
  c.fillText(t, x, y);
}
function dotC(c, x, y, r, color) {
  c.fillStyle = color;
  c.beginPath();
  c.arc(x, y, r, 0, Math.PI * 2);
  c.fill();
}
// Trim the shaft at the arrowhead base: rounded caps never protrude past the tip.
function arrowTip(c, x, y, angle, color, width = 2) {
  const length = 8,
    half = 3.5;
  c.save();
  c.translate(x, y);
  c.rotate(angle);
  c.fillStyle = color;
  c.beginPath();
  c.moveTo(0, 0);
  c.lineTo(-length, -half);
  c.lineTo(-length, half);
  c.closePath();
  c.fill();
  c.restore();
}
function arrowPath(c, points, color, width, head, angle) {
  if (points.length < 2) return;
  let lengths = [0];
  for (let j = 1; j < points.length; j++)
    lengths.push(
      lengths[j - 1] +
        Math.hypot(
          points[j][0] - points[j - 1][0],
          points[j][1] - points[j - 1][1],
        ),
    );
  const total = lengths.at(-1),
    cut = head ? Math.max(0, total - 7) : total;
  c.save();
  c.strokeStyle = color;
  c.lineWidth = width;
  c.lineCap = "round";
  c.lineJoin = "round";
  c.beginPath();
  c.moveTo(...points[0]);
  for (let j = 1; j < points.length; j++) {
    if (lengths[j] <= cut) c.lineTo(...points[j]);
    else {
      const u = (cut - lengths[j - 1]) / (lengths[j] - lengths[j - 1] || 1);
      c.lineTo(
        points[j - 1][0] + u * (points[j][0] - points[j - 1][0]),
        points[j - 1][1] + u * (points[j][1] - points[j - 1][1]),
      );
      break;
    }
  }
  c.stroke();
  if (head && total > 8) arrowTip(c, ...points.at(-1), angle, color, width);
  c.restore();
}
function segment(c, x, y, x2, y2, color, width = 2, head = false) {
  arrowPath(
    c,
    [
      [x, y],
      [x2, y2],
    ],
    color,
    width,
    head,
    Math.atan2(y2 - y, x2 - x),
  );
}
function kineticLine(c, text, x, y, size, color, bold, t = 1, oldText = "") {
  c.save();
  c.beginPath();
  c.rect(x - 3, y - size - 6, 905 - x, size + 14);
  c.clip();
  const write = (line, offset) => {
    let xx = x;
    c.font = `${bold ? 700 : 500} ${size}px "Plus Jakarta Sans"`;
    line.split(" ").forEach((word, j) => {
      const local = smoothOut((t - j * 0.015) / 0.7);
      label(c, word, xx, y + offset(local), size, color, bold);
      xx += c.measureText(word + " ").width;
    });
  };
  if (t < 1 && oldText) write(oldText, (u) => -u * (size + 15));
  write(text, (u) => (1 - u) * (size + 15));
  c.restore();
}
function headerC(c, k, title, foot) {
  const t =
    motion.exportTime === null && presenter.manual
      ? (motion.chapterProgress ?? 1)
      : clamp(sceneSeconds() / 0.38);
  label(c, k, 52, 43, 15, blue, true);
  kineticLine(c, title, 52, 94, 31, ink, true, t, motion.oldHeader || "");
  motion.header = title;
  const source =
    {
      3: "OpenVLA · LAPA · UniVLA",
      4: "Paper §3.2",
      5: "Paper §3.2",
      6: "Paper §3.2",
      7: "Paper §3.2",
      8: "Paper §3.2",
      9: "Paper Eqs. 7–8",
      10: "Paper Eqs. 7–8",
      11: "Paper §3.3",
      16: "Our proposal",
      17: "Paper §4.4",
    }[index] || foot.split(" · ")[0];
  label(c, source, 52, 520, 10, muted);
  label(
    c,
    `${String(sceneNumber(index)).padStart(2, "0")} / 18`,
    908,
    520,
    11,
    muted,
    false,
    "right",
  );
}
const humanExamples = [1, 2, 5].map((i) => {
  const im = new Image();
  im.src = [
    "assets/human-example-001.png",
    "assets/human-example-002.png",
    "assets/human-example-005.png",
  ][[1, 2, 5].indexOf(i)];
  return im;
});

function bez(t, pts) {
  const u = 1 - t;
  return [
    u * u * u * pts[0][0] +
      3 * u * u * t * pts[1][0] +
      3 * u * t * t * pts[2][0] +
      t * t * t * pts[3][0],
    u * u * u * pts[0][1] +
      3 * u * u * t * pts[1][1] +
      3 * u * t * t * pts[2][1] +
      t * t * t * pts[3][1],
  ];
}
function curve(c, pts, color, width = 2, end = 1, head = false) {
  end = clamp(end);
  if (end <= 0) return;
  const points = Array.from({ length: 81 }, (_, i) => bez((i / 80) * end, pts)),
    u = 1 - end,
    dx =
      3 * u * u * (pts[1][0] - pts[0][0]) +
      6 * u * end * (pts[2][0] - pts[1][0]) +
      3 * end * end * (pts[3][0] - pts[2][0]),
    dy =
      3 * u * u * (pts[1][1] - pts[0][1]) +
      6 * u * end * (pts[2][1] - pts[1][1]) +
      3 * end * end * (pts[3][1] - pts[2][1]);
  arrowPath(c, points, color, width, head, Math.atan2(dy, dx));
}
