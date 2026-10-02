const fs = require("node:fs"),
  vm = require("node:vm"),
  assert = require("node:assert/strict");
const context = vm.createContext({
  Image: function () {},
  clamp: (x) => Math.max(0, Math.min(1, x)),
});
vm.runInContext(
  fs.readFileSync(
    require("node:path").join(__dirname, "../../src/drawing.js"),
    "utf8",
  ),
  context,
);
const draw = vm.runInContext("({curve,segment})", context);
function canvas() {
  let tx = 0,
    ty = 0,
    a = 0,
    stack = [],
    path = [];
  const shapes = [];
  return {
    shapes,
    save() {
      stack.push([tx, ty, a]);
    },
    restore() {
      [tx, ty, a] = stack.pop();
    },
    translate(x, y) {
      tx += x;
      ty += y;
    },
    rotate(x) {
      a += x;
    },
    beginPath() {
      path = [];
    },
    moveTo(x, y) {
      path.push([
        tx + x * Math.cos(a) - y * Math.sin(a),
        ty + x * Math.sin(a) + y * Math.cos(a),
      ]);
    },
    lineTo(x, y) {
      this.moveTo(x, y);
    },
    closePath() {},
    stroke() {
      shapes.push({ type: "shaft", points: path });
    },
    fill() {
      shapes.push({ type: "tip", points: path });
    },
  };
}
const routes = [
  [
    [748, 354],
    [742, 362],
    [736, 370],
    [730, 378],
  ],
  [
    [0, 0],
    [0, 80],
    [90, 60],
    [100, 100],
  ],
  [
    [0, 0],
    [60, -80],
    [80, 80],
    [100, 0],
  ],
  [
    [0, 0],
    [30, 0],
    [100, 0],
    [100, 0],
  ],
  [
    [100, 100],
    [80, 30],
    [0, 50],
    [0, 0],
  ],
];
let count = 0;
for (const route of routes)
  for (const end of [0.2, 0.5, 0.9, 1])
    for (const width of [1.4, 2, 3]) {
      const c = canvas();
      draw.curve(c, route, "blue", width, end, true);
      const shaft = c.shapes.find((s) => s.type === "shaft"),
        tip = c.shapes.find((s) => s.type === "tip");
      if (!tip) continue;
      const base = [
          (tip.points[1][0] + tip.points[2][0]) / 2,
          (tip.points[1][1] + tip.points[2][1]) / 2,
        ],
        last = shaft.points.at(-1);
      assert(
        Math.hypot(base[0] - last[0], base[1] - last[1]) < 1e-6,
        "Arrow base must meet shaft exactly",
      );
      assert(tip.points.flat().every(Number.isFinite));
      count++;
    }
console.log(
  `${count} curved arrowhead joins checked, including short, reversed and zero-terminal-tangent paths.`,
);
