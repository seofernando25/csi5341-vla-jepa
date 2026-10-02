/* Action chunks are arrays of controls, never a physical path through a table. */
const CONTROL_EXAMPLE = [
  [-0.58, -0.42, -0.2, 0.03, 0.3, 0.5, 0.46, 0.37],
  [0.3, 0.27, 0.1, -0.14, -0.33, -0.29, -0.1, 0.02],
  [-0.68, -0.67, -0.58, 0.52, 0.64, 0.64, 0.54, 0.5],
];
const CONTROL_NOISE = [
  [0.61, -0.55, 0.42, -0.64, 0.55, -0.24, 0.72, -0.43],
  [-0.48, 0.58, -0.64, 0.55, -0.05, 0.65, -0.43, 0.59],
  [0.56, -0.34, 0.69, -0.52, 0.34, -0.58, 0.73, -0.24],
];
function controlValues(t, wobble = 0) {
  return CONTROL_NOISE.map((row, j) =>
    row.map(
      (v, k) =>
        mix(v, CONTROL_EXAMPLE[j][k], t) +
        wobble * Math.sin(k * 2.3 + j + t * 18),
    ),
  );
}
function controlPanel(c, x, y, w, h, values, title, color = ink) {
  c.save();
  c.fillStyle = "#f5f8fa";
  c.beginPath();
  c.roundRect(x, y, w, h, 12);
  c.fill();
  label(c, title, x + 12, y + 23, 13, color, true);
  const colors = [blue, orange, "#517e79"],
    names = ["Move", "Rotate", "Grip"];
  for (let j = 0; j < 3; j++) {
    const yy = y + 61 + j * 48;
    label(c, names[j], x + 12, yy + 3, 10, colors[j], true);
    segment(c, x + 52, yy, x + w - 13, yy, "#dce6ed", 0.7);
    c.strokeStyle = colors[j];
    c.lineWidth = 2;
    c.lineJoin = "round";
    c.lineCap = "round";
    c.beginPath();
    values[j].forEach((v, k) => {
      const xx = x + 53 + ((w - 69) * k) / 7,
        py = yy - v * 16;
      if (k) c.lineTo(xx, py);
      else c.moveTo(xx, py);
    });
    c.stroke();
    values[j].forEach((v, k) =>
      dotC(c, x + 53 + ((w - 69) * k) / 7, yy - v * 16, 1.8, colors[j]),
    );
  }
  label(c, "Earlier commands", x + 53, y + h - 12, 8.5, muted);
  label(c, "Later", x + w - 13, y + h - 12, 8.5, muted, false, "right");
  c.restore();
}
function flowMotion(c) {
  const sec = sceneSeconds(),
    training = index === 9;
  headerC(
    c,
    "Method · Flow matching · " +
      (training ? "Learn the update" : "Generate the controls"),
    training
      ? "Learn how to update a sequence of controls"
      : "Turn a random sequence into an action chunk",
    "Paper Eqs. 7–8 · Illustrative controls · Mixing time is not physical robot time",
  );
  if (training) {
    const phaseA = beat(sec, 10, 0.4),
      phaseB = beat(sec, 20, 0.4),
      match = beat(sec, 39, 6);
    controlPanel(
      c,
      664,
      169,
      244,
      221,
      CONTROL_EXAMPLE,
      "Demonstrated sequence",
      blue,
    );
    emerge(
      c,
      sec,
      10,
      () =>
        controlPanel(
          c,
          52,
          169,
          244,
          221,
          CONTROL_NOISE,
          "Random sequence",
          muted,
        ),
      8,
    );
    emerge(
      c,
      sec,
      20,
      () =>
        controlPanel(
          c,
          358,
          169,
          244,
          221,
          controlValues(0.48),
          "A training mixture",
          orange,
        ),
      8,
    );
    if (phaseB > 0) {
      link(c, 306, 263, 348, 263, gray, phaseB);
      link(c, 654, 263, 612, 263, gray, phaseB);
    }
    if (sec >= 34) {
      c.save();
      c.globalAlpha = beat(sec, 34, 0.45);
      const px = 412,
        py = 418,
        dy = mix(15, -13, match);
      segment(c, px, py, px + 78, py - 13, orange, 2.2, true);
      segment(
        c,
        px,
        py + 24,
        px + mix(35, 78, match),
        py + 24 + dy,
        blue,
        2.2,
        true,
      );
      pill(c, "Target update", 560, 409, orange);
      pill(c, "Model estimate", 563, 446, blue);
      c.restore();
    }
    const steps = [
      ["One sample = a whole control sequence", 0],
      ["Sample random values of the same shape", 10],
      ["Mix noise and demonstration at a chosen time", 20],
      ["Predict the update; match its training target", 34],
    ];
    const j = Math.max(
      0,
      steps.findLastIndex((x) => sec >= x[1]),
    );
    reelCardText(
      c,
      steps[j][0],
      52,
      145,
      18,
      ink,
      true,
      clamp((sec - steps[j][1]) / TYPE_MOTION.reelSeconds),
      j ? steps[j - 1][0] : "",
    );
    label(
      c,
      "Conditioning: observation + instruction → policy tokens",
      52,
      480,
      18,
      blue,
      true,
    );
  } else {
    const u = beat(sec, 4, 23),
      t = smoothOut(u);
    controlPanel(
      c,
      52,
      169,
      244,
      221,
      CONTROL_NOISE,
      "Fresh random sequence",
      muted,
    );
    controlPanel(
      c,
      358,
      169,
      244,
      221,
      controlValues(t, 0.1 * Math.sin(Math.PI * u)),
      "Current generated sequence",
      blue,
    );
    controlPanel(
      c,
      664,
      169,
      244,
      221,
      controlValues(1),
      "Illustrative action chunk",
      blue,
    );
    link(c, 306, 263, 348, 263, blue, beat(sec, 2, 0.4));
    link(c, 612, 263, 654, 263, blue, beat(sec, 27, 0.4));
    for (let k = 1; k <= 4; k++) {
      const x = 396 + (k - 1) * 57,
        on = beat(sec, 4 + (k - 1) * 5.7, 0.35);
      c.save();
      c.globalAlpha = 0.2 + 0.8 * on;
      dotC(c, x, 422, 13, blue);
      label(c, String(k), x, 426, 12, "white", true, "center");
      c.restore();
      if (k < 4) segment(c, x + 17, 422, x + 40, 422, "#d4e6f0", 1.2);
    }
    const times = [0, 4, 9.75, 15.5, 27];
    const texts = [
      "Start with random controls",
      "Apply the first learned update",
      "Ask again at the new mixture",
      "Repeat the learned updates",
      "Execute the generated controls",
    ];
    const manual = presenter.manual && motion.exportTime === null;
    const j = manual
      ? presenter.cursor
      : Math.max(
          0,
          times.findLastIndex((t) => sec >= t),
        );
    const reveal = manual
      ? presenter.run?.direction < 0
        ? 1 - presenter.progress
        : presenter.progress
      : clamp((sec - times[j]) / TYPE_MOTION.reelSeconds);
    reelCardText(
      c,
      texts[j],
      52,
      145,
      18,
      ink,
      true,
      reveal,
      j ? texts[j - 1] : "",
    );
    label(
      c,
      "The model sees the task and generation time at every update.",
      52,
      480,
      19,
      blue,
      true,
    );
  }
}
