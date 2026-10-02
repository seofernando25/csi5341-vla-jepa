/* Editorial bookends and evidence scenes. No rasterized slide templates. */
function orbitRibbon(c, s, x, y, scale = 1) {
  if (window.featureReel?.ready) {
    c.drawImage(
      window.featureReel.render(s),
      x - 175 * scale,
      y - 108 * scale,
      350 * scale,
      216 * scale,
    );
    return;
  }
  c.save();
  c.translate(x, y);
  c.scale(scale, scale);
  for (let j = 0; j < 40; j++) {
    let t = j / 39,
      theta = t * Math.PI * 1.72 + Math.min(s, 8) * 0.11,
      xx = Math.cos(theta) * (90 + 30 * t),
      yy = Math.sin(theta) * (43 + 18 * t),
      d = (Math.sin(theta) + 1) / 2;
    c.save();
    c.globalAlpha = 0.25 + 0.7 * d;
    c.fillStyle = j % 4 === 0 ? orange : blue;
    c.translate(xx, yy);
    c.rotate(-0.16 + Math.sin(theta) * 0.3);
    c.beginPath();
    c.roundRect(-4, -(10 + d * 17), 8, 20 + d * 34, 3);
    c.fill();
    c.restore();
  }
  c.restore();
}
function editorialCover(c) {
  const s = sceneSeconds();
  label(c, "NOAH SPRENGER  /  FERNANDO NOGUEIRA", 52, 51, 12, muted);
  label(c, "CSI 5341 · SUN ET AL. 2026", 52, 77, 11, muted);
  emerge(c, s, 0, () => label(c, "VLA–", 52, 212, 78, ink, true));
  emerge(c, s, 0.15, () => label(c, "JEPA", 52, 297, 78, ink, true));
  typewriterText(
    c,
    "Can watching",
    56,
    373,
    30,
    ink,
    false,
    clamp((s - 0.65) / 0.65),
  );
  typewriterText(
    c,
    "become doing?",
    56,
    414,
    30,
    blue,
    true,
    clamp((s - 1.3) / 0.7),
  );
  const u = beat(s, 0.5, 1.2),
    turn = beat(s, 8, 1.2),
    px = mix(508, 590, turn),
    py = mix(142, 112, turn),
    pw = mix(352, 255, turn),
    ph = (pw * 169) / 373;
  c.save();
  c.translate(0, 24 * (1 - u));
  c.beginPath();
  c.roundRect(px, py, pw, ph, 12);
  c.clip();
  if (robotClip.readyState >= 2) c.drawImage(robotClip, px, py, pw, ph);
  else photo(c, robotPoster, px, py, pw, ph);
  c.restore();
  c.save();
  c.globalAlpha = 1 - turn;
  label(c, "A real robot demonstration", 508, 370, 15, muted);
  c.restore();
  frameToFeatures(
    c,
    robotPoster,
    [508, 142, 352, 159],
    [687, 330, 230],
    beat(s, 7, 1.8),
  );
  c.save();
  c.globalAlpha = turn;
  orbitRibbon(c, s, 690, 330, 1.05);
  label(c, "Transition → state → control", 690, 441, 17, blue, true, "center");
  c.restore();
  label(
    c,
    "Enhancing vision–language–action models with latent world prediction",
    52,
    504,
    13,
    muted,
  );
}
function editorialClosing(c) {
  let s = sceneSeconds();
  label(c, "THE TAKEAWAY", 52, 51, 13, blue, true);
  stampText(
    c,
    "Predict states.",
    52,
    192,
    52,
    ink,
    true,
    clamp(s / TYPE_MOTION.stampSeconds),
  );
  stampText(
    c,
    "Learn control.",
    52,
    257,
    52,
    blue,
    true,
    clamp((s - 0.16) / TYPE_MOTION.stampSeconds),
  );
  orbitRibbon(c, s, 768, 231, 1.06);
  emerge(c, s, 1.4, () => {
    label(c, "Strongest evidence", 52, 351, 15, muted);
    label(c, "Robustness under perturbations", 52, 389, 28, ink, true);
  });
  emerge(c, s, 5, () => {
    segment(c, 52, 422, 908, 422, "#dbe6ee", 1);
    label(c, "Our open question: lower resource cost", 52, 463, 21, ink);
    label(c, "while keeping useful control?", 52, 494, 21, blue, true);
  });
  label(
    c,
    "Sun et al. · LIBERO-Plus and human-video ablation",
    52,
    529,
    10,
    muted,
  );
}
function jointMotion(c) {
  const s = sceneSeconds();
  headerC(
    c,
    "Method · Joint supervision",
    "Video teaches prediction. Robot data teaches control.",
    "Paper §3.3 · Human videos have no robot action labels · Robot demonstrations provide both objectives",
  );
  photo(c, humanExamples[1], 52, 165, 180, 102);
  photo(c, robotPoster, 52, 324, 180, 102);
  label(c, "Human video", 52, 293, 17, ink, true);
  label(c, "Robot demonstration", 52, 452, 17, ink, true);
  block(c, 434, 300, "Policy", "shared latent|tokens");
  const curves = [
    [
      [240, 213],
      [306, 213],
      [297, 280],
      [362, 280],
    ],
    [
      [240, 375],
      [306, 375],
      [297, 320],
      [362, 320],
    ],
    [
      [508, 283],
      [588, 283],
      [596, 221],
      [665, 221],
    ],
    [
      [508, 320],
      [588, 320],
      [596, 373],
      [665, 373],
    ],
  ];
  curves.forEach((q, j) => {
    curve(
      c,
      q,
      j === 3 ? orange : blue,
      2,
      beat(s, 0.3 + j * 0.12, 0.45),
      true,
    );
  });
  featureRibbon(c, 787, 215, 172, blue);
  label(c, "Future-state alignment", 787, 262, 20, blue, true, "center");
  actionGlyph(c, 707, 340, 160, 55);
  label(c, "Robot action objective", 787, 417, 20, orange, true, "center");
  const before = "Video → prediction. Robot data → control.";
  const after = "New robot? Adapt its control interface.";
  reelCardText(
    c,
    s < 24 ? before : after,
    52,
    478,
    17,
    ink,
    true,
    s < 24 ? 1 : clamp((s - 24) / TYPE_MOTION.reelSeconds),
    s < 24 ? "" : before,
  );
}
function quantMotion(c) {
  const s = sceneSeconds();
  headerC(
    c,
    "Our project · Proposal and work so far",
    "Can we lower the cost and keep useful control?",
    "Planned comparisons · Outcomes remain open · No project performance results announced",
  );
  const columns = [
    [
      "01",
      "Reduced precision",
      [
        "Compare against the same baseline",
        "Measure memory, latency and success",
      ],
      blue,
    ],
    [
      "02",
      "Smaller backbone",
      ["Explore SmolVLM integration", "Keep prediction and action components"],
      orange,
    ],
    [
      "03",
      "Work so far",
      [
        "Implementation and evaluation setup",
        "Validate before drawing conclusions",
      ],
      ink,
    ],
  ];
  columns.forEach(([n, title, lines, color], j) => {
    const x = 52 + j * 296;
    emerge(c, s, j * 0.12, () => {
      label(c, n, x, 175, 14, color, true);
      label(c, title, x, 220, 23, ink, true);
      segment(c, x, 246, x + 258, 246, color, 2);
      lines.forEach((line, k) => label(c, line, x, 286 + k * 34, 14, muted));
    });
  });
  emerge(c, s, 1.3, () => {
    pill(c, "Matched baseline → resource use + task success", 480, 405, blue);
    label(
      c,
      "The trade-off is a question to test, not an improvement to promise.",
      480,
      468,
      21,
      ink,
      true,
      "center",
    );
  });
}
function limitsMotion(c) {
  let s = sceneSeconds();
  headerC(
    c,
    "Critical analysis · Beyond headline averages",
    "Strong averages leave deployment questions open",
    "Paper Table 3, §4.4 and Appendix B · Our single-GPU comparison is proposed",
  );
  label(c, "SENSOR NOISE", 52, 172, 13, muted, true);
  let t = beat(s, 0.2, 0.7);
  label(c, (66.3 * t).toFixed(1) + "%", 52, 245, 49, orange, true);
  label(c, "VLA-JEPA", 52, 274, 15, ink, true);
  label(c, "79.0% · pi zero", 52, 314, 22, muted);
  label(c, "A robustness counterexample", 52, 348, 13, muted);
  label(c, "REAL-WORLD EVIDENCE", 369, 172, 13, muted, true);
  for (let j = 0; j < 10; j++) {
    let u = beat(s, 0.3 + j * 0.035, 0.4),
      x = 374 + (j % 5) * 35,
      y = 204 + Math.floor(j / 5) * 35;
    c.save();
    c.translate(0, 8 * (1 - u));
    c.fillStyle = "#e6f3fa";
    c.beginPath();
    c.roundRect(x, y, 25, 25, 5);
    c.fill();
    label(c, String(j + 1), x + 12.5, y + 17, 10, blue, false, "center");
    c.restore();
  }
  label(c, "10 trials per task", 369, 314, 22, ink, true);
  label(c, "Wrong-object selections", 369, 348, 13, muted);
  label(c, "TRAINING → DEPLOYMENT", 680, 172, 13, muted, true);
  for (let j = 0; j < 8; j++) {
    c.fillStyle = "#dfeaf1";
    c.beginPath();
    c.roundRect(686 + (j % 4) * 39, 201 + Math.floor(j / 4) * 32, 30, 23, 4);
    c.fill();
  }
  label(c, "8 A100 GPUs · paper training", 680, 314, 18, blue, true);
  label(c, "Proposed study: single-GPU efficiency", 680, 348, 13, muted);
  emerge(c, s, 1.3, () =>
    label(
      c,
      "Headline averages do not establish deployment reliability.",
      52,
      464,
      25,
      ink,
      true,
    ),
  );
  label(
    c,
    "No reported confidence intervals on these headline comparisons.",
    52,
    491,
    16,
    muted,
  );
}
function controlTrace(c, time) {
  const a = RECORDED_CONTROL.recordedCartesianPoseCommands,
    t = RECORDED_CONTROL.timeSeconds,
    x = 535,
    y = 434,
    w = 373,
    h = 54;
  label(
    c,
    "Recorded pose commands · x/y/z normalized per axis",
    x,
    y - 7,
    12,
    muted,
  );
  const colors = [blue, "#4b9b89", orange];
  for (let k = 0; k < 3; k++) {
    const values = a.map((row) => row[k]),
      lo = Math.min(...values),
      hi = Math.max(...values);
    c.strokeStyle = colors[k];
    c.lineWidth = 1.6;
    c.beginPath();
    for (let j = 0; j < a.length; j++) {
      if (t[j] > time) break;
      let xx = x + (t[j] / t.at(-1)) * w,
        yy = y + h - ((values[j] - lo) / (hi - lo || 1)) * h;
      if (j === 0) c.moveTo(xx, yy);
      else c.lineTo(xx, yy);
    }
    c.stroke();
  }
  segment(c, x, y + h + 4, x + w, y + h + 4, "#d9e4eb", 1);
}
function humanRobotScene(c) {
  const time = comparisonTime(),
    blend = beat(time, 6.1, 0.7);
  headerC(
    c,
    "The supervision gap · Watching is not doing",
    time < 6.1
      ? "What information can a demonstration provide?"
      : "Video shows outcomes. Robot data supplies control.",
    "SSV2 validation #174198 · DROID AUTOLab, 27 Oct 2023 · Dataset examples, not VLA-JEPA rollouts",
  );
  label(c, "WATCH", 52, 145, 12, blue, true);
  label(c, "Human video", 52, 178, 26, ink, true);
  label(c, "ACT", 535, 145, 12, orange, true);
  label(c, "Robot demonstration", 535, 178, 26, ink, true);
  c.save();
  c.beginPath();
  const humanHeight = mix(189, 140, blend);
  c.roundRect(52, 206, 418, humanHeight, 12);
  c.clip();
  if (humanClip.readyState >= 2)
    c.drawImage(humanClip, 52, 206, 418, humanHeight);
  else photo(c, humanExamples[0], 52, 206, 418, humanHeight);
  c.restore();
  c.save();
  c.globalAlpha = blend;
  featureRibbon(c, 261, 375, 230, blue);
  c.restore();
  c.save();
  c.beginPath();
  c.roundRect(535, 206, 373, 169, 12);
  c.clip();
  if (robotClip.readyState >= 2) c.drawImage(robotClip, 535, 206, 373, 169);
  else if (robotPoster.complete) c.drawImage(robotPoster, 535, 206, 373, 169);
  c.restore();
  label(c, "“Put the jar into the box.”", 52, 433, 16, ink, true);
  label(
    c,
    "Video + description; no robot control labels",
    52,
    477,
    16,
    orange,
    true,
  );
  label(
    c,
    "“Place the green block inside the black bowl.”",
    535,
    397,
    13,
    ink,
    true,
  );
  controlTrace(
    c,
    presenter.manual && comparisonExportTime === null
      ? robotClip.currentTime
      : Math.min(time, 5.901),
  );
}

function actionGlyph(c, x, y, w, h) {
  const a = RECORDED_CONTROL.recordedCartesianPoseCommands;
  for (let k = 0; k < 3; k++) {
    const v = a.map((row) => row[k]),
      lo = Math.min(...v),
      hi = Math.max(...v);
    c.strokeStyle = [orange, "#4b9b89", blue][k];
    c.lineWidth = 1.8;
    c.beginPath();
    v.forEach((z, j) => {
      let xx = x + (j / (v.length - 1)) * w,
        yy = y + h - ((z - lo) / (hi - lo || 1)) * h;
      if (j) c.lineTo(xx, yy);
      else c.moveTo(xx, yy);
    });
    c.stroke();
  }
}
