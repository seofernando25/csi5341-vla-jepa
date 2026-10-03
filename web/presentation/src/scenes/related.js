function relatedMotion(c) {
  const sec = sceneSeconds();
  headerC(
    c,
    "Related work · Why learn from video?",
    "How earlier methods learn from video",
    "OpenVLA · LAPA §3 · UniVLA §III · VLA-JEPA §3",
  );
  const rows = [
    [
      "Robot-labelled VLA",
      "Recorded controls",
      "Image + instruction → executable actions",
      ink,
      0,
    ],
    [
      "LAPA",
      "Future pixels",
      "Learn latent actions from video; adapt the policy",
      muted,
      16,
    ],
    [
      "UniVLA",
      "DINO features",
      "Feature reconstruction already existed",
      blue,
      31,
    ],
    [
      "VLA-JEPA",
      "Future state features",
      "Train policy tokens through predictive alignment",
      orange,
      45,
    ],
  ];
  rows.forEach(([name, target, detail, color, start], j) => {
    const y = 176 + j * 77;
    emerge(
      c,
      sec,
      start,
      () => {
        stampText(
          c,
          name,
          52,
          y,
          21,
          color,
          true,
          clamp((sec - start) / TYPE_MOTION.stampSeconds),
        );
        pill(c, target, 369, y - 4, color);
        kineticLine(c, detail, 505, y, 14, ink, false, 1);
        segment(c, 52, y + 29, 908, y + 29, "#e5edf2", 1);
      },
      8,
    );
  });
}
