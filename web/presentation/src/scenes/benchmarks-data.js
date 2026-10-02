const charts = {
  12: {
    title: "Standard LIBERO: a near tie",
    labels: ["OpenVLA-OFT", "VLA-JEPA"],
    values: [97.1, 97.2],
    claim: "+0.1 percentage points",
    foot: "Paper Table 1 · Success rate (%) · No reported confidence intervals",
  },
  13: {
    title: "LIBERO-Plus: stronger under perturbations",
    labels: ["OpenVLA-OFT", "VLA-JEPA"],
    values: [69.6, 79.5],
    claim: "+9.9 percentage points",
    foot: "Paper Table 1 · Camera, lighting and layout shifts · Success rate (%)",
  },
  14: {
    title: "Human video improves LIBERO-Plus robustness",
    labels: ["No human video", "With human video"],
    values: [62.9, 79.5],
    claim: "+16.6 percentage points",
    foot: "Paper Table 2 · Within-method ablation · Success rate (%)",
  },
  15: {
    title: "Human video does not improve every task",
    labels: ["No human video", "With human video"],
    values: [78.4, 65.2],
    claim: "−13.2 percentage points · SimplerEnv Google",
    foot: "Paper Table 2 · Within-method ablation · Success rate (%)",
  },
};
