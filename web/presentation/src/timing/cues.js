/* Named active windows in the original export timeline. Idle gaps are never played.
   Ordinary transitions: 600 ms. Media retains its real timing. Continuous teaching
   demonstrations have an explicit longer duration, never inferred from audio. */
const cue = (label, start, end, duration = 600) => ({
  label,
  start,
  end,
  duration,
});
const SCENE_CUES = {
  0: [
    cue("Opening question", 0, 2.1, 1100),
    cue("Video becomes a representation", 7, 10, 1000),
  ],
  2: [
    cue("Watch the demonstrations", 0, 5.9, 800),
    cue("Compare the available supervision", 6.09, 6.9),
  ],
  3: [
    cue("Cost of control labels", 0, 0.8),
    cue("Why use video?", 8.99, 9.6),
    cue("LAPA: pixel reconstruction", 15.99, 16.7),
    cue("UniVLA: feature reconstruction", 30.99, 31.7),
    cue("A target is not a pipeline", 35.99, 36.6),
    cue("VLA-JEPA: predictive alignment", 44.99, 45.7),
    cue("What does the policy see?", 54.99, 55.7),
  ],
  5: [
    cue("Represent the task", 0, 0.7),
    cue("Encode both inputs", 2.99, 4),
    cue("Produce latent tokens", 6.99, 7.65, 650),
  ],
  6: [
    cue("Condition the predictor", 0, 1.1),
    cue("Predict the next state", 8.99, 11, 800),
  ],
  7: [
    cue("Separate prediction and target", 0, 1.1),
    cue("Learn from the mismatch", 4.99, 17.8, 1200),
  ],
  8: [
    cue("Inputs above the boundary", 0, 1.1),
    cue("Future is a target only", 1.49, 2.4),
  ],
  9: [
    cue("A demonstration is a control sequence", 0, 0.8),
    cue("Sample random controls", 9.99, 11),
    cue("Create a training mixture", 19.99, 21),
    cue("Define the target update", 33.99, 34.65),
    cue("Match the update", 38.99, 46, 1100),
  ],
  10: [
    cue("Start with random controls", 0, 2.8),
    cue("First learned update", 3.99, 4.8),
    cue("Second learned update", 4.8, 10.5),
    cue("Third learned update", 10.5, 16.3),
    cue("Generate the action chunk", 16.3, 27.8, 850),
  ],
  11: [
    cue("Two sources, two objectives", 0, 1.5, 750),
    cue("Adapt the robot interface", 23.99, 24.7),
  ],
  12: [cue("Standard LIBERO comparison", 0, 1.5, 750)],
  13: [cue("LIBERO-Plus comparison", 0, 1.5, 750)],
  14: [cue("Human-video ablation", 0, 1.5, 750)],
  15: [cue("A benchmark counterexample", 0, 1.5, 750)],
  17: [
    cue("Deployment limitations", 0, 0.95),
    cue("Interpret the evidence", 1.29, 1.9),
  ],
  18: [
    cue("Predict states; learn control", 0, 0.8),
    cue("Strongest evidence", 1.39, 2),
    cue("Limits of the evidence", 4.99, 5.8),
  ],
};
const PRESENTATION_BEATS = Object.fromEntries(
  Object.entries(SCENE_CUES).map(([i, specs]) => [i, specs.map((s) => s.end)]),
);
