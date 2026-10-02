VLA-JEPA · immersive browser presentation

Serve this folder with a local static server; see README.md for repository commands.
18 narrated chapters. Main narration: about 9:40.
Noah covers 1–8; Fernando covers 9–18, approximately equal time.

Click / Right / Space: next animation beat. Right click / Left: previous beat.
Replay beat repeats the current cue; Chapters jumps sections. End is authoring-only.
N: notes. Hover near the bottom: chapter rail and playback controls.
Narration assets are retained for exports. The main browser has no narration button.
Chapter rail navigates the whole talk. Each beat holds until the next input.

Organization
  index.html       minimal page, local script dependencies
  styles.css       immersive stage, hover controls and chapter rail
  src/content.js   chapter content, notes, timing, sources and speaker allocation
  src/player.js    chapter selection and notes
  src/controls.js  single keyboard, mouse and button bindings
  src/timing/cues.js named windows and explicit transition durations
  src/timing/clock.js pure playback state machine
  src/beat-controller.js browser clock adapter
  src/media.js     deterministic footage playback
  src/timeline.js  chapter rail view
  src/drawing.js   canvas primitives, verified media and the human/robot scene
  src/motion-type.js  reel-card, typewriter and squash/settle text effects
  src/renderer.js  one retained canvas, display-refresh rendering
  src/scenes/method.js  shared semantic objects and interpolated chapter poses
  src/scenes/flow.js    shared training/inference action-space visual
  src/scenes/results.js synchronized paired benchmark motion
  src/scenes/bookends.js opening, real control traces, joint training and evidence
  src/choreography.js    explicit visual beat lengths and transformation primitives
  src/recorded-control.js verified DROID pose commands
  assets/three/feature-reel.js original procedural 3D feature sculpture
  src/scenes/related.js related-work comparison
  src/scenes/benchmarks-data.js benchmark values
  src/scenes/dispatch.js scene routing
  data.json        human-readable content mirror
  assets/          local footage, source frames, fonts, math and stock voice audio

Shared method objects retain continuity between chapters 4–8. The future target dissolves in place to avoid crossing the diagram.
Feature ribbons and action-space curves are schematic, not model measurements.
The target is a future-state embedding. Transition tokens condition prediction and control.
A separate flow-matching action head generates controls; no future-image decoder is required.
UniVLA already operates in DINO feature space; the comparison concerns pipelines and inputs.

Human example: SSV2 validation sample 174198, putting jar into box.
Verified against validation metadata; not identified as a VLA-JEPA training sample.
Robot example: DROID AUTOLab+0d4edc83+2023-10-27-19h-52m-50s, camera 24400334.
Real timing restored from control timestamps. This is dataset footage, not VLA-JEPA rollout.
Footage plays once then holds; native capture frames are duplicated, not invented by interpolation.
The exported movie is constant 60 fps; live browser speed depends on the display and hardware.

Sources
https://arxiv.org/abs/2602.10098v1
https://arxiv.org/abs/2505.06111
https://latentactionpretraining.github.io/
https://www.qualcomm.com/developer/software/something-something-v-2-dataset
https://huggingface.co/datasets/morpheushoc/something-something-v2
https://droid-dataset.github.io/visualizer/

Puck and Charon are synthetic stock voices generated via the authorized OpenRouter account.
The project section is proposal-only: planned quantization and smaller-backbone comparisons, with no announced project performance results.
The original procedural Three.js feature sculpture is used on the bookends.
It is an illustrative motif, not a measured embedding.
Unused slide images and legacy Panda assets are archived outside this deliverable.
PDF companion slides show final states. Rehearsal guide contains dialogue and delivery cues.
