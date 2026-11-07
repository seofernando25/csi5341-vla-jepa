# Paper presentation: VLA-JEPA

Static browser presentation of the VLA-JEPA paper for the October paper-presentation milestone:
18 narrated chapters, about 9:40. Noah covers chapters 1–8, Fernando 9–18. It presents the paper
and the project proposal; it reports no project results.

[Slides PDF](documents/slides.pdf) · [Presenter guide](documents/presenter-guide.pdf) · [Second-screen setup](PRESENTER.md)

## Run

```sh
python3 -m http.server 8000 --bind 127.0.0.1   # from the repository root
```

Open http://localhost:8000/web/presentation/. Serve over HTTP; opening `index.html` directly
breaks module loading. No packages, accounts or keys are needed.

| Input | Action |
|---|---|
| Click, Right, Space | Play the next beat |
| Right-click, Left | Previous beat |
| N | Notes |
| Hover at bottom | Chapter rail and controls |

For notes on a second computer, run `python3 web/presentation/tools/relay.py` and open its Slides
and Notes URLs ([details](PRESENTER.md)).

## Source layout

| Path | Contents |
|---|---|
| `src/content.js`, `data.json` | Wording, notes, timing, sources; keep both in sync |
| `src/timing/` | Cue durations (`cues.js`) and the pure playback state machine (`clock.js`) |
| `src/beat-controller.js`, `src/controls.js`, `src/player.js` | Browser adapter, input bindings, chapter selection |
| `src/scenes/`, `src/choreography.js`, `src/drawing.js`, `src/motion-type.js` | Scenes, motion and drawing primitives |
| `assets/` | Fonts, math images, footage, Three.js, p5, narration; hashes in `assets/manifest.json` |
| `tools/` | Authoring server, rendering and movie assembly ([rebuild](tools/README.txt)) |
| `documents/` | Slide PDF, presenter guide and export checks |

Checks: `node web/presentation/tools/tests/manual-cues.cjs` and
`node web/presentation/tools/tests/arrowheads.cjs`. For a visual transition audit, run
`python3 web/presentation/tools/serve.py`, open it with `?authoring=1`, then Notes → Audit every
transition. Output goes to ignored `build/rendered/`.

The spoken script in `src/content.js` is newer than the recorded narration and the presenter PDF;
regenerate narration before exporting audio or video.

## Provenance

- Footage is illustrative dataset material, not VLA-JEPA rollouts: SSV2 validation sample 174198
  (putting a jar into a box) and DROID episode AUTOLab+0d4edc83+2023-10-27-19h-52m-50s, camera
  24400334, with real timing restored from control timestamps. Frames are duplicated for 60 fps,
  not interpolated.
- Feature ribbons, action-space curves and the 3D feature sculpture are schematic, not measured
  embeddings or policy outputs.
- Narration uses the synthetic stock voices Puck and Charon.
- Sources: [VLA-JEPA](https://arxiv.org/abs/2602.10098v1), [UniVLA](https://arxiv.org/abs/2505.06111),
  [LAPA](https://latentactionpretraining.github.io/), [SSV2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset),
  [DROID](https://droid-dataset.github.io/visualizer/).
- Licenses: Three.js MIT, p5.js LGPL-2.1 and the font license are in `assets/`. Dataset clips
  keep their source rights; check the source terms before redistribution.
- The two clips are tracked as an explicit exception to the no-video rule.
