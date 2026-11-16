# Project explainer video

A 240-second narrated explainer (`1280×720`, `30 fps`, 18 scenes) of the project:
what the policy is, how the evaluation works, and what the search found.
Built to the OhRats paper look (warm grid paper, hairlines, one brown accent,
Inter + JetBrains Mono); all motion is drawn on a 2D canvas, so no GPU is used.

## What it covers

The question (how much of a released checkpoint must you keep), the policy
(VLA-JEPA on LIBERO), what success counts as, one rollout, the memory rule
(largest of tensor walk, live CUDA, peak CUDA), the three phases
(smoke / development / reserved), the paired five hundred, the fixed evaluator,
Dream-RSI, the caught cycle-3 exploit, the development and reserved numbers,
and two footnotes worth keeping (held-out loss is not success; smoke is not skill).

Numbers quoted on screen come from recorded runs: the uncompressed reference
(`475/500` reserved, `82/100` development), the two selected cycle winners
(`478/500` reserved, `84/100` development, one slower by ~2.7×), and the
development retention percentages (`−87 %` cycle 1, `−84 %` cycle 2).
The program-v2 search is still running; the reserved read for its winners
has not happened. The narration says so.

## Pipeline

- `script.txt` — narration, one sentence per line. `*…*` emphasis,
  `{shown|spoken}` for literal display vs. spoken form, `@cue:token` for
  visual triggers (`tools/script.mjs` parses it).
- `tools/tts.mjs` — per-sentence TTS via OpenRouter
  (`google/gemini-3.8-flash-tts`, voice `Charon`, natural documentary pace,
  ~165 wpm). PCM is returned; silence-trimmed, 48 kHz mono.
- `tools/align.py` — word-level timestamps with faster-whisper (`small.en`),
  mapped onto script tokens; writes `build/timeline.json`.
- `tools/audio.py` — 48 kHz stereo master: narration, ducked synth pulse,
  whooshes on scene wipes, slam impacts on cues.
- `src/render.mjs` — direct Skia renderer (no browser, no screenshots): draws every
  frame with the same scene painters and captions, streams raw RGBA into ffmpeg,
  and muxes the audio master into `dist/project-explainer-720p30.mp4`.
- `src/world3d/` + `tools/render3d.mjs` — the cinematic 3D scenes (bench, bowl, plate,
  6-axis arm, warm workshop): a deterministic Three.js world rendered to RGBA PNG
  frames at 30 fps (1280×720) in the scene windows (`arm` 10.6–21.7 s, `policy`
  22.3–33.7 s, `success` 34.3–43.4 s, `episode` 44.0–61.4 s). The driver relaunches
  chromium every 128 frames; captions and 2D overlays are composited on top by
  `src/render.mjs`. `build/3d/` is not committed.

## Build

```sh
cd web/project-explainer
cp .env.example .env   # set OPENROUTER_API_KEY (gitignored)
bun install           # fonts + skia-canvas
python3 -m venv .venv-align && ./.venv-align/bin/pip install faster-whisper
bun run build         # writes dist/project-explainer-720p30.mp4
```

The MP4 is tracked so the video can be watched without a rebuild;
`bun run build` needs network access for TTS.

## Captions and transitions

- Word highlighting is a karaoke reveal: faded ink base text, with a brown pill
  clipped left-to-right over the active word (emphasis words get a deeper
  brown pill); the active word also springs slightly at word start.
- Scene transitions are a diagonal three-slab whip (brown → ink) sweeping left
  to right in 0.55 s, aligned to the whoosh in the audio master.
- Chrome (scene eyebrow, brand line, hairline progress) slides/fades in on
  every scene boundary.
