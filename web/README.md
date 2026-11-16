# Web deliverables

Two standalone subprojects under `web/`.

## `web/presentation/` — Paper presentation

Static browser presentation of the VLA-JEPA paper and project proposal (October milestone, 18 narrated chapters). Reports no project results.

- **Serve:** `python3 -m http.server 8000 --bind 127.0.0.1` from the repository root, then open `http://127.0.0.1:8000/web/presentation/`.
- **Author & assemble movie:** `python3 web/presentation/tools/serve.py`, open `http://127.0.0.1:8765/?authoring=1`, export via Notes → *Export revised presentation*, then run `python3 web/presentation/tools/assemble.py` (`ffmpeg` required; outputs under ignored `build/`).
- **Checks:** `node web/presentation/tools/tests/manual-cues.cjs` and `node web/presentation/tools/tests/arrowheads.cjs`.
- **Tracked media:** `assets/narration-*.mp3`, `assets/droid-demonstration.mp4`, `assets/human-demonstration.mp4`, `documents/slides.pdf`, and `documents/presenter-guide.pdf` stay tracked because the static viewer uses them directly and they cannot be regenerated offline from tracked sources.

Details: [presentation/README.md](presentation/README.md).

## `web/project-explainer/` — Project explainer video

Narrated 4-minute explainer of the project and how the evaluation works (18 scenes, `1280×720`, `30 fps`). OhRats paper look (warm grid, hairlines, brown accent, Inter + JetBrains Mono); 2D canvas, no GPU. Numbers come from recorded runs only.

- **Build:** `cd web/project-explainer && bun run build` (needs network for OpenRouter TTS; `dist/` is gitignored).
- **Style:** soft whip transitions aligned to the audio, karaoke word highlighting, voice `Charon` via Gemini TTS.

Details: [project-explainer/README.md](project-explainer/README.md).
