# Web deliverables

Two standalone subprojects under `web/`.

## `web/presentation/` — Paper presentation

Static browser presentation of the VLA-JEPA paper and project proposal (October milestone, 18 narrated chapters). Reports no project results.

- **Serve:** `python3 -m http.server 8000 --bind 127.0.0.1` from the repository root, then open `http://127.0.0.1:8000/web/presentation/`.
- **Author & assemble movie:** `python3 web/presentation/tools/serve.py`, open `http://127.0.0.1:8765/?authoring=1`, export via Notes → *Export revised presentation*, then run `python3 web/presentation/tools/assemble.py` (`ffmpeg` required; outputs under ignored `build/`).
- **Checks:** `node web/presentation/tools/tests/manual-cues.cjs` and `node web/presentation/tools/tests/arrowheads.cjs`.
- **Tracked media:** `assets/narration-*.mp3`, `assets/droid-demonstration.mp4`, `assets/human-demonstration.mp4`, `documents/slides.pdf`, and `documents/presenter-guide.pdf` stay tracked because the static viewer uses them directly and they cannot be regenerated offline from tracked sources.

Details: [presentation/README.md](presentation/README.md).

## `web/dream-rsi-explainer/` — Dream-RSI explainer video

Deterministic Three.js + Canvas 2D renderer producing a 245 s (`1280×720`, `24 fps`) narrated explainer from `script.txt` and `studies/dream-rsi/attempts.jsonl`.

- **Build:**
  ```sh
  cd web/dream-rsi-explainer
  cp .env.example .env   # set OPENROUTER_API_KEY
  bun install
  python3 -m venv .venv-align && ./.venv-align/bin/pip install faster-whisper
  bun run build          # writes dist/dream-rsi-explainer-720p24.mp4
  ```
- **Tracked output:** `dist/dream-rsi-explainer-720p24.mp4` stays tracked because `tools/tts.mjs` requires online OpenRouter TTS (`OPENROUTER_API_KEY`) and synthesized `audio/` is gitignored, so the video cannot be regenerated offline from tracked sources alone.

Details: [dream-rsi-explainer/README.md](dream-rsi-explainer/README.md).
