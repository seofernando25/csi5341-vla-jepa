# Dream-RSI explainer video

> **2026-11-15 — pre-cleanroom, do not present; superseded by the clean-room cut.**
> The MP4 in `dist/` was rendered from the old numbered cut and is kept only as history.
> Re-render before presenting; nothing here is a report claim.

208-second explainer (`1280×720`, `24 fps`) covering the VLA-JEPA compression
story and the Dream-RSI arc (`arXiv:2609.14858`) across 14 scenes. Duration comes
from the current narration timeline (`build/timeline.json`).

1. **Hook** — candidate `n0008` lowers held-out action loss, then scores `0/10` on the robot.
2. **The Idea** — live search over branches where every candidate burns a GPU training run.
3. **The Dream** — accumulated search history becomes an offline replay simulator with monotonic `argmax V(π)` selection.
4. **The Loop** — Search → Remember → Dream → Select.
5. **The Codebase** — `evaluation/` + `fastpath.py` + `nf4.py`, `evaluation/compression/`, `rsi/`, `studies/`.
6. **The Protocol** — screen, smoke, development, reserved; no phase is pooled and reserved never guides selection.
7. **The First Attempt** — SmolVLM backbone swap with LLM-proposed residual adapters; held-out loss falls.
8. **The Failure** — `10,000`-step cloud confirmation (`0/10`) caused by a double `÷255` image rescale. Held-out loss is not control success.
9. **The Audit** — four ways a search misleads (in-sample replay, wrong base arm, gate that kills good branches, missing M=4 revision and frozen control), each blocked by a rule written before the run.
10. **The Harness** — expansion, offline replay, replay-gated policy development, fixed gates, parallel rollout, per-cycle record.
11. **Live Search** — the clean loop is not launched; the scene states the method, not a score.
12. **The Precision Surface** — language, vision and action quantized separately at INT8, NF4, HQQ3 and HQQ2, plus a joint 2-bit cell.
13. **Where the Loss Comes From** — weight precision, execution path, search procedure, separated by the protocol.
14. **Where We Stopped** — selection frozen before the reserved read; experiments stop for the report and talk.

## Provenance

Clean-room studies only (`studies/program-v1`, `studies/confirmation`, `studies/recovery`),
plus the closed SmolVLM negative result (`studies/smolvla`). Pre-cleanroom numbers are not
quoted and no scene carries a per-task bar chart; results are stated qualitatively.

## Pipeline

- `tools/prep.mjs` — builds the 3D search-tree geometry from `studies/program-v1/protocol.json` (`dream_rsi.grid_plan`) into `build/tree.json`. No run data is read: the tree is the pre-registered plan.
- `tools/tts.mjs` — per-sentence OpenRouter TTS (`mistralai/voxtral-mini-tts-2603`, `en_paul_confident`), silence-trimmed and sped up `1.25×` via `ffmpeg atempo`.
- `tools/align.py` — word-level timestamps with `faster-whisper` (`small.en`), mapped onto script tokens and `@cue:` visual triggers in `build/timeline.json`.
- `tools/audio.py` — `48 kHz` stereo master (`-14 LUFS`) combining narration, ducked synth pulse, transition whooshes and slam impacts.
- `tools/capture.mjs` — deterministic GPU capture (`src/stage3d.js` Three.js + UnrealBloom, `src/scenes.js` HUD cards, `src/captions.js` word-highlighted subtitles). Clears `frames/` first, then fails if the written set does not match `ceil(total × 24)`.
- `tools/mux.mjs` — Instagram-compliant MP4 (`H.264 High@L4.0`, `yuv420p`, `bt709`, CFR `24 fps`, closed GOP, `AAC-LC` `48 kHz` stereo, `+faststart`). Refuses a `frames/` sequence that does not match `build/timeline.json`.

## Rebuild

```sh
cd web/dream-rsi-explainer
cp .env.example .env   # set OPENROUTER_API_KEY; never commit .env
bun install
python3 -m venv .venv-align && ./.venv-align/bin/pip install faster-whisper
bun run build
```

Output: `dist/dream-rsi-explainer-720p24.mp4` (overwrites the stale pre-cleanroom render).
