# Dream-RSI explainer video

245-second explainer (`1280×720`, `24 fps`) covering the VLA-JEPA compression
story and the Dream-RSI arc (`arXiv:2609.14858`) across 14 scenes:

1. **Hook** — candidate `n0008` cuts held-out action loss `0.32 → 0.28`, then scores `0/10` on the robot.
2. **The Idea** — live search over branches where every candidate burns a GPU training run.
3. **The Dream** — accumulated search history becomes an offline replay simulator (`16 ms`, `0` GPU runs) with monotonic `argmax V(π)` selection.
4. **The Loop** — Search → Remember → Dream → Select.
5. **Our Project** — `5.5 GB` VLA-JEPA baseline (`B16`), SmolVLM backbone swap (`S500`), `27` LLM-proposed conditioning adapters (`n0008`: `0.3197 → 0.2762`).
5b. **The Codebase** — `evaluation/` + `fastpath.py` + `nf4.py`, `evaluation/compression/`, `rsi/`, `studies/`.
5c. **The Headline Arms** — reserved-500: `B16 475/500 = 95.0%`, `C1 484/500 = 96.8% @ 1.38 GB`, `C2 490/500 = 98.0% @ 31.6 ms`.
6. **The Failure** — `10,000`-step cloud confirmation (`0/10`), caused by a double `÷255` image rescale (`255×` contrast crush). Held-out loss is not control success.
7. **The First Attempt** — SmolVLM backbone swap, `27` adapters, best loss `0.3197 → 0.2762`.
8. **The Failure** — `10,000`-step cloud confirmation (`0/10`), double `÷255` rescale, recovery `31/100` vs `82/100` B16. Loss is not success.
9. **The Audit** — Noah's Stage 2 on Qwen (`s2-c2-t10-v25`, `85/100`, `904.8 MB`); faithful scoring, in-sample replay, T2 base, `13.08%` false kills, missing M=4 + control.
10. **The Repair Kit** — RTX 3090: leave-one-world-out replay (`10/10` + `5/5` folds), M=4 generator, exact-binomial gate (`4.43%`), parallel harness.
11. **Live Rounds 1–3** — `r1` promoted (`83/100`, `853.1 MB`), pruning cliff (`820–845 MB`), `r3` promoted (`88/100 [80.2, 93.0]`, `885.1 MB`).
12. **The Full-Vocab Jump** — Round 4 C1 base (`r4`, `86/100`, `1199.6 MB`); Round 5 HQQ 3-bit (`85/100`), control bottleneck, head-pruning fails (`6/10`).
13. **Why 83–88** — tasks 0, 5, 8 at `40–70%`; the gripper is the ceiling.
14. **Where We Stopped** — Round 6 staged, `500` reserved trials `100%` untouched, freeze for report + talk.

## Pipeline

- `tools/prep.mjs` — exports `studies/dream-rsi/attempts.jsonl` into `build/tree.json`.
- `tools/tts.mjs` — per-sentence OpenRouter TTS (`mistralai/voxtral-mini-tts-2603`, `en_paul_confident`), silence-trimmed and sped up `1.25×` via `ffmpeg atempo`.
- `tools/align.py` — word-level timestamps with `faster-whisper` (`small.en`), mapped onto script tokens and `@cue:` visual triggers in `build/timeline.json`.
- `tools/audio.py` — `48 kHz` stereo master (`-14 LUFS`) combining narration, ducked synth pulse, transition whooshes and slam impacts.
- `tools/capture.mjs` — deterministic GPU capture (`src/stage3d.js` Three.js + UnrealBloom, `src/scenes.js` HUD cards, `src/captions.js` word-highlighted subtitles).
- `tools/mux.mjs` — Instagram-compliant MP4 (`H.264 High@L4.0`, `yuv420p`, `bt709`, CFR `24 fps`, closed GOP, `AAC-LC` `48 kHz` stereo, `+faststart`).

## Rebuild

```sh
cd web/dream-rsi-explainer
cp .env.example .env   # set OPENROUTER_API_KEY; never commit .env
bun install
python3 -m venv .venv-align && ./.venv-align/bin/pip install faster-whisper
bun run build
```

Output: `dist/dream-rsi-explainer-720p24.mp4`.
