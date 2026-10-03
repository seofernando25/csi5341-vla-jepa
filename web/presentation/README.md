# VLA-JEPA motion presentation

Self-contained static browser presentation: 18 narrated chapters. The main talk lasts about 9:40, split approximately equally between Noah and Fernando.

## Run

From the repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://localhost:8000/web/presentation/. No package installation, account or API key is required. Serve over HTTP rather than opening index.html directly so local modules and media load reliably. All runtime resources use relative paths, including when hosted under a subdirectory.

Click the stage, press Right or Space to play the next animation beat. Right-click or press Left to return to the previous beat; chapter navigation jumps directly to a section. Each beat plays and holds without audio. Replay beat repeats the current beat; N opens notes. Hover near the bottom for controls; use Full screen for presenting. Narration is retained as an export asset, with its browser controls removed. End is a hidden authoring shortcut for the final state.

## Presenter on a second computer

Run `python3 web/presentation/tools/relay.py` from the repo root, open its printed Slides URL on this computer and its Notes URL on the second computer. This local relay follows chapters/cues, reconnects automatically and provides adjustable large-text notes. N / Notes remains the manual fallback. See [setup, URLs and Wi-Fi troubleshooting](PRESENTER.md).

## Edit and rebuild

- `src/content.js`: wording, presenter notes, sources and timing; keep `data.json` in sync.
- `src/timing/cues.js`: one named active window and explicit duration per advance; idle authored time is skipped.
- `src/timing/clock.js`: pure forward/back/replay state machine, with no DOM, audio or drawing dependencies.
- `src/beat-controller.js`: clock-to-browser adapter; `src/controls.js` binds every input once.
- `src/player.js`: chapter selection and notes; `src/timeline.js`: chapter rail only.
- `src/media.js`: footage seeking/playback; `src/renderer.js`: frame loop.
- `src/scenes/dispatch.js`: scene routing; scene files own compositions, and shared drawing primitives stay in `drawing.js`.
- `src/choreography.js` and `src/scenes/`: motion and composition.
- `assets/`: runtime fonts, math images, footage, Three.js, p5 and stock narration.
- `tools/`: local authoring server, browser rendering and movie assembly. See [rebuild instructions](tools/README.txt).
- `documents/`: audited slide PDFs, presenter guide and export validation report.
- `assets/manifest.json`: byte sizes and SHA-256 hashes for included runtime assets.

Text edits require matching narration updates. No credential files, voice generation keys, research weights, checkpoints, full datasets or scratch renders are included. Generated render output stays in ignored `build/`. The final full movie and duplicate embedded HTML remain outside Git; the source and audio here can rebuild the movie.

## Provenance

See [detailed presentation notes](README.txt) for scientific distinctions and source links. The included clips are illustrative dataset examples, not VLA-JEPA rollouts or identified training samples. Human footage is SSV2 validation sample 174198; robot footage is the documented DROID example. Source capture frames are duplicated for the 60 fps timeline, without interpolation. Feature ribbons and control sequences are schematic, not measured embeddings or policy outputs. The project section describes the proposal and work in progress; no project performance results are announced.

The website source and narrowly scoped example clips are intentionally retained under the user's request, as an exception to the repository's usual exclusion of videos/dataset artifacts. Original research code, results and recovery work are unchanged.

Three.js is MIT licensed (license included), p5.js is LGPL-2.1 licensed (license included), and the font license is included. External dataset examples retain their source rights; see the linked source terms before redistribution. Puck and Charon are synthetic stock voices generated through the authorized OpenRouter account.

The related-work sequence introduces the cost of control labels before contrasting pixel, feature and future-state supervision. The flow-matching sequence uses complete move/rotate/grip control chunks; its command-time axis is separate from generation time. Robot examples illustrate multiple embodiments, with compatible control conventions and target-specific post-training explained explicitly. Shared connector primitives keep one tangent-aligned tip per path; masked text changes take about 0.38 seconds.

Motion typography lives in `src/motion-type.js`: masked reel-card changes retain a shared word prefix, selected terms squash and settle, and short instructions type in. Existing heading transitions are preserved. Effects stop at their settled state and the browser respects reduced-motion preferences. Authoring exports use the same deterministic beat times.

## Transition review

The source contains 23 JavaScript files, grouped by concern. All 48 beats across 18 chapters have explicit durations: ordinary transitions are 600 ms; deliberate footage and continuous demonstrations are longer. An advance plays one beat and holds. Right during a playing beat accelerates its remaining motion to at most 120 ms, preserving its current progress. No extra beats are queued. Previous reverses the current beat; at a chapter boundary it returns to the preceding chapter’s final state. Replay repeats only the current beat. Neither the clock nor navigation depends on narration.

Run `node web/presentation/tools/tests/manual-cues.cjs` from the repository root. For visual review, run `python3 web/presentation/tools/serve.py`, open its localhost address with `?authoring=1`, open Notes and choose **Audit every transition**. It captures all forward/replay paths, within-chapter reversals, and chapter boundaries in both directions at five positions. Output stays in ignored `build/rendered/`. This deterministic sampled audit checks layouts and replay consistency; it does not measure sustained real-device frame rate.

Live footage loops at normal speed in chapters 1 and 2, independently of manual cue timing. Export captures retain explicit video seeks for deterministic rendering. All arrowheads share a trimmed shaft/base join; short connectors use a direct path to avoid hooked tips.

Arrowhead regression checks: `node web/presentation/tools/tests/arrowheads.cjs`. These verify that each head meets the shaft exactly, including short connectors, reversed paths, and a zero terminal Bézier tangent.
