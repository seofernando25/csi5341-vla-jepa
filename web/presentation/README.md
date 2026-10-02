# VLA-JEPA motion presentation

Self-contained static browser presentation: 18 narrated chapters and four optional appendices. The main talk lasts about 9:40, split approximately equally between Noah and Fernando.

## Run

From the repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://localhost:8000/web/presentation/. No package installation, account or API key is required. Serve over HTTP rather than opening index.html directly so local modules and media load reliably. All runtime resources use relative paths, including when hosted under a subdirectory.

Arrow keys change chapters; Space controls scene motion; End shows the final visual state; N opens notes. Hover near the bottom for the chapter timeline and narration controls. Use Full screen for presenting. Narration continues through the main talk; scene motion has an independent clock.

## Edit and rebuild

- `src/content.js`: wording, presenter notes, sources and timing; keep `data.json` in sync.
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
