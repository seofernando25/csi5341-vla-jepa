Rebuilding the source presentation

The viewer itself needs no installation: all assets are local.
Edit src/content.js for chapter wording, notes, timings and sources.
Keep data.json in sync with that array; export tools use the JSON mirror.
Edit src/choreography.js for beat lengths and the scene modules for visual changes.
Narration audio is stored separately. Text changes need a fresh matching voice recording.

For local authoring:
  python3 tools/serve.py
  Open http://localhost:8765/?authoring=1
  Press N, then choose Export revised presentation.

The browser writes deterministic 1080p/60fps frames and scene videos into build/rendered.
Then run python3 tools/assemble.py to produce a complete narrated movie.
Assembly requires an available ffmpeg installation. No credentials are included.
The ordinary viewer and portable HTML do not display authoring controls.

Stable asset slots preserve the earlier audio filenames; slot 2 is reserved and hidden.
Visible chapter numbers are in the content records. Four B entries are optional appendices.
Final PDFs and the rehearsal guide shipped alongside the source are the audited deliverables.
