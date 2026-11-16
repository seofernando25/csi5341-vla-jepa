// Mux PNG frames + 48 kHz stereo master.wav into an Instagram-compliant MP4.
// H.264 High@L4.0, yuv420p, bt709, CFR 30 fps, closed GOP, AAC-LC 48 kHz, +faststart.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";

const root = new URL("..", import.meta.url).pathname;
const FPS = 30;
const out = root + "dist/project-explainer-720p30.mp4";

// The image sequence is numbered by time, so a frame left by another cut would silently
// extend the video past its audio. Require exactly the frames this timeline asks for.
const total = Math.ceil(JSON.parse(readFileSync(root + "build/timeline.json", "utf8")).total * FPS);
const expected = [...Array(total).keys()].map((i) => `f${String(i).padStart(5, "0")}.png`);
const present = new Set(
  existsSync(root + "frames") ? readdirSync(root + "frames").filter((f) => f.endsWith(".png")) : [],
);
const extra = [...present].filter((f) => !expected.includes(f));
const missing = expected.filter((f) => !present.has(f));
if (extra.length || missing.length) {
  throw new Error(
    `frames/ is not a clean ${total}-frame sequence: ${extra.length} unexpected` +
      (extra.length ? ` (${extra.slice(0, 3).join(", ")})` : "") +
      `, ${missing.length} missing` +
      (missing.length ? ` (${missing.slice(0, 3).join(", ")})` : "") +
      "; re-run bun run capture",
  );
}
mkdirSync(root + "dist", { recursive: true });

execFileSync(
  "ffmpeg",
  [
    "-y", "-v", "error",
    "-framerate", "30", "-i", root + "frames/f%05d.png",
    "-i", root + "build/master.wav",
    "-frames:v", String(total),
    "-c:v", "libx264",
    "-profile:v", "high", "-level:v", "4.0",
    "-preset", "medium", "-crf", "21",
    "-maxrate", "4M", "-bufsize", "8M",
    "-pix_fmt", "yuv420p",
    "-g", "60", "-keyint_min", "30", "-sc_threshold", "0",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-r", "30", "-fps_mode", "cfr",
    "-c:a", "aac", "-profile:a", "aac_low", "-ar", "48000", "-ac", "2", "-b:a", "192k",
    "-movflags", "+faststart",
    "-shortest",
    out,
  ],
  { stdio: "inherit" },
);
console.log("mux done:", out);
