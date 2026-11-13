// Mux PNG frames + 48 kHz stereo master.wav into an Instagram-compliant MP4.
// Key Instagram requirements: H.264 High@L4.0, yuv420p, CFR 24fps, closed GOP, bt709 tags,
// AAC-LC 48 kHz stereo, +faststart moov atom at head of file.
import { execFileSync } from "node:child_process";
import { mkdirSync } from "node:fs";

const root = new URL("..", import.meta.url).pathname;
mkdirSync(root + "dist", { recursive: true });
const out = root + "dist/dream-rsi-explainer-720p24.mp4";

execFileSync(
  "ffmpeg",
  [
    "-y", "-v", "error",
    "-framerate", "24", "-i", root + "frames/f%05d.png",
    "-i", root + "build/master.wav",
    "-c:v", "libx264",
    "-profile:v", "high", "-level:v", "4.0",
    "-preset", "medium", "-crf", "23",
    "-maxrate", "3M", "-bufsize", "6M",
    "-pix_fmt", "yuv420p",
    "-g", "48", "-keyint_min", "24", "-sc_threshold", "0",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-r", "24", "-fps_mode", "cfr",
    "-c:a", "aac", "-profile:a", "aac_low", "-ar", "48000", "-ac", "2", "-b:a", "192k",
    "-movflags", "+faststart",
    "-shortest",
    out,
  ],
  { stdio: "inherit" },
);
console.log("mux done:", out);
