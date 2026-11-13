// Synthesize narration per sentence via OpenRouter TTS, then trim silence and speed up.
// Writes audio/proc/s###.wav (48 kHz mono) and build/sentences.json for tools/audio.py.
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { parseScript } from "./script.mjs";

const root = new URL("..", import.meta.url).pathname;
const env = Object.fromEntries(
  readFileSync(root + ".env", "utf8")
    .split("\n")
    .map((l) => l.match(/^([A-Za-z_]+)=(.*)$/))
    .filter(Boolean)
    .map((m) => [m[1], m[2].trim()]),
);
const MODEL = env.TTS_MODEL || "mistralai/voxtral-mini-tts-2603";
const VOICE = env.TTS_VOICE || "en_paul_confident";
const TEMPO = Number(env.TTS_TEMPO || 1.25);

for (const d of ["audio/raw", "audio/proc", "build"]) mkdirSync(root + d, { recursive: true });
const scenes = parseScript(readFileSync(root + "script.txt", "utf8"));

const out = { voice: VOICE, tempo: TEMPO, scenes: [], sentences: [] };
let n = 0;
for (const scene of scenes) {
  out.scenes.push({ id: scene.id, chip: scene.chip });
  for (const s of scene.sentences) {
    const key = createHash("sha1").update(`${MODEL}|${VOICE}|${s.spokenText}`).digest("hex").slice(0, 16);
    const raw = `${root}audio/raw/${key}.mp3`;
    if (!existsSync(raw)) {
      const res = await fetch("https://openrouter.ai/api/v1/audio/speech", {
        method: "POST",
        headers: { Authorization: `Bearer ${env.OPENROUTER_API_KEY}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: MODEL, input: s.spokenText, voice: VOICE, response_format: "mp3" }),
      });
      if (!res.ok) throw new Error(`TTS "${s.spokenText}": ${res.status} ${await res.text()}`);
      writeFileSync(raw, Buffer.from(await res.arrayBuffer()));
      console.log("synth:", s.spokenText);
    }
    const wav = `audio/proc/s${String(n).padStart(3, "0")}.wav`;
    const trim = "silenceremove=start_periods=1:start_threshold=-48dB:start_silence=0.02";
    execFileSync("ffmpeg", [
      "-y", "-v", "error", "-i", raw,
      "-af", `${trim},areverse,${trim},areverse,atempo=${TEMPO},aresample=48000`,
      "-ac", "1", "-c:a", "pcm_s16le", root + wav,
    ]);
    out.sentences.push({ index: n, scene: scene.id, wav, spoken: s.spokenText, tokens: s.tokens });
    n++;
  }
}
writeFileSync(root + "build/sentences.json", JSON.stringify(out, null, 1));
console.log(`${n} sentences, voice ${VOICE}, tempo ${TEMPO}`);
