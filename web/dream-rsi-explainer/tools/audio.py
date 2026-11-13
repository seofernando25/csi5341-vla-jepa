"""Build 48 kHz stereo master audio: placed narration + ducked synth bed + transition whooshes + slam impacts."""

import json
import subprocess
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 48000


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        return (
            np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        )


def main() -> None:
    tl = json.loads((ROOT / "build/timeline.json").read_text())
    dur = float(tl["total"])
    n = int(dur * SR) + SR
    voice = np.zeros(n, dtype=np.float32)
    cues = {}
    for s in tl["sentences"]:
        x = read_wav(ROOT / s["wav"])
        i0 = int(s["start"] * SR)
        voice[i0 : i0 + len(x)] += x
        for tok in s["tokens"]:
            if tok["cue"]:
                cues[tok["cue"]] = float(tok["start"])

    # Envelope of voice for sidechain ducking (250 ms window)
    win = int(0.12 * SR)
    env = np.convolve(np.abs(voice), np.ones(win, dtype=np.float32) / win, mode="same")
    duck = 1.0 - 0.62 * np.clip(env * 5.0, 0.0, 1.0)

    # Subtle 120 BPM minor synth pulse + pad (stereo)
    t = np.arange(n, dtype=np.float32) / SR
    roots = np.array([55.0, 43.65, 65.41, 49.0], dtype=np.float32)  # A1, F1, C2, G1 (2s each)
    f0 = roots[(t // 2.0).astype(np.int32) % 4]
    phase = 2 * np.pi * np.cumsum(f0) / SR
    bass = np.tanh(1.6 * np.sin(phase)) * (0.5 + 0.5 * np.maximum(0.0, np.sin(2 * np.pi * 2.0 * t)))
    pad_l = 0.35 * np.sin(phase * 2.0 + 0.2 * np.sin(2 * np.pi * 0.3 * t)) + 0.25 * np.sin(
        phase * 3.0
    )
    pad_r = 0.35 * np.sin(phase * 2.0 - 0.2 * np.sin(2 * np.pi * 0.3 * t)) + 0.25 * np.sin(
        phase * 2.995
    )
    bed_l = (0.055 * bass + 0.032 * pad_l) * duck
    bed_r = (0.055 * bass + 0.032 * pad_r) * duck

    # Whoosh on scene wipes
    rng = np.random.default_rng(7)
    sfx_l = np.zeros(n, dtype=np.float32)
    sfx_r = np.zeros(n, dtype=np.float32)
    for sc in tl["scenes"][1:]:
        w0 = int((sc["start"] - 0.28) * SR)
        wl = int(0.38 * SR)
        if 0 <= w0 < n - wl:
            u = np.linspace(0, 1, wl, dtype=np.float32)
            w_env = np.sin(np.pi * u) ** 2
            noise = np.convolve(
                rng.standard_normal(wl).astype(np.float32), np.ones(18) / 18, mode="same"
            )
            sw = 0.09 * w_env * noise
            sfx_l[w0 : w0 + wl] += sw * (1 - 0.4 * u)
            sfx_r[w0 : w0 + wl] += sw * (0.6 + 0.4 * u)

    # Low impact on key slam cues
    for name in ("zero", "who", "never", "zero2", "lossnot", "r3", "fail6", "robot"):
        if name not in cues:
            continue
        i0 = int(cues[name] * SR)
        hl = int(0.45 * SR)
        if 0 <= i0 < n - hl:
            u = np.linspace(0, 1, hl, dtype=np.float32)
            boom = np.sin(2 * np.pi * (95.0 * (1 - 0.55 * u)) * u * 0.45) * np.exp(-u * 6.5) * 0.22
            sfx_l[i0 : i0 + hl] += boom
            sfx_r[i0 : i0 + hl] += boom

    left = voice[: int(dur * SR)] + bed_l[: int(dur * SR)] + sfx_l[: int(dur * SR)]
    right = voice[: int(dur * SR)] + bed_r[: int(dur * SR)] + sfx_r[: int(dur * SR)]
    stereo = np.stack([left, right], axis=1)
    peak = max(1e-4, float(np.max(np.abs(stereo))))
    pcm = np.clip(stereo / peak * 0.88 * 32767.0, -32768, 32767).astype(np.int16)

    raw_wav = ROOT / "build/master-raw.wav"
    with wave.open(str(raw_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())

    # Loudness-normalize to -14 LUFS (Instagram / social standard), true peak -1 dBTP
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(raw_wav),
            "-af",
            "loudnorm=I=-14:TP=-1.0:LRA=9",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            str(ROOT / "build/master.wav"),
        ],
        check=True,
    )
    print(f"master.wav ready ({dur:.2f}s, 48kHz stereo)")


if __name__ == "__main__":
    main()
