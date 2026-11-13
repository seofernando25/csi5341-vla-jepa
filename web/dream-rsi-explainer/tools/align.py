"""Word-align each narration clip with faster-whisper, map onto script tokens, write build/timeline.json."""

import difflib
import json
import re
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent
GAP_SENTENCE = 0.16
GAP_SCENE = 0.55
LEAD = 0.7
TAIL = 3.2


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def clip_len(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def load_16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        sr = w.getframerate()
    n = int(len(x) * 16000 / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def align_sentence(model, wav: Path, sent: dict) -> list[tuple[float, float]]:
    dur = clip_len(wav)
    segs, _ = model.transcribe(
        load_16k(wav),
        language="en",
        word_timestamps=True,
        initial_prompt=sent["spoken"],
        beam_size=5,
    )
    wwords = [w for s in segs for w in s.words]
    # char streams with per-char times
    wchars, wtimes = [], []
    for w in wwords:
        text = norm(w.word)
        for k, ch in enumerate(text):
            f = (k + 0.5) / max(1, len(text))
            wchars.append(ch)
            wtimes.append(w.start + f * (w.end - w.start))
    spoken_words = []  # (token_idx, word)
    for ti, tok in enumerate(sent["tokens"]):
        for w in tok["spoken"].split():
            spoken_words.append((ti, norm(w)))
    schars, owner = [], []
    for wi, (_, w) in enumerate(spoken_words):
        for ch in w:
            schars.append(ch)
            owner.append(wi)
    sm = difflib.SequenceMatcher(None, "".join(schars), "".join(wchars), autojunk=False)
    hit = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            hit.setdefault(owner[a + k], []).append(wtimes[b + k])
    # per spoken word: anchor time (center) where matched
    nw = len(spoken_words)
    centers = [None] * nw
    for wi, ts in hit.items():
        centers[wi] = (min(ts), max(ts))
    # interpolate unmatched by char count between anchors
    lens = [max(1, len(w)) for _, w in spoken_words]
    first = next((i for i, c in enumerate(centers) if c), None)
    if first is None:
        step = dur / nw
        return [(i * step, (i + 1) * step) for i in range(nw)]
    starts = [None] * nw
    ends = [None] * nw
    for i, c in enumerate(centers):
        if c:
            starts[i], ends[i] = c
    i = 0
    while i < nw:
        if starts[i] is not None:
            i += 1
            continue
        j = i
        while j < nw and starts[j] is None:
            j += 1
        t0 = ends[i - 1] if i > 0 else 0.0
        t1 = starts[j] if j < nw else dur
        total = sum(lens[i:j])
        t = t0
        for k in range(i, j):
            d = (t1 - t0) * lens[k] / total
            starts[k], ends[k] = t, t + d
            t += d
        i = j
    # enforce monotonic, contiguous word boundaries: word ends where the next begins
    for k in range(nw - 1):
        starts[k + 1] = max(starts[k + 1], starts[k] + 0.04)
        ends[k] = starts[k + 1]
    ends[-1] = max(ends[-1], starts[-1] + 0.12)
    return list(zip(starts, ends, strict=True))


def main() -> None:
    data = json.loads((ROOT / "build/sentences.json").read_text())
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    t = LEAD
    last_scene = None
    scenes = {}
    sentences = []
    for s in data["sentences"]:
        wav = ROOT / s["wav"]
        dur = clip_len(wav)
        if last_scene is not None and s["scene"] != last_scene:
            t += GAP_SCENE - GAP_SENTENCE
        words = align_sentence(model, wav, s)
        # collapse spoken words into tokens
        toks, wi = [], 0
        for tok in s["tokens"]:
            n = len(tok["spoken"].split())
            span = words[wi : wi + n]
            wi += n
            toks.append(
                {
                    "shown": tok["shown"],
                    "punct": tok["punct"],
                    "emph": tok["emph"],
                    "cue": tok["cue"],
                    "start": round(t + span[0][0], 3),
                    "end": round(t + span[-1][1], 3),
                }
            )
        sentences.append(
            {
                "scene": s["scene"],
                "wav": s["wav"],
                "start": round(t, 3),
                "end": round(t + dur, 3),
                "tokens": toks,
            }
        )
        sc = scenes.setdefault(s["scene"], {"id": s["scene"], "start": t})
        sc["end"] = t + dur
        last_scene = s["scene"]
        t += dur + GAP_SENTENCE
    chips = {x["id"]: x["chip"] for x in data["scenes"]}
    for sc in scenes.values():
        sc["chip"] = chips[sc["id"]]
        sc["start"] = round(sc["start"], 3)
        sc["end"] = round(sc["end"], 3)
    total = round(t - GAP_SENTENCE + TAIL, 3)
    out = {"total": total, "scenes": list(scenes.values()), "sentences": sentences}
    (ROOT / "build/timeline.json").write_text(json.dumps(out, indent=1))
    print(f"total {total}s; narration ends {t - GAP_SENTENCE:.2f}s")
    for sc in out["scenes"]:
        print(f"  {sc['id']:8s} {sc['start']:6.2f} -> {sc['end']:6.2f}")


if __name__ == "__main__":
    main()
