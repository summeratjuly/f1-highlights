"""Stage 1: extract audio and transcribe with word-level timestamps.

Output JSON shape:
{
  "duration": float,
  "segments": [
    {"start": float, "end": float, "text": str,
     "words": [{"start": float, "end": float, "word": str}, ...]}
  ]
}
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from shared import AUDIO_WAV_NAME, TRANSCRIPT_JSON_NAME


def extract_audio(video: Path, out_wav: Path) -> None:
    if out_wav.exists():
        return
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", "16000",
         "-vn", "-loglevel", "error", str(out_wav)],
        check=True,
    )


def transcribe(video: Path, workdir: Path, model_size: str = "small.en") -> dict:
    out_json = workdir / TRANSCRIPT_JSON_NAME
    if out_json.exists():
        return json.loads(out_json.read_text())

    from faster_whisper import WhisperModel

    wav = workdir / AUDIO_WAV_NAME
    extract_audio(video, wav)

    # int8 runs well on CPU; users with GPU can swap compute_type.
    model = WhisperModel(model_size, device="auto", compute_type="int8")
    segments, info = model.transcribe(
        str(wav),
        word_timestamps=True,
        vad_filter=True,
        language="en",
    )

    out_segments = []
    for seg in segments:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({"start": w.start, "end": w.end, "word": w.word})
        out_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "words": words,
        })

    result = {"duration": info.duration, "segments": out_segments}
    out_json.write_text(json.dumps(result))
    return result
