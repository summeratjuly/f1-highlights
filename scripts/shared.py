"""Constants and helpers shared across the pipeline stages."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterable

# Relevance-score weights (fused by build_timeline.py).
WEIGHT_TEXT_DRIVER = 2.0          # commentator said driver's name
WEIGHT_TEXT_TEAM = 1.0            # commentator said team's name
WEIGHT_OCR_DRIVER = 3.0           # driver name on broadcast graphic
WEIGHT_OCR_TEAM = 1.5             # team name on broadcast graphic
WEIGHT_LIVERY_NUMBER = 2.5        # colour-gated car-number match
WEIGHT_LIVERY_COLOR_MIN = 0.5     # colour coverage at threshold
WEIGHT_LIVERY_COLOR_MAX = 2.0     # colour coverage saturation
WEIGHT_LIVERY_COLOR_SCALE = 40.0  # coverage-to-weight multiplier

# Discriminator for hit dicts emitted by analyze_livery.
HIT_COLOR = "color"
HIT_NUMBER = "number"

# Filenames inside the workdir. Owned by the producing stage but read
# by others - keep the names here so the contract is explicit.
FRAMES_DIR_NAME = "frames"
OCR_JSON_NAME = "ocr.json"
VISUAL_JSON_NAME = "visual.json"
LIVERY_JSON_NAME = "livery.json"
TRANSCRIPT_JSON_NAME = "transcript.json"
AUDIO_WAV_NAME = "audio.wav"
CLIPS_JSON_NAME = "clips.json"

# Minimum alias length considered for substring OCR matching - avoids
# 1-2 char garbage tokens matching on every frame.
MIN_ALIAS_LEN_OCR = 3


def compile_alias_re(aliases: Iterable[str], *, word_boundary: bool,
                     min_len: int = 0) -> re.Pattern:
    """Case-insensitive regex over a list of aliases.

    word_boundary=True for transcript (whole-word matches in commentary);
    False for OCR where tokens can be concatenated/truncated.
    """
    parts = [re.escape(a.lower()) for a in aliases if a and len(a) >= min_len]
    if not parts:
        return re.compile(r"(?!x)x")
    body = "|".join(parts)
    pattern = rf"\b({body})\b" if word_boundary else body
    return re.compile(pattern, re.IGNORECASE)


def probe_duration(video: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())
