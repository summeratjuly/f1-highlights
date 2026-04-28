"""Stage 4: fuse text mentions + visual hits into a per-second relevance
score timeline, then smooth to reduce flicker.
"""
from __future__ import annotations

import math


def build_timeline(duration: float, text_mentions: list[dict],
                   visual_hits: list[dict],
                   livery_hits: list[dict] | None = None,
                   smooth_window: int = 3) -> list[float]:
    n = int(math.ceil(duration)) + 1
    scores = [0.0] * n

    for m in text_mentions:
        s, e = int(m["start"]), int(math.ceil(m["end"]))
        for i in range(max(0, s), min(n, e + 1)):
            scores[i] += m["weight"]

    for h in visual_hits:
        t = int(h["time"])
        if 0 <= t < n:
            scores[t] += h["weight"]

    for h in (livery_hits or []):
        t = int(h["time"])
        if 0 <= t < n:
            scores[t] += h["weight"]

    if smooth_window > 1:
        smoothed = [0.0] * n
        half = smooth_window // 2
        for i in range(n):
            a, b = max(0, i - half), min(n, i + half + 1)
            smoothed[i] = sum(scores[a:b]) / (b - a)
        scores = smoothed

    return scores
