"""Stage 5.3: refine clip start/end boundaries via LLM.

After build_clips snaps to sentence-end punctuation, the cuts feel
unnatural surprisingly often:
  • Clip opens with an unresolved pronoun ("He's now ahead.") or a
    coordinator ("And the gap is two seconds.").
  • Clip ends on a half-built list ("...so it's Hamilton, Vettel,").
  • Clip splits a topic the commentator was building across two
    sentences ("...the Mercedes pulled away. The pit window…").

The LLM gets the proposed boundary plus a short list of sentence-edge
*candidates* nearby and picks the most narratively coherent ones.
Falls back to the originals if the call fails.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from bridge_summarizer import _query_llm, _parse_response


_REFINE_WINDOW = 12.0       # ±seconds around proposed boundary to consider
_MAX_CANDIDATES = 6         # cap candidates per side to keep prompt tight
_CONCURRENCY = 3            # parallel LLM calls


_REFINE_SYSTEM_PROMPT = """\
You refine F1 highlight clip boundaries so the cut feels natural to the
viewer. Given a proposed clip and lists of sentence-boundary candidates
near the start and end, pick the candidate index for each that produces
the most coherent self-contained clip.

Rules of thumb (in priority order):

  1. The clip's first sentence must have a complete subject.
     Avoid candidates whose first words are: And, But, So, Or, Yet,
     Because, As, While, He, She, It, They, That, This, Those.
     A coordinating conjunction or unresolved pronoun makes the clip
     feel like it starts mid-thought.

  2. The clip's last sentence must finish a thought.
     Avoid candidates that end on a dangling list ("...Hamilton, Vettel,")
     or a hanging modifier ("...because of the").

  3. Keep the topic intact. If the speaker is building a list or a
     comparison across two sentences, don't cut between them — extend
     end candidate forward, or pull start candidate back.

  4. If the current boundary is already the best choice, keep it.

You'll be given:
  - PROPOSED start and end timestamps.
  - START_CANDIDATES: a small list of nearby sentence-start times
    (each tagged [s0], [s1], ...) with the first ~80 chars of that
    sentence.
  - END_CANDIDATES: same shape for sentence-end times tagged [e0]...
  - A RAW TRANSCRIPT excerpt covering the whole window for context.

Reply with ONLY a single JSON object — no prose, no code fences.
Schema:
  {"start": "s2", "end": "e1", "reason": "<one short phrase>"}

Use the candidate id strings (e.g. "s0", "e3") — never raw timestamps.
"""


@dataclass
class _Candidate:
    id: str
    time: float
    snippet: str


_BAD_FIRST_WORDS = {
    "and", "but", "so", "or", "yet", "because", "as", "while",
    "he", "she", "it", "they", "that", "this", "those", "these",
    "him", "her", "his", "their", "its",
}
_DANGLING_END = {"and", "but", "or", "so", "because", "as", "while", "if",
                 "then", "yet", "the", "a", "an", "to", "of", "in", "on",
                 "at", "with", "for"}


def _segments_in_window(segments: list[dict], t0: float, t1: float) -> list[dict]:
    return [s for s in segments
            if s["end"] >= t0 - 0.1 and s["start"] <= t1 + 0.1]


def _ends_sentence(text: str) -> bool:
    if not text:
        return False
    s = text.rstrip(" \"'”’)").rstrip()
    return bool(s) and s[-1] in ".?!"


def _start_candidates(segments: list[dict], target: float) -> list[_Candidate]:
    """Return up to _MAX_CANDIDATES sentence-starts within ±_REFINE_WINDOW
    of the proposed start. If no segments in the window qualify as
    sentence-starts (dense comma-spliced commentary), fall back to any
    segment edge in the window so the LLM still has options to choose
    from.
    """
    strict: list[_Candidate] = []
    loose: list[_Candidate] = []
    for i, seg in enumerate(segments):
        if abs(seg["start"] - target) > _REFINE_WINDOW:
            continue
        snippet = (seg.get("text") or "").strip()[:80]
        is_sentence_start = (i == 0 or _ends_sentence(
            (segments[i - 1].get("text") or "").strip()))
        cand = _Candidate(id=f"s{len(loose)}", time=seg["start"], snippet=snippet)
        loose.append(cand)
        if is_sentence_start:
            strict.append(_Candidate(
                id=f"s{len(strict)}", time=seg["start"], snippet=snippet,
            ))
            if len(strict) >= _MAX_CANDIDATES:
                break
    if strict:
        return strict
    # Fallback: dense commentary — re-id the loose set so the LLM gets
    # contiguous ids starting at s0.
    loose = loose[:_MAX_CANDIDATES]
    for i, c in enumerate(loose):
        c.id = f"s{i}"
    return loose


def _end_candidates(segments: list[dict], target: float) -> list[_Candidate]:
    strict: list[_Candidate] = []
    loose: list[_Candidate] = []
    for seg in segments:
        if abs(seg["end"] - target) > _REFINE_WINDOW:
            continue
        text = (seg.get("text") or "").strip()
        snippet = text[:80]
        cand = _Candidate(id=f"e{len(loose)}", time=seg["end"], snippet=snippet)
        loose.append(cand)
        if _ends_sentence(text):
            strict.append(_Candidate(
                id=f"e{len(strict)}", time=seg["end"], snippet=snippet,
            ))
            if len(strict) >= _MAX_CANDIDATES:
                break
    if strict:
        return strict
    loose = loose[:_MAX_CANDIDATES]
    for i, c in enumerate(loose):
        c.id = f"e{i}"
    return loose


def _build_user_prompt(clip: dict, transcript: dict,
                       start_cands: list[_Candidate],
                       end_cands: list[_Candidate]) -> str:
    parts: list[str] = []
    parts.append(f"PROPOSED CLIP: start={clip['start']:.2f}  "
                 f"end={clip['end']:.2f}  duration={clip['duration']:.2f}s")
    parts.append(f"EVENT TYPE: {clip.get('event_type', 'mention')}")
    parts.append("")
    parts.append("START_CANDIDATES (pick one):")
    for c in start_cands:
        marker = "  ← current" if abs(c.time - clip["start"]) < 0.1 else ""
        parts.append(f"  [{c.id}] {c.time:8.2f}  {c.snippet!r}{marker}")
    parts.append("")
    parts.append("END_CANDIDATES (pick one):")
    for c in end_cands:
        marker = "  ← current" if abs(c.time - clip["end"]) < 0.1 else ""
        parts.append(f"  [{c.id}] {c.time:8.2f}  {c.snippet!r}{marker}")
    parts.append("")

    # Raw transcript covering the full window so the LLM can see context.
    win_start = min(c.time for c in start_cands) - 1.0 if start_cands else clip["start"] - _REFINE_WINDOW
    win_end = max(c.time for c in end_cands) + 1.0 if end_cands else clip["end"] + _REFINE_WINDOW
    parts.append("TRANSCRIPT (start_time | text):")
    for seg in _segments_in_window(transcript["segments"], win_start, win_end):
        text = (seg.get("text") or "").strip().replace("\n", " ")
        if not text:
            continue
        parts.append(f"  {seg['start']:8.2f} | {text}")
    return "\n".join(parts)


def _heuristic_refine(clip: dict, start_cands: list[_Candidate],
                      end_cands: list[_Candidate]) -> tuple[float, float]:
    """Cheap fallback when LLM is unavailable. Skips candidates whose
    first words are bad (orphan pronouns / coordinators) or that end on
    a dangling word, picking the closest acceptable to the original."""
    def first_word(snip: str) -> str:
        m = re.match(r"\W*(\w+)", snip)
        return m.group(1).lower() if m else ""

    def last_word(snip: str) -> str:
        m = re.findall(r"(\w+)", snip.rstrip(" \"'”’).?!"))
        return m[-1].lower() if m else ""

    # Start: closest candidate whose first word isn't in BAD_FIRST_WORDS.
    new_start = clip["start"]
    best_dist = float("inf")
    for c in start_cands:
        if first_word(c.snippet) in _BAD_FIRST_WORDS:
            continue
        d = abs(c.time - clip["start"])
        if d < best_dist:
            best_dist = d
            new_start = c.time

    # End: closest candidate whose last word (sans punctuation) isn't dangling.
    new_end = clip["end"]
    best_dist = float("inf")
    for c in end_cands:
        if last_word(c.snippet) in _DANGLING_END:
            continue
        d = abs(c.time - clip["end"])
        if d < best_dist:
            best_dist = d
            new_end = c.time

    return new_start, new_end


async def _refine_one(clip: dict, transcript: dict) -> tuple[float, float, str, str | None]:
    """Refine a single clip's boundaries. Returns (start, end, source, reason).
    `reason` is the LLM's free-text justification when available, else None.
    """
    segments = transcript["segments"]
    start_cands = _start_candidates(segments, clip["start"])
    end_cands = _end_candidates(segments, clip["end"])

    if not start_cands or not end_cands:
        return clip["start"], clip["end"], "no-candidates", None

    user_prompt = _build_user_prompt(clip, transcript, start_cands, end_cands)
    text = await _query_llm(_REFINE_SYSTEM_PROMPT, user_prompt)
    parsed = _parse_response(text) if text else None

    if parsed and isinstance(parsed.get("start"), str) and isinstance(parsed.get("end"), str):
        s_lookup = {c.id: c for c in start_cands}
        e_lookup = {c.id: c for c in end_cands}
        sc = s_lookup.get(parsed["start"])
        ec = e_lookup.get(parsed["end"])
        if sc and ec and ec.time > sc.time:
            reason = parsed.get("reason")
            return sc.time, ec.time, "llm", str(reason) if reason else None

    # Fallback to heuristic — still better than the raw boundary in many cases.
    h_start, h_end = _heuristic_refine(clip, start_cands, end_cands)
    if h_end > h_start:
        return h_start, h_end, "heuristic", None
    return clip["start"], clip["end"], "original", None


async def _refine_clips_async(clips: list[dict], transcript: dict,
                              concurrency: int) -> tuple[list[dict], dict[str, int]]:
    sem = asyncio.Semaphore(concurrency)

    async def _work(clip: dict) -> tuple[dict, str]:
        async with sem:
            orig_start = clip["start"]
            orig_end = clip["end"]
            new_start, new_end, source, reason = await _refine_one(clip, transcript)
            new_clip = dict(clip)
            new_clip["start"] = round(new_start, 2)
            new_clip["end"] = round(new_end, 2)
            new_clip["duration"] = round(new_end - new_start, 2)
            new_clip["original_start"] = round(orig_start, 2)
            new_clip["original_end"] = round(orig_end, 2)
            new_clip["refine_source"] = source
            if reason:
                new_clip["refine_reason"] = reason
            return new_clip, source

    results = await asyncio.gather(*(_work(c) for c in clips))
    counts: dict[str, int] = {}
    out_clips = []
    for new_clip, source in results:
        out_clips.append(new_clip)
        counts[source] = counts.get(source, 0) + 1
    return out_clips, counts


def refine_clips(clips: list[dict], transcript: dict,
                 concurrency: int = _CONCURRENCY) -> tuple[list[dict], dict[str, int]]:
    """Refine all clip boundaries via LLM (with heuristic fallback per clip).

    Each refined clip gets enriched with `original_start`, `original_end`,
    `refine_source` ("llm" / "heuristic" / "original" / "no-candidates"),
    and optionally `refine_reason` (the LLM's one-phrase justification).

    Returns (refined_clips, source_counts).
    """
    if not clips:
        return clips, {}
    return asyncio.run(_refine_clips_async(clips, transcript, concurrency))
