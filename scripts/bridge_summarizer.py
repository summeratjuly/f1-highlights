"""LLM-driven summarizer for the gap between two highlight clips.

For each bridge we ask Claude (via the Agent SDK — uses the user's
existing Claude Code login, no separate API key) to:
  1. Read the gap's commentary transcript.
  2. Pick up to 3 narrative events.
  3. For each event, choose a frame_time_seconds from a list of candidate
     timestamps we provide (so the visual + caption stay anchored together).
  4. Write a short title (2–4 words, ALL CAPS) and a 1-line caption
     (12–20 words).

The candidate-frame list comes from frames where OCR detected text — those
are usually broadcast graphics / onboards / replays / timing overlays,
i.e. the visually narrative-rich moments.

If the SDK call fails (network, auth, quota), we fall back to a
deterministic captioner so the pipeline still completes.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clarity import classify_event


_MODEL = "claude-sonnet-4-6"
_MAX_CANDIDATES = 24


@dataclass
class GapEvent:
    frame_time_seconds: float
    title: str
    caption: str


@dataclass
class GapSummary:
    events: list[GapEvent] = field(default_factory=list)
    fallback: bool = False


_SYSTEM_PROMPT = """\
You are an F1 race summariser writing bridge cards for a *focus-driver
highlight reel*. The reel's clips already cover every notable thing the
focus driver did. The bridge cards exist to fill the viewer in on what
the REST OF THE FIELD was doing while the focus driver was off-screen —
the "meanwhile, in the rest of the race…" beats.

HARD RULE — focus-driver EXCLUSION:
The subject of every event MUST be another driver, not the focus driver.
The focus driver's actions, pit stops, on-track battles, lap-time
milestones, defensive moves, etc. are ALREADY covered by the clips —
including them on a bridge card is redundant and wastes the viewer's
attention.

You MAY briefly reference the focus driver's race position as context
for an other-driver event — e.g. "Ricciardo closes within 0.3s of
fourth-placed Verstappen" — but the SUBJECT and ACTION of every event
must be another driver.

EXAMPLES (focus driver = Verstappen):
  ✓ "Hamilton extends his lead to 8.2s, managing ultrasoft tyres on lap 18"
  ✓ "Vettel pits for mediums on lap 22, emerging behind Bottas in P5"
  ✓ "Ricciardo struggles for grip in P4, falling 3s behind Verstappen"
  ✗ "Verstappen holds P3 with stable gap"            ← about focus, redundant
  ✗ "Verstappen pits for fresh tyres on lap 17"      ← about focus, redundant
  ✗ "Verstappen defends from Vettel into turn 10"    ← about focus, redundant

Look for `[no-★]` markers in the transcript: segments that DO NOT mention
the focus driver are usually the best anchor points for bridge events.
Segments marked `[★]` mention the focus driver — they're already covered
by the clips, so they're poor bridge material.

For each event you must:
- pick `frame_time_seconds` from the candidate list provided (a number
  copied verbatim from the candidates — never invent one). Prefer frames
  near a `[no-★]` segment.
- write `title`: 2–4 words, ALL CAPS, headline-y. Use FULL LAST NAMES
  (e.g. "HAMILTON EXTENDS LEAD", "VETTEL PITS", "RICCIARDO STRUGGLES").
  Never invent 3-letter abbreviations.
- write `caption`: 12–20 words, single sentence, declarative present
  tense, no quotation marks. The subject is a non-focus driver.

Choose at most 3 events that are *narratively distinct* and worth knowing
about for a viewer who only sees the focus driver's clips.

FACTS, NOT INVENTION:
Only state facts that are clearly grounded in the transcript or the OCR
text on the candidate frames. Do not invent specific positions, gap
times, lap numbers, tyre compounds, or sector deltas — if the transcript
doesn't say it, don't claim it.

If nothing other-driver-worthy happened in the gap (i.e. the gap is just
filler chatter, or every interesting beat is about the focus driver and
already in the clips), return {"events": []} — the bridge will be
silently skipped. A skipped bridge is better than a redundant one.

Reply with ONLY a single JSON object — no prose before or after, no code
fences. Schema:

{"events": [{"frame_time_seconds": <number>, "title": "...", "caption": "..."}]}
"""


def _build_user_prompt(*, focus_label: str, focus_aliases: list[str],
                       year: int, session: str,
                       gap_start: float, gap_end: float,
                       transcript_segments: list[dict],
                       candidate_frames: list[tuple[float, str]]) -> str:
    import re
    alias_re = re.compile(
        r"\b(" + "|".join(re.escape(a) for a in focus_aliases if a) + r")\b",
        re.IGNORECASE,
    ) if focus_aliases else None

    parts: list[str] = []
    parts.append(f"FOCUS DRIVER: {focus_label}")
    parts.append(f"SESSION: {year} {session.upper()}")
    parts.append(
        f"GAP: {_fmt_clock(gap_start)} → {_fmt_clock(gap_end)} "
        f"(duration {gap_end - gap_start:.0f}s)"
    )
    parts.append("")
    parts.append("CANDIDATE FRAMES (pick frame_time_seconds from this list):")
    for t, ocr_snippet in candidate_frames:
        snippet = ocr_snippet[:90].replace("\n", " ").strip() or "(no on-screen text)"
        parts.append(f"  {t:.2f}  {snippet}")
    parts.append("")
    parts.append("TRANSCRIPT (start_time | text). "
                 "[★] = mentions focus driver (already in clips — avoid). "
                 "[no-★] = no focus driver mention (good bridge material).")
    for seg in transcript_segments:
        text = seg.get("text", "").strip().replace("\n", " ")
        if not text:
            continue
        marker = "[★]    " if alias_re and alias_re.search(text) else "[no-★] "
        parts.append(f"  {marker} {seg['start']:.2f} | {text}")
    return "\n".join(parts)


def _fmt_clock(t: float) -> str:
    s = int(t)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _candidate_frames_for_gap(workdir: Path, gap_start: float,
                              gap_end: float, *,
                              max_n: int = _MAX_CANDIDATES) -> list[tuple[float, str]]:
    """Return [(time, ocr_text_joined), ...] for frames in the gap that
    have any OCR text, capped at max_n by even-spaced sampling."""
    ocr_path = workdir / "ocr.json"
    if not ocr_path.exists():
        return []
    ocr = json.loads(ocr_path.read_text())
    in_gap = [f for f in ocr
              if gap_start <= float(f.get("time", -1)) <= gap_end
              and f.get("tokens")]
    if not in_gap:
        return []
    if len(in_gap) <= max_n:
        chosen = in_gap
    else:
        step = len(in_gap) / max_n
        chosen = [in_gap[int(i * step)] for i in range(max_n)]
    out: list[tuple[float, str]] = []
    for f in chosen:
        text = " ".join(t.get("text", "") for t in f.get("tokens", [])).strip()
        out.append((float(f["time"]), text))
    return out


def _segments_in_gap(transcript: dict, gap_start: float, gap_end: float) -> list[dict]:
    out = []
    for seg in transcript.get("segments", []):
        if seg["end"] < gap_start:
            continue
        if seg["start"] > gap_end:
            break
        out.append(seg)
    return out


def _snap_to_candidate(t: float, candidates: list[tuple[float, str]]) -> float:
    if not candidates:
        return t
    return min(candidates, key=lambda c: abs(c[0] - t))[0]


def _parse_response(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    # Strip leading code fence (` ```json ` or ` ``` `) if present.
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        else:
            text = text[3:]
        text = text.lstrip()
    # Strip trailing fence if present.
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3].rstrip()
    # Try direct parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Last resort: extract the outermost balanced { ... } substring.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


async def _query_llm_once(system_prompt: str, user_prompt: str,
                          model: str, timeout_s: float) -> str | None:
    try:
        from claude_agent_sdk import (
            query, ClaudeAgentOptions, AssistantMessage, TextBlock,
        )
    except ImportError:
        return None

    chunks: list[str] = []

    async def _go() -> None:
        async for msg in query(
            prompt=user_prompt,
            options=ClaudeAgentOptions(
                system_prompt=system_prompt,
                tools=[],
                allowed_tools=[],
                max_turns=1,
                model=model,
                setting_sources=[],
                permission_mode="bypassPermissions",
            ),
        ):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)

    try:
        await asyncio.wait_for(_go(), timeout=timeout_s)
    except (asyncio.TimeoutError, Exception):
        return None
    return "".join(chunks) if chunks else None


async def _query_llm(system_prompt: str, user_prompt: str,
                     model: str = _MODEL, timeout_s: float = 90.0,
                     retries: int = 1) -> str | None:
    """One-shot text query via the Agent SDK with retry on empty/timeout.

    Returns response text or None after all retries exhausted.
    """
    for attempt in range(retries + 1):
        text = await _query_llm_once(system_prompt, user_prompt, model, timeout_s)
        if text and text.strip():
            return text
        # Brief backoff before retry — SDK transient failures usually clear
        # within a couple seconds.
        if attempt < retries:
            await asyncio.sleep(2.0)
    return None


def _deterministic_fallback(transcript_segments: list[dict],
                            candidate_frames: list[tuple[float, str]],
                            gap_start: float, gap_end: float,
                            n: int = 3) -> list[GapEvent]:
    """When the LLM is unavailable: pick up to N segments by sentence
    length × event-keyword score and snap to nearest candidate frame."""
    if not transcript_segments:
        return []
    scored = []
    for seg in transcript_segments:
        text = (seg.get("text") or "").strip()
        if len(text) < 30:
            continue
        ev = classify_event(text)
        score = (3.0 if ev != "mention" else 0.0) + min(len(text) / 50.0, 4.0)
        scored.append((score, seg, ev))
    scored.sort(key=lambda x: -x[0])
    chosen: list[tuple[dict, str]] = []
    for score, seg, ev in scored:
        too_close = any(abs(seg["start"] - c[0]["start"]) < 30.0 for c in chosen)
        if too_close:
            continue
        chosen.append((seg, ev))
        if len(chosen) >= n:
            break
    chosen.sort(key=lambda c: c[0]["start"])
    out: list[GapEvent] = []
    for seg, ev in chosen:
        t = _snap_to_candidate((seg["start"] + seg["end"]) / 2, candidate_frames)
        title = ev.upper().replace("_", " ") if ev != "mention" else "INTERLUDE"
        caption = (seg["text"] or "").strip().replace("\n", " ")
        if len(caption) > 140:
            caption = caption[:137].rsplit(" ", 1)[0] + "…"
        out.append(GapEvent(frame_time_seconds=t, title=title, caption=caption))
    return out


async def summarize_gap(*, workdir: Path, transcript: dict,
                        gap_start: float, gap_end: float,
                        focus_label: str, focus_aliases: list[str],
                        year: int, session: str,
                        max_events: int = 3) -> GapSummary:
    candidates = _candidate_frames_for_gap(workdir, gap_start, gap_end)
    segs = _segments_in_gap(transcript, gap_start, gap_end)
    if not segs and not candidates:
        return GapSummary()

    user_prompt = _build_user_prompt(
        focus_label=focus_label, focus_aliases=focus_aliases,
        year=year, session=session,
        gap_start=gap_start, gap_end=gap_end,
        transcript_segments=segs, candidate_frames=candidates,
    )

    text = await _query_llm(_SYSTEM_PROMPT, user_prompt)
    parsed = _parse_response(text) if text else None

    llm_succeeded = (
        parsed is not None
        and isinstance(parsed.get("events"), list)
    )
    if not llm_succeeded:
        # Only fall back when the LLM call itself failed (no response or
        # unparseable JSON). An LLM that returns {"events": []} is making a
        # legitimate "nothing focus-relevant in this gap" call — respect it.
        events = _deterministic_fallback(segs, candidates, gap_start, gap_end,
                                          n=max_events)
        return GapSummary(events=events, fallback=True)

    events: list[GapEvent] = []
    for raw in parsed["events"][:max_events]:
        try:
            t = float(raw["frame_time_seconds"])
            title = str(raw["title"]).strip().upper()
            caption = str(raw["caption"]).strip()
        except (KeyError, ValueError, TypeError):
            continue
        if not (gap_start - 5 <= t <= gap_end + 5):
            continue
        if not title or not caption:
            continue
        t_snapped = _snap_to_candidate(t, candidates) if candidates else t
        events.append(GapEvent(frame_time_seconds=t_snapped, title=title,
                               caption=caption))

    return GapSummary(events=events)


def summarize_gap_sync(**kwargs: Any) -> GapSummary:
    return asyncio.run(summarize_gap(**kwargs))
