"""Stage 5: segment the relevance timeline into narrative clips that feel
*self-contained*. The viewer of any clip should instantly understand
what happened and why.

Pipeline:
  1. Find contiguous score >= threshold regions.
  2. Classify each region by event type (overtake / incident / pit /
     qualifying-lap / start / finish / mention) from commentary keywords.
  3. Pad each region with event-specific pre/post-roll (an overtake gets
     more setup, a passing mention barely any).
  4. Glue trailing replay regions to their parent moment.
  5. Force-include race-bookend windows (start, finish) for the target.
  6. Merge close regions; snap edges to sentence boundaries; extend
     past dangling conjunctions and orphan-pronoun sentence heads.
  7. Annotate each clip with event_type, transcript window, and any
     orphan-reference cues — the post-run review surfaces these for
     a model-layer pass (Claude or the user).
"""
from __future__ import annotations

import bisect

from clarity import (
    EVENT_PADDING,
    classify_event,
    ends_dangling,
    find_orphan_references,
    is_pronoun_only_head,
)


_REPLAY_BIND_GAP = 5.0       # replay must start within Xs of region end to glue
_BOOKEND_THRESHOLD = 0.5     # lower threshold inside bookend windows
_BOOKEND_START = 540.0       # first 9 min of a race counts as the start
_BOOKEND_END = 240.0         # last 4 min counts as the finish


_SENTENCE_END_CHARS = (".", "?", "!")
_TRAILING_STRIP = " \"'”’)"  # quotes, brackets


def _ends_sentence(text: str) -> bool:
    if not text:
        return False
    cleaned = text.rstrip(_TRAILING_STRIP).rstrip()
    return bool(cleaned) and cleaned[-1] in _SENTENCE_END_CHARS


def _sentence_boundaries(transcript: dict) -> tuple[list[float], list[float], list[dict]]:
    """Return TRUE sentence boundaries — Whisper segment edges that
    coincide with real sentence punctuation. Whisper splits on VAD
    silences which often fall mid-clause; snapping to those leaves clips
    starting/ending on comma fragments. Filtering by punctuation gives
    us readable cut points.

    If the model produced no punctuation (very rare with `small.en`), we
    fall back to raw segment boundaries so the pipeline keeps working.
    """
    segments = transcript["segments"]
    sentence_starts: list[float] = []
    sentence_ends: list[float] = []

    for i, seg in enumerate(segments):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        # Sentence END: this segment's text terminates a sentence.
        if _ends_sentence(text):
            sentence_ends.append(seg["end"])
        # Sentence START: first segment, OR previous segment ended a sentence.
        if i == 0:
            sentence_starts.append(seg["start"])
        else:
            prev_text = (segments[i - 1].get("text") or "").strip()
            if _ends_sentence(prev_text):
                sentence_starts.append(seg["start"])

    # Fallback if punctuation was sparse (defensive — small.en is reliable).
    if len(sentence_ends) < 5 or len(sentence_starts) < 5:
        sentence_starts = [s["start"] for s in segments]
        sentence_ends = [s["end"] for s in segments]

    return sentence_starts, sentence_ends, segments


def _snap_to_boundary(t: float, boundaries: list[float],
                      max_snap: float, direction: str) -> float:
    if not boundaries:
        return t
    i = bisect.bisect_left(boundaries, t)
    before = boundaries[i - 1] if i > 0 else None
    after = boundaries[i] if i < len(boundaries) else None
    if direction == "start":
        primary, fallback = before, after
    else:
        primary, fallback = after, before
    for cand in (primary, fallback):
        if cand is not None and abs(cand - t) <= max_snap:
            return cand
    return t


def _extend_past_dangling_end(end_t: float, segments: list[dict],
                              max_extra: float = 4.0) -> float:
    """If the segment ending at end_t finishes with a coordinating
    conjunction or filler word, extend to the next segment end. Cap the
    extension so a runaway monologue can't blow up the clip.
    """
    seg_ends = [s["end"] for s in segments]
    i = bisect.bisect_left(seg_ends, end_t - 0.05)
    while i < len(segments):
        seg = segments[i]
        if abs(seg["end"] - end_t) > 0.5:
            break
        if not ends_dangling(seg["text"]):
            break
        next_seg = segments[i + 1] if i + 1 < len(segments) else None
        if not next_seg or next_seg["end"] - end_t > max_extra:
            break
        end_t = next_seg["end"]
        i += 1
    return end_t


def _extend_past_pronoun_head(start_t: float, segments: list[dict],
                              max_extra: float = 4.0) -> float:
    """If the segment starting at start_t opens with an unresolved
    pronoun ("he was" / "that's why" ...), pull the start back to the
    previous segment so the antecedent is present.
    """
    seg_starts = [s["start"] for s in segments]
    i = bisect.bisect_left(seg_starts, start_t)
    while i < len(segments):
        seg = segments[i]
        if abs(seg["start"] - start_t) > 0.5:
            break
        if not is_pronoun_only_head(seg["text"]):
            break
        if i == 0:
            break
        prev = segments[i - 1]
        if start_t - prev["start"] > max_extra:
            break
        start_t = prev["start"]
        i -= 1
    return start_t


def _classify_region(start: float, end: float,
                     mentions: list[dict]) -> tuple[str, list[dict]]:
    in_region = [m for m in mentions if m["end"] >= start and m["start"] <= end]
    text_blob = " ".join(m["text"] for m in in_region)
    return classify_event(text_blob), in_region


def _bind_replays(regions: list[list[float]],
                  replay_spans: list[tuple[float, float]]) -> list[list[float]]:
    """Extend each region's end to swallow any replay span starting
    within _REPLAY_BIND_GAP seconds after the region. Replays inside a
    region are already absorbed by the threshold pass; this rule rescues
    replays that score below threshold on their own.
    """
    if not replay_spans:
        return regions
    spans = sorted(replay_spans)
    for r in regions:
        end = r[1]
        for s, e in spans:
            if s <= end + _REPLAY_BIND_GAP and e > end:
                r[1] = max(r[1], e)
                end = r[1]
    return regions


def _bookend_regions(scores: list[float], duration: float,
                     session: str, mentions: list[dict]) -> list[list[float]]:
    """For race sessions, force-include any target activity in the first
    and last few minutes — the grid/start and the finish/podium.
    """
    if session != "race" or duration <= 0:
        return []
    extra: list[list[float]] = []
    windows = [
        (0.0, min(_BOOKEND_START, duration)),
        (max(0.0, duration - _BOOKEND_END), duration),
    ]
    for ws, we in windows:
        for m in mentions:
            if m["end"] < ws or m["start"] > we:
                continue
            extra.append([max(ws, m["start"] - 2.0),
                          min(we, m["end"] + 3.0)])
    # Coalesce overlapping/adjacent extras.
    if not extra:
        return []
    extra.sort()
    out = [extra[0]]
    for r in extra[1:]:
        if r[0] - out[-1][1] <= 8.0:
            out[-1][1] = max(out[-1][1], r[1])
        else:
            out.append(r)
    return out


def build_clips(scores: list[float], transcript: dict,
                text_mentions: list[dict] | None = None,
                replay_spans: list[tuple[float, float]] | None = None,
                session: str = "race",
                threshold: float = 1.0,
                pre_roll: float = 2.0,
                post_roll: float = 3.0,
                merge_gap: float = 8.0,
                min_len: float = 6.0,
                snap_window: float = 5.0) -> list[dict]:
    text_mentions = text_mentions or []
    replay_spans = replay_spans or []
    duration = float(len(scores) - 1) if scores else 0.0

    # 1. Threshold sweep — contiguous score >= threshold regions.
    regions: list[list[float]] = []
    in_region = False
    start = 0.0
    for i, s in enumerate(scores):
        if s >= threshold and not in_region:
            start = float(i)
            in_region = True
        elif s < threshold and in_region:
            regions.append([start, float(i)])
            in_region = False
    if in_region:
        regions.append([start, float(len(scores))])

    # 1b. Bookend safety net — race-only.
    regions += _bookend_regions(scores, duration, session, text_mentions)
    regions.sort()

    if not regions:
        return []

    # 2 + 3. Per-region event classification → event-specific padding.
    region_events: list[str] = []
    for r in regions:
        event, _ = _classify_region(r[0], r[1], text_mentions)
        region_events.append(event)
        ev_pre, ev_post = EVENT_PADDING.get(event, EVENT_PADDING["mention"])
        # Take the larger of caller-provided defaults vs event padding —
        # never shrink below what the user (or session default) asked for.
        pre = max(pre_roll, ev_pre)
        post = max(post_roll, ev_post)
        r[0] = max(0.0, r[0] - pre)
        r[1] = min(duration, r[1] + post)

    # 4. Replay binding — pull trailing replays into their parent.
    regions = _bind_replays(regions, replay_spans)

    # 5. Merge close regions (with their event labels).
    merged: list[list[float]] = []
    merged_events: list[set[str]] = []
    for r, ev in zip(regions, region_events):
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], r[1])
            merged_events[-1].add(ev)
        else:
            merged.append(r)
            merged_events.append({ev})

    # 6. Snap to sentence boundaries; tighten dangling/pronoun edges.
    sent_starts, sent_ends, segments = _sentence_boundaries(transcript)
    clips: list[dict] = []
    for (s, e), evs in zip(merged, merged_events):
        snapped_s = _snap_to_boundary(s, sent_starts, snap_window, "start")
        snapped_e = _snap_to_boundary(e, sent_ends, snap_window, "end")
        snapped_s = _extend_past_pronoun_head(snapped_s, segments)
        snapped_e = _extend_past_dangling_end(snapped_e, segments)
        if snapped_e - snapped_s < min_len:
            continue

        # 7. Annotate clip with model-layer review hints.
        window = [
            seg for seg in segments
            if seg["end"] >= snapped_s and seg["start"] <= snapped_e
        ]
        text_blob = " ".join(seg["text"] for seg in window).strip()
        orphan_refs = find_orphan_references(text_blob)
        # Pick the highest-priority event among the merged regions.
        event_type = next(
            (e for e in ("incident", "finish", "start", "overtake",
                         "pit", "qualifying_lap", "mention")
             if e in evs),
            "mention",
        )
        clips.append({
            "start": round(snapped_s, 2),
            "end": round(snapped_e, 2),
            "duration": round(snapped_e - snapped_s, 2),
            "event_type": event_type,
            "orphan_refs": orphan_refs,
            "transcript": text_blob[:400],
        })
    return clips
