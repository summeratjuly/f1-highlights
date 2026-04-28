"""Stage 5.5: build interstitial bridge cards between adjacent clips.

Each card is a 3-panel "magazine triptych" (layout G) summarising up to
3 narrative events from the gap. Captions + keyframe-anchor timestamps
are produced by the LLM via bridge_summarizer.py — falling back to a
deterministic captioner if the SDK call fails.

Bridges are skipped when:
  - the gap is shorter than min_gap (no story to tell),
  - the gap overlaps a replay span (the parent clip already covers it),
  - the gap ends BEFORE the race actually starts (no pre-grid filler).
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from bridge_summarizer import GapSummary, summarize_gap
from clarity import classify_event
from render_card import (
    DEFAULT_MANIFEST,
    DEFAULT_TEAM_COLORS,
    _hex_to_rgba,
    _load_team_color,
    render_triptych,
)
from shared import FRAMES_DIR_NAME


_CARD_ENCODE_VARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                      "-pix_fmt", "yuv420p"]
_CARD_ENCODE_AARGS = ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]

_RACE_START_KEYWORDS = (
    "lights out", "lights are out", "off they go", "off the line",
    "they're racing", "they are racing", "and we're racing",
    "the race is on", "racing for real", "race start",
)


def _frame_for_time(frames_dir: Path, target_time: float, fps: float) -> Path | None:
    i = max(1, int(round(target_time * fps + 0.5)))
    cand = frames_dir / f"f_{i:06d}.jpg"
    if cand.exists():
        return cand
    for delta in (1, -1, 2, -2, 3, -3):
        cand = frames_dir / f"f_{i + delta:06d}.jpg"
        if cand.exists():
            return cand
    files = sorted(frames_dir.glob("f_*.jpg"))
    if not files:
        return None
    return min(files, key=lambda p: abs(int(p.stem.split("_")[1]) - i))


def _gap_overlaps_replay(gap_start: float, gap_end: float,
                         replay_spans: list[tuple[float, float]]) -> bool:
    return any(s < gap_end and e > gap_start for s, e in replay_spans)


def _detect_race_start(clips: list[dict], transcript: dict) -> float:
    """Return the timestamp at which the race actually starts.

    Heuristics:
      1. First clip whose `event_type == "start"` — its start is the race start.
      2. First transcript segment containing a race-start keyword.
      3. Fallback: 0.0 (no filtering).
    """
    for c in clips:
        if c.get("event_type") == "start":
            return float(c["start"])
    for seg in transcript.get("segments", []):
        text = (seg.get("text") or "").lower()
        if any(kw in text for kw in _RACE_START_KEYWORDS):
            return float(seg["start"])
    return 0.0


def _fmt_clock(t: float) -> str:
    s = int(t)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def _encode_card_to_mp4(png_path: Path, mp4_path: Path, *,
                        duration: float, fps: float = 60.0,
                        size: tuple[int, int] = (1920, 1080)) -> None:
    w, h = size
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{duration}", "-r", f"{fps}",
        "-i", str(png_path),
        "-f", "lavfi", "-t", f"{duration}",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-vf", f"scale={w}:{h},format=yuv420p",
        *_CARD_ENCODE_VARGS,
        *_CARD_ENCODE_AARGS,
        "-shortest",
        "-loglevel", "error",
        str(mp4_path),
    ]
    subprocess.run(cmd, check=True)


async def _build_bridges_async(clips: list[dict], transcript: dict,
                               replay_spans: list[tuple[float, float]],
                               workdir: Path, *,
                               team_id: str | None,
                               year: int,
                               session: str,
                               focus_label: str,
                               focus_aliases: list[str],
                               ocr_fps: float,
                               min_gap: float,
                               duration: float,
                               manifest_path: Path) -> dict[int, Path]:
    if len(clips) < 2:
        return {}
    frames_dir = workdir / FRAMES_DIR_NAME
    if not frames_dir.exists():
        return {}

    bridge_dir = workdir / "bridges"
    bridge_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    accent_fallback = manifest.get("accent", {}).get("team_color_fallback", "#ff1801")
    accent = _load_team_color(team_id, year, accent_fallback)
    subtitle = f"{year} {session.upper()} · {focus_label.split('(')[0].strip().upper()}"

    race_start = _detect_race_start(clips, transcript)
    print(f"[f1-highlights]   race start detected at {_fmt_clock(race_start)}")

    bridges: dict[int, Path] = {}
    audit: list[dict] = []     # full per-gap record persisted to bridges.json
    fallbacks = 0

    for i in range(len(clips) - 1):
        a, b = clips[i], clips[i + 1]
        gap_start = float(a["end"])
        gap_end = float(b["start"])
        gap_dur = gap_end - gap_start
        gap_record: dict = {
            "after_clip_index": i,
            "gap_start": round(gap_start, 2),
            "gap_end": round(gap_end, 2),
            "gap_duration": round(gap_dur, 2),
        }

        if gap_dur < min_gap:
            gap_record["status"] = "skipped"
            gap_record["reason"] = f"gap shorter than min_gap ({min_gap}s)"
            audit.append(gap_record)
            continue
        if gap_end <= race_start + 1.0:
            gap_record["status"] = "skipped"
            gap_record["reason"] = "before race start"
            audit.append(gap_record)
            continue
        if _gap_overlaps_replay(gap_start, gap_end, replay_spans):
            gap_record["status"] = "skipped"
            gap_record["reason"] = "overlaps replay span"
            audit.append(gap_record)
            continue

        summary = await summarize_gap(
            workdir=workdir, transcript=transcript,
            gap_start=gap_start, gap_end=gap_end,
            focus_label=focus_label, focus_aliases=focus_aliases,
            year=year, session=session,
        )
        if summary.fallback:
            fallbacks += 1
            gap_record["status"] = "skipped"
            gap_record["reason"] = "LLM call failed or unparseable"
            audit.append(gap_record)
            continue
        if not summary.events:
            gap_record["status"] = "skipped"
            gap_record["reason"] = "LLM returned no focus-relevant events"
            audit.append(gap_record)
            continue

        keyframe_paths: list[Path] = []
        events_for_render: list[dict] = []
        events_audit: list[dict] = []
        for ev in summary.events:
            kf = _frame_for_time(frames_dir, ev.frame_time_seconds, ocr_fps)
            if kf is None:
                continue
            keyframe_paths.append(kf)
            events_for_render.append({
                "time": _fmt_clock(ev.frame_time_seconds),
                "title": ev.title,
                "caption": ev.caption,
            })
            events_audit.append({
                "frame_time_seconds": round(ev.frame_time_seconds, 2),
                "frame_clock": _fmt_clock(ev.frame_time_seconds),
                "frame_path": str(kf.relative_to(workdir)),
                "title": ev.title,
                "caption": ev.caption,
            })
        if not events_for_render:
            gap_record["status"] = "skipped"
            gap_record["reason"] = "no keyframe could be mapped to any event"
            audit.append(gap_record)
            continue

        header_title = f"{_fmt_clock(gap_start)} — {_fmt_clock(gap_end)}"
        png_path = bridge_dir / f"bridge_{i:03d}.png"
        mp4_path = bridge_dir / f"bridge_{i:03d}.mp4"

        if not png_path.exists():
            img = render_triptych(
                events_for_render,
                header_title=header_title,
                subtitle=subtitle,
                accent_hex=accent,
                keyframe_paths=keyframe_paths,
            )
            img.save(png_path, format="PNG")

        if not mp4_path.exists():
            _encode_card_to_mp4(png_path, mp4_path, duration=duration)

        bridges[i] = mp4_path
        gap_record.update({
            "status": "ok",
            "header_title": header_title,
            "png": str(png_path.relative_to(workdir)),
            "mp4": str(mp4_path.relative_to(workdir)),
            "events": events_audit,
        })
        audit.append(gap_record)

    # Persist the audit so captions / decisions are reviewable later.
    audit_doc = {
        "focus_label": focus_label,
        "focus_aliases": focus_aliases,
        "year": year,
        "session": session,
        "min_gap": min_gap,
        "duration_seconds_per_card": duration,
        "race_start": round(race_start, 2),
        "totals": {
            "candidate_gaps": len(clips) - 1,
            "bridges_generated": len(bridges),
            "bridges_skipped": (len(clips) - 1) - len(bridges),
            "skipped_due_to_llm_failure": fallbacks,
        },
        "gaps": audit,
    }
    (workdir / "bridges.json").write_text(json.dumps(audit_doc, indent=2))

    if fallbacks:
        print(f"[f1-highlights]   {fallbacks} bridges skipped (LLM unavailable or "
              f"returned unparseable output)")
    return bridges


def build_bridges(clips: list[dict], transcript: dict,
                  replay_spans: list[tuple[float, float]],
                  workdir: Path, *,
                  team_id: str | None,
                  year: int | None,
                  session: str,
                  focus_label: str = "",
                  focus_aliases: list[str] | None = None,
                  ocr_fps: float,
                  min_gap: float = 30.0,
                  duration: float = 10.0,
                  manifest_path: Path = DEFAULT_MANIFEST) -> dict[int, Path]:
    return asyncio.run(_build_bridges_async(
        clips=clips, transcript=transcript, replay_spans=replay_spans,
        workdir=workdir, team_id=team_id, year=year or 0,
        session=session,
        focus_label=focus_label,
        focus_aliases=focus_aliases or [],
        ocr_fps=ocr_fps, min_gap=min_gap, duration=duration,
        manifest_path=manifest_path,
    ))
