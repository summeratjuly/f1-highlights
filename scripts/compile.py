"""Stage 6: cut clips from the source video and concatenate them.

Default mode: stream-copy (fast, lossless, hard-cut at GOP boundaries).
Smooth mode: re-encode with short crossfades for seamless transitions.

Cut filenames include a hash of (start, end) so the cache only hits when
timing matches — clip lists from different threshold/padding runs won't
silently reuse stale cuts.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


_REENCODE_ARGS = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    "-c:a", "aac", "-b:a", "160k",
]
_COPY_ARGS = ["-c", "copy", "-avoid_negative_ts", "make_zero"]


def _cut(video: Path, start: float, end: float, out: Path, *, reencode: bool) -> None:
    codec = _REENCODE_ARGS if reencode else _COPY_ARGS
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{start}", "-to", f"{end}", "-i", str(video),
         *codec, "-loglevel", "error", str(out)],
        check=True,
    )


def compile_highlights(video: Path, clips: list[dict], workdir: Path,
                       out_path: Path, smooth: bool = False,
                       crossfade: float = 0.4,
                       bridges: dict[int, Path] | None = None) -> None:
    bridges = bridges or {}
    # Mixed-source MP4s (cuts + bridges) don't survive stream-copy concat
    # cleanly — codec/timebase/SAR drift. Force re-encode when bridges
    # are present.
    must_reencode = bool(bridges) or smooth
    cut_dir = workdir / ("cuts_smooth" if must_reencode else "cuts")
    cut_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    for i, c in enumerate(clips):
        h = hashlib.md5(f"{c['start']:.2f}-{c['end']:.2f}".encode()).hexdigest()[:8]
        cp = cut_dir / f"c_{i:04d}_{h}.mp4"
        if not cp.exists():
            _cut(video, c["start"], c["end"], cp, reencode=must_reencode)
        clip_paths.append(cp)

    if not clip_paths:
        raise RuntimeError("no clips to compile")

    # Interleave bridges: bridges[i] inserts AFTER clip[i].
    interleaved: list[Path] = []
    for i, cp in enumerate(clip_paths):
        interleaved.append(cp)
        if i in bridges:
            interleaved.append(bridges[i])

    if smooth and crossfade > 0 and len(interleaved) > 1:
        # Build durations list aligned with interleaved order.
        durations: list[float] = []
        for i, c in enumerate(clips):
            durations.append(c["duration"])
            if i in bridges:
                # Probe the bridge for its actual duration.
                durations.append(_probe_duration(bridges[i]))
        _concat_with_xfade(interleaved, durations, out_path, crossfade)
    else:
        concat_file = workdir / "concat.txt"
        concat_file.write_text("\n".join(f"file '{p}'" for p in interleaved) + "\n")
        codec_args = (["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                       "-c:a", "aac", "-b:a", "160k", "-pix_fmt", "yuv420p"]
                      if must_reencode else
                      ["-c", "copy", "-avoid_negative_ts", "make_zero"])
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-fflags", "+genpts",
             "-i", str(concat_file),
             *codec_args,
             "-movflags", "+faststart",
             "-loglevel", "error", str(out_path)],
            check=True,
        )


def _probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def _concat_with_xfade(clip_paths: list[Path], durations: list[float],
                       out_path: Path, xfade: float) -> None:
    inputs = []
    for p in clip_paths:
        inputs += ["-i", str(p)]

    filter_parts = []
    current_v = "[0:v]"
    current_a = "[0:a]"
    running = durations[0]
    for i in range(1, len(clip_paths)):
        offset = max(0.0, running - xfade)
        nv, na = f"[v{i}]", f"[a{i}]"
        filter_parts.append(
            f"{current_v}[{i}:v]xfade=transition=fade:duration={xfade}:offset={offset}{nv}"
        )
        filter_parts.append(
            f"{current_a}[{i}:a]acrossfade=d={xfade}{na}"
        )
        current_v, current_a = nv, na
        running = running + durations[i] - xfade

    subprocess.run(
        ["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filter_parts),
         "-map", current_v, "-map", current_a,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
         "-c:a", "aac", "-b:a", "160k",
         "-movflags", "+faststart",
         "-loglevel", "error", str(out_path)],
        check=True,
    )
