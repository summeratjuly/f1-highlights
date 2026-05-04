"""Preprocess an OBS-style screen recording: keep only the time spans
where the broadcast was actually fullscreen, drop stretches where
browser chrome / taskbar / OS menu bar / loading screens were visible.

Heuristic: at 1fps, pipe BGR24 raw frames through ffmpeg, compute the
pixel-value standard deviation of the top 6 % and bottom 6 % of each
frame. Browser chrome and taskbars are mostly uniform colour bars
(stdev < ~20). A fullscreen broadcast reaches the edges with varied
content (sky, asphalt, logos, graphics — stdev > ~25). When BOTH edges
are varied, we call the frame fullscreen.

We then smooth, find contiguous runs ≥ min_run_sec, pad ±transition_pad,
and stream-copy concat them into the output file. Stream-copy snaps to
GOP boundaries (~2s drift), which is fine for this kind of trim and
keeps preprocessing fast.

CLI:
  python preprocess_recording.py INPUT OUTPUT [--sample-fps 1.0]
                                              [--min-run-sec 10]
                                              [--stdev-threshold 22]

Side effect: writes <output>.preprocess.json with the kept-segment
manifest (source_duration, kept_duration, segments, settings) so you
can audit what was dropped without re-running.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _probe_duration(video: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        text=True,
    )
    return float(out.strip())


def detect_fullscreen_segments(
    video: Path, *,
    sample_fps: float = 1.0,
    scale_w: int = 320,
    scale_h: int = 180,
    edge_height_pct: float = 0.06,
    stdev_threshold: float = 22.0,
    min_run_sec: float = 10.0,
    transition_pad_sec: float = 0.5,
) -> tuple[list[tuple[float, float]], float, list[dict]]:
    """Return (kept_segments, source_duration, per_sample_diagnostics).
    Each kept_segment is a (start_seconds, end_seconds) pair.
    """
    duration = _probe_duration(video)

    cmd = [
        "ffmpeg", "-i", str(video),
        "-vf", f"fps={sample_fps},scale={scale_w}:{scale_h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-loglevel", "error", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None

    frame_size = scale_w * scale_h * 3
    edge_h = max(1, int(scale_h * edge_height_pct))

    samples: list[dict] = []
    sample_interval = 1.0 / sample_fps
    t = 0.5 / sample_fps  # centre each sample in its window
    while True:
        buf = proc.stdout.read(frame_size)
        if len(buf) < frame_size:
            break
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(scale_h, scale_w, 3)
        top_std = float(frame[:edge_h].std())
        bot_std = float(frame[-edge_h:].std())
        is_fs = (top_std > stdev_threshold) and (bot_std > stdev_threshold)
        samples.append({
            "t": round(t, 2),
            "top_std": round(top_std, 2),
            "bot_std": round(bot_std, 2),
            "fullscreen": bool(is_fs),
        })
        t += sample_interval

    proc.stdout.close()
    proc.wait()

    if not samples:
        return [], duration, samples

    # Median-of-3 smoothing to suppress single-sample flicker.
    flags = [s["fullscreen"] for s in samples]
    smoothed = list(flags)
    for i in range(len(flags)):
        window = flags[max(0, i - 1): i + 2]
        smoothed[i] = sum(window) > len(window) / 2

    # Build runs of contiguous fullscreen=True samples.
    raw_runs: list[tuple[float, float]] = []
    in_run = False
    run_start = 0.0
    prev_t = samples[0]["t"]
    for s, fs in zip(samples, smoothed):
        if fs and not in_run:
            run_start = s["t"]
            in_run = True
        elif not fs and in_run:
            raw_runs.append((run_start, prev_t))
            in_run = False
        prev_t = s["t"]
    if in_run:
        raw_runs.append((run_start, samples[-1]["t"]))

    # Apply min-run filter and transition padding (toward the centre).
    kept: list[tuple[float, float]] = []
    for s, e in raw_runs:
        s2 = s + transition_pad_sec
        e2 = e - transition_pad_sec
        if e2 - s2 < min_run_sec:
            continue
        s2 = max(0.0, s2)
        e2 = min(duration, e2)
        kept.append((round(s2, 2), round(e2, 2)))

    return kept, duration, samples


def extract_kept_segments(
    video: Path, segments: list[tuple[float, float]],
    output: Path, work_dir: Path,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    seg_paths: list[Path] = []
    for i, (start, end) in enumerate(segments):
        seg_path = work_dir / f"keep_{i:04d}.mp4"
        if not seg_path.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{start}", "-to", f"{end}",
                 "-i", str(video), "-c", "copy",
                 "-avoid_negative_ts", "make_zero",
                 "-loglevel", "error", str(seg_path)],
                check=True,
            )
        seg_paths.append(seg_path)

    concat_file = work_dir / "concat.txt"
    concat_file.write_text("\n".join(f"file '{p}'" for p in seg_paths) + "\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-fflags", "+genpts",
         "-i", str(concat_file), "-c", "copy",
         "-avoid_negative_ts", "make_zero",
         "-movflags", "+faststart",
         "-loglevel", "error", str(output)],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Trim an OBS recording down to the fullscreen-broadcast portions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="Source recording (any ffmpeg-readable format).")
    ap.add_argument("output", type=Path, help="Where to write the trimmed copy.")
    ap.add_argument("--sample-fps", type=float, default=1.0,
                    help="Frames per second to sample for analysis.")
    ap.add_argument("--min-run-sec", type=float, default=10.0,
                    help="Drop fullscreen runs shorter than this.")
    ap.add_argument("--stdev-threshold", type=float, default=22.0,
                    help="Below this top/bottom edge stdev = looks like static UI chrome.")
    ap.add_argument("--transition-pad-sec", type=float, default=0.5,
                    help="Seconds trimmed off each side of every kept run "
                         "(absorbs the loading-frame at fullscreen toggle).")
    ap.add_argument("--keep-work-dir", action="store_true",
                    help="Don't delete the per-segment temp dir on success.")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"[preprocess] input not found: {args.input}")

    work_dir = args.output.with_suffix(".prep")

    print(f"[preprocess] analyzing {args.input.name}...")
    segments, duration, samples = detect_fullscreen_segments(
        args.input,
        sample_fps=args.sample_fps,
        stdev_threshold=args.stdev_threshold,
        min_run_sec=args.min_run_sec,
        transition_pad_sec=args.transition_pad_sec,
    )

    kept_total = sum(e - s for s, e in segments)
    pct = (kept_total / duration * 100.0) if duration > 0 else 0.0
    print(f"[preprocess]   source: {duration:.1f}s ({duration / 60:.1f}min)")
    print(f"[preprocess]   kept:   {kept_total:.1f}s ({kept_total / 60:.1f}min)  = {pct:.1f}% of source")
    print(f"[preprocess]   {len(segments)} fullscreen segment(s):")
    for s, e in segments:
        print(f"     {s:8.1f} → {e:8.1f}  ({e - s:6.1f}s)")

    if not segments:
        raise SystemExit("[preprocess] no fullscreen segments detected — "
                         "lower --stdev-threshold or check the source")

    print(f"[preprocess] extracting + concatenating → {args.output}")
    extract_kept_segments(args.input, segments, args.output, work_dir)

    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "source_duration": round(duration, 2),
        "kept_duration": round(kept_total, 2),
        "kept_pct": round(pct, 2),
        "settings": {
            "sample_fps": args.sample_fps,
            "stdev_threshold": args.stdev_threshold,
            "min_run_sec": args.min_run_sec,
            "transition_pad_sec": args.transition_pad_sec,
        },
        "segments": [
            {"start": s, "end": e, "duration": round(e - s, 2)}
            for s, e in segments
        ],
        # Down-sampled diagnostics: every 30th sample is plenty to debug.
        "diagnostics_every_30s": samples[::30],
    }
    manifest_path = args.output.with_suffix(".preprocess.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[preprocess]   manifest: {manifest_path}")

    if not args.keep_work_dir:
        for p in work_dir.glob("keep_*.mp4"):
            p.unlink()
        (work_dir / "concat.txt").unlink(missing_ok=True)
        try:
            work_dir.rmdir()
        except OSError:
            pass

    print("[preprocess] done.")


if __name__ == "__main__":
    main()
