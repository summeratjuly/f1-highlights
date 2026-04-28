"""Stage 3: sample frames, run OCR once, and extract name/team mentions
from broadcast graphics.

The OCR pass is expensive (easyocr over thousands of frames). To avoid
running it twice, this module writes two caches:

  ocr.json     — every OCR token with bounding-box centre per frame.
                 Consumed by analyze_livery.py for number matching.
  visual.json  — name/team alias hits on those OCR tokens.
                 Consumed by build_timeline.py.

Why OCR: F1 broadcasts put driver name graphics on-screen for every
onboard, replay, team radio, and timing-tower entry. Highest-precision
cheap-to-compute signal for "who is the broadcast discussing right now."
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from clarity import ocr_tokens_say_replay
from shared import (
    FRAMES_DIR_NAME,
    MIN_ALIAS_LEN_OCR,
    OCR_JSON_NAME,
    VISUAL_JSON_NAME,
    WEIGHT_OCR_DRIVER,
    WEIGHT_OCR_TEAM,
    compile_alias_re,
)


def _gpu_available() -> bool:
    try:
        import torch  # easyocr already depends on torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _sample_frames(video: Path, out_dir: Path, fps: float) -> list[tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "f_%06d.jpg"
    existing = sorted(out_dir.glob("f_*.jpg"))
    if not existing:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video),
             "-vf", f"fps={fps},scale=960:-2",
             "-qscale:v", "3", "-loglevel", "error", str(pattern)],
            check=True,
        )
        existing = sorted(out_dir.glob("f_*.jpg"))
    return [((i / fps) - (0.5 / fps), p) for i, p in enumerate(existing, start=1)]


def _run_ocr_all_frames(frames: list[tuple[float, Path]], workdir: Path) -> list[dict]:
    """Run easyocr once per frame with bounding boxes, cache to ocr.json."""
    out_path = workdir / OCR_JSON_NAME
    if out_path.exists():
        return json.loads(out_path.read_text())

    import easyocr
    reader = easyocr.Reader(["en"], gpu=_gpu_available(), verbose=False)

    results: list[dict] = []
    for ts, img_path in frames:
        try:
            raw = reader.readtext(str(img_path), detail=1, paragraph=False)
        except Exception:
            results.append({"time": round(ts, 2), "tokens": []})
            continue
        tokens = []
        for bbox, text, conf in raw:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            tokens.append({
                "cx": int(sum(xs) / len(xs)),
                "cy": int(sum(ys) / len(ys)),
                "text": text,
                "conf": round(float(conf), 3),
            })
        results.append({"time": round(ts, 2), "tokens": tokens})

    out_path.write_text(json.dumps(results))
    return results


def detect_visual_presence(video: Path, workdir: Path,
                           driver_aliases: list[str],
                           team_aliases: list[str],
                           fps: float = 1.0) -> list[dict]:
    out_json = workdir / VISUAL_JSON_NAME
    ocr_json = workdir / OCR_JSON_NAME
    if out_json.exists() and ocr_json.exists():
        return json.loads(out_json.read_text())

    frames_dir = workdir / FRAMES_DIR_NAME
    frames = _sample_frames(video, frames_dir, fps)
    ocr_data = _run_ocr_all_frames(frames, workdir)

    driver_re = compile_alias_re(driver_aliases, word_boundary=False, min_len=MIN_ALIAS_LEN_OCR)
    team_re = compile_alias_re(team_aliases, word_boundary=False, min_len=MIN_ALIAS_LEN_OCR)

    hits = []
    for frame_ocr in ocr_data:
        joined = " ".join(t["text"] for t in frame_ocr["tokens"]).lower()
        d = driver_re.search(joined)
        t = team_re.search(joined)
        if not (d or t):
            continue
        reasons = []
        if d:
            reasons.append(f"driver:{d.group(0)}")
        if t:
            reasons.append(f"team:{t.group(0)}")
        hits.append({
            "time": frame_ocr["time"],
            "reasons": reasons,
            "weight": WEIGHT_OCR_DRIVER if d else WEIGHT_OCR_TEAM,
            "ocr_preview": joined[:120],
        })

    out_json.write_text(json.dumps(hits))
    return hits


def find_replay_graphic_spans(workdir: Path,
                              span_pad: float = 1.5) -> list[tuple[float, float]]:
    """Read the cached ocr.json and return spans where a 'REPLAY' graphic
    is on screen. Each frame hit is widened by span_pad seconds on each
    side so adjacent hits naturally merge into one span.
    """
    ocr_path = workdir / OCR_JSON_NAME
    if not ocr_path.exists():
        return []
    ocr_data = json.loads(ocr_path.read_text())
    raw = [f["time"] for f in ocr_data if ocr_tokens_say_replay(f.get("tokens", []))]
    if not raw:
        return []
    raw.sort()
    spans: list[list[float]] = [[raw[0] - span_pad, raw[0] + span_pad]]
    for t in raw[1:]:
        if t - span_pad <= spans[-1][1]:
            spans[-1][1] = t + span_pad
        else:
            spans.append([t - span_pad, t + span_pad])
    return [(round(s, 2), round(e, 2)) for s, e in spans]
