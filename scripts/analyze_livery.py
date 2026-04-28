"""Stage 3.5: team-livery colour fingerprint + car-number detection.

Two signals, both cheap since they reuse frames and OCR tokens already
produced by analyze_video.py.

(a) Team-colour fingerprint
    Each frame is converted to HSV. UI strips (timing tower on the left,
    top title bar, bottom ticker) are masked out. The remaining pixels
    are matched against each target team's colour palette via hue+sat+val
    tolerances. If enough pixels match, the frame gets a team-level hit.

(b) Car-number match (colour-gated)
    We scan OCR tokens for pure 1-2 digit integers, reject ones inside
    the timing-tower strip (where race positions 1-20 live continuously),
    and for each remaining number that matches a target driver's number
    we sample the pixels around it. Only if the driver's team colour
    dominates that neighbourhood do we emit a driver-level hit - so
    "P4" showing next to the lap counter doesn't pretend to be Norris.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from shared import (
    FRAMES_DIR_NAME,
    HIT_COLOR,
    HIT_NUMBER,
    LIVERY_JSON_NAME,
    OCR_JSON_NAME,
    WEIGHT_LIVERY_COLOR_MAX,
    WEIGHT_LIVERY_COLOR_MIN,
    WEIGHT_LIVERY_COLOR_SCALE,
    WEIGHT_LIVERY_NUMBER,
)

# cv2 + numpy are heavy - defer so `--help` works before deps are installed.
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore


HUE_TOL = 8
S_MIN_BRIGHT = 90
V_MIN_BRIGHT = 50
V_MAX_BLACK = 45
S_MAX_BLACK = 60

COLOR_COVERAGE_THRESHOLD = 0.008
NUMBER_GATE_COVERAGE = 0.05
NUMBER_REGION_RADIUS = 60

UI_LEFT_FRAC = 0.15
UI_TOP_FRAC = 0.08
UI_BOTTOM_FRAC = 0.92

_NUMBER_TOKEN_RE = re.compile(r"\d{1,2}")


def _hex_to_hsv(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    bgr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def load_team_palette(palette_path: Path, year: int) -> dict[str, list[tuple[int, int, int]]]:
    raw = json.loads(palette_path.read_text())
    defaults = raw.get("default", {})
    overrides = raw.get("overrides", {}).get(str(year), {})
    merged_hex = {**defaults, **overrides}
    return {tid: [_hex_to_hsv(c) for c in colors] for tid, colors in merged_hex.items()}


def _color_mask(hsv_img, target: tuple[int, int, int]):
    """Return a uint8 {0,255} mask of pixels matching `target` (H,S,V)."""
    h_t, s_t, v_t = target
    if v_t < 60 and s_t < 80:
        return cv2.inRange(hsv_img, (0, 0, 0), (180, S_MAX_BLACK, V_MAX_BLACK))
    h_lo = (h_t - HUE_TOL) % 180
    h_hi = (h_t + HUE_TOL) % 180
    lo_s, lo_v, hi = S_MIN_BRIGHT, V_MIN_BRIGHT, 255
    if h_lo <= h_hi:
        return cv2.inRange(hsv_img, (h_lo, lo_s, lo_v), (h_hi, hi, hi))
    # Hue wraparound (e.g. red): match [h_lo..179] or [0..h_hi].
    m1 = cv2.inRange(hsv_img, (h_lo, lo_s, lo_v), (179, hi, hi))
    m2 = cv2.inRange(hsv_img, (0, lo_s, lo_v), (h_hi, hi, hi))
    return cv2.bitwise_or(m1, m2)


def _build_ui_mask(h: int, w: int):
    """255 in the active region, 0 in UI strips we want to ignore."""
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[:, :int(w * UI_LEFT_FRAC)] = 0
    mask[:int(h * UI_TOP_FRAC), :] = 0
    mask[int(h * UI_BOTTOM_FRAC):, :] = 0
    return mask


def _palette_coverage(hsv_img, team_hsvs, ui_mask, active_px: int) -> float:
    team_mask = None
    for target in team_hsvs:
        m = _color_mask(hsv_img, target)
        team_mask = m if team_mask is None else cv2.bitwise_or(team_mask, m)
    gated = cv2.bitwise_and(team_mask, ui_mask)
    return cv2.countNonZero(gated) / active_px


def _region_palette_coverage(hsv_img, cx: int, cy: int,
                             team_hsvs) -> float:
    h, w = hsv_img.shape[:2]
    x1 = max(0, cx - NUMBER_REGION_RADIUS)
    y1 = max(0, cy - NUMBER_REGION_RADIUS)
    x2 = min(w, cx + NUMBER_REGION_RADIUS)
    y2 = min(h, cy + NUMBER_REGION_RADIUS)
    region = hsv_img[y1:y2, x1:x2]
    size = region.shape[0] * region.shape[1]
    if size == 0:
        return 0.0
    best = 0.0
    for target in team_hsvs:
        m = _color_mask(region, target)
        cov = cv2.countNonZero(m) / size
        if cov > best:
            best = float(cov)
    return best


def detect_livery(
    workdir: Path,
    team_palette_path: Path,
    year: int,
    targets: list[dict],
) -> list[dict]:
    """targets: [{team_id, driver_id?, driver_number?}, ...].

    Emits hits keyed by `type` = HIT_COLOR (team) or HIT_NUMBER (driver).
    """
    out_path = workdir / LIVERY_JSON_NAME
    if out_path.exists():
        return json.loads(out_path.read_text())

    if cv2 is None or np is None:
        raise RuntimeError(
            "analyze_livery requires opencv-python + numpy. "
            "Run `pip install -r requirements.txt` or pass --no-livery."
        )

    frame_paths = sorted((workdir / FRAMES_DIR_NAME).glob("f_*.jpg"))
    if not frame_paths:
        out_path.write_text("[]")
        return []

    ocr_path = workdir / OCR_JSON_NAME
    ocr_data = json.loads(ocr_path.read_text()) if ocr_path.exists() else []

    palette = load_team_palette(team_palette_path, year)
    target_teams = {t["team_id"] for t in targets if t.get("team_id")}
    number_lookup: dict[int, tuple[str, str]] = {}
    for t in targets:
        n, did, tid = t.get("driver_number"), t.get("driver_id"), t.get("team_id")
        if n is not None and did and tid:
            number_lookup[int(n)] = (did, tid)

    ui_mask = None
    active_px = 0
    tower_x = 0
    hits: list[dict] = []

    for i, path in enumerate(frame_paths):
        frame_ocr = ocr_data[i] if i < len(ocr_data) else {"time": float(i), "tokens": []}
        ts = frame_ocr["time"]

        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if ui_mask is None:
            ui_mask = _build_ui_mask(h, w)
            active_px = int(cv2.countNonZero(ui_mask))
            tower_x = int(w * UI_LEFT_FRAC)
        if active_px == 0:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        for tid in target_teams:
            team_hsvs = palette.get(tid, [])
            if not team_hsvs:
                continue
            coverage = _palette_coverage(hsv, team_hsvs, ui_mask, active_px)
            if coverage < COLOR_COVERAGE_THRESHOLD:
                continue
            weight = max(WEIGHT_LIVERY_COLOR_MIN,
                         min(WEIGHT_LIVERY_COLOR_MAX,
                             coverage * WEIGHT_LIVERY_COLOR_SCALE))
            hits.append({
                "time": round(ts, 2),
                "type": HIT_COLOR,
                "team_id": tid,
                "coverage": round(coverage, 4),
                "weight": round(weight, 2),
            })

        if not number_lookup:
            continue
        for tok in frame_ocr.get("tokens", []):
            text = tok["text"].strip()
            if not _NUMBER_TOKEN_RE.fullmatch(text):
                continue
            num = int(text)
            target = number_lookup.get(num)
            if not target or tok["cx"] < tower_x:
                continue
            driver_id, team_id = target
            team_hsvs = palette.get(team_id, [])
            if not team_hsvs:
                continue
            cov = _region_palette_coverage(hsv, tok["cx"], tok["cy"], team_hsvs)
            if cov < NUMBER_GATE_COVERAGE:
                continue
            hits.append({
                "time": round(ts, 2),
                "type": HIT_NUMBER,
                "driver_id": driver_id,
                "team_id": team_id,
                "number": num,
                "color_coverage": round(cov, 4),
                "weight": WEIGHT_LIVERY_NUMBER,
            })

    out_path.write_text(json.dumps(hits))
    return hits
