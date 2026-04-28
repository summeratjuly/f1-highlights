"""Pure PIL renderer for interstitial bridge cards.

Inputs are a keyframe JPG, an event label, a timestamp string, and a team
accent colour. Output is a PNG that matches the locked layout in
data/templates/bridge_card.json. The whole point is determinism: same
inputs + same manifest → byte-identical PNG every time.

CLI for standalone testing:
  python render_card.py FRAME.jpg OVERTAKE 1:23:45 RBR OUT.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DEFAULT_MANIFEST = DATA_DIR / "templates" / "bridge_card.json"
DEFAULT_TEAM_COLORS = DATA_DIR / "team_colors.json"


def _load_team_color(team_id: str | None, year: int | None,
                     fallback: str) -> str:
    if not team_id:
        return fallback
    data = json.loads(DEFAULT_TEAM_COLORS.read_text())
    if year is not None:
        override = data.get("overrides", {}).get(str(year), {}).get(team_id)
        if override:
            return override[0] if isinstance(override, list) else override
    default = data.get("default", {}).get(team_id)
    if default:
        return default[0] if isinstance(default, list) else default
    return fallback


def _hex_to_rgba(hex_color: str, default_alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), default_alpha)
    if len(h) == 8:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16))
    raise ValueError(f"bad hex color: {hex_color}")


def _font(spec: dict) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(spec["font"], size=spec["size"], index=spec.get("font_index", 0))


def _measure(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
             letter_spacing: int = 0) -> tuple[int, int]:
    if letter_spacing <= 0:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    width = 0
    height = 0
    for ch in text:
        bbox = draw.textbbox((0, 0), ch, font=font)
        width += (bbox[2] - bbox[0]) + letter_spacing
        height = max(height, bbox[3] - bbox[1])
    return max(0, width - letter_spacing), height


def _draw_text(draw: ImageDraw.ImageDraw, base: Image.Image, xy: tuple[int, int],
               text: str, font: ImageFont.FreeTypeFont, fill: tuple[int, int, int, int],
               *, letter_spacing: int = 0,
               shadow: dict | None = None) -> None:
    if shadow:
        sh_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        sh_draw = ImageDraw.Draw(sh_layer)
        sh_color = _hex_to_rgba(shadow["color"], default_alpha=180)
        sh_off = tuple(shadow.get("offset", [0, 3]))
        if letter_spacing > 0:
            x, y = xy[0] + sh_off[0], xy[1] + sh_off[1]
            for ch in text:
                sh_draw.text((x, y), ch, font=font, fill=sh_color)
                bbox = sh_draw.textbbox((0, 0), ch, font=font)
                x += (bbox[2] - bbox[0]) + letter_spacing
        else:
            sh_draw.text((xy[0] + sh_off[0], xy[1] + sh_off[1]), text, font=font, fill=sh_color)
        blur = shadow.get("blur", 0)
        if blur:
            sh_layer = sh_layer.filter(ImageFilter.GaussianBlur(radius=blur))
        base.alpha_composite(sh_layer)

    if letter_spacing > 0:
        x, y = xy
        for ch in text:
            draw.text((x, y), ch, font=font, fill=fill)
            bbox = draw.textbbox((0, 0), ch, font=font)
            x += (bbox[2] - bbox[0]) + letter_spacing
    else:
        draw.text(xy, text, font=font, fill=fill)


def _resolve_anchor(anchor: str, padding: tuple[int, int], canvas: tuple[int, int],
                    text_size: tuple[int, int]) -> tuple[int, int]:
    px, py = padding
    cw, ch = canvas
    tw, th = text_size
    if anchor == "top-left":
        return (px, py)
    if anchor == "top-right":
        return (cw - tw - px, py)
    if anchor == "bottom-left":
        return (px, ch - th - py)
    if anchor == "bottom-right":
        return (cw - tw - px, ch - th - py)
    raise ValueError(f"unknown anchor: {anchor}")


def _format_timestamp(seconds: float, fmt: str) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return fmt.format(h=h, mm=f"{m:02d}", ss=f"{sec:02d}")


def render_card(keyframe_path: Path, event_label: str, source_seconds: float,
                accent_hex: str, *, manifest_path: Path = DEFAULT_MANIFEST,
                session_label: str | None = None) -> Image.Image:
    manifest = json.loads(manifest_path.read_text())
    width, height = manifest["size"]

    keyframe = Image.open(keyframe_path).convert("RGB")
    keyframe = _cover_resize(keyframe, (width, height))
    canvas = keyframe.convert("RGBA")

    darken = manifest.get("darken_alpha", 0.0)
    if darken > 0:
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, int(255 * darken)))
        canvas.alpha_composite(overlay)

    accent = manifest.get("accent")
    if accent:
        bar = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bar)
        w = accent.get("width", 12)
        side = accent.get("side", "left")
        rgba = _hex_to_rgba(accent_hex)
        if side == "left":
            bd.rectangle([0, 0, w, height], fill=rgba)
        elif side == "right":
            bd.rectangle([width - w, 0, width, height], fill=rgba)
        elif side == "bottom":
            bd.rectangle([0, height - w, width, height], fill=rgba)
        elif side == "top":
            bd.rectangle([0, 0, width, w], fill=rgba)
        canvas.alpha_composite(bar)

    draw = ImageDraw.Draw(canvas)

    ev_spec = manifest["event"]
    text = event_label.upper() if ev_spec.get("uppercase") else event_label
    font = _font(ev_spec)
    ls = ev_spec.get("letter_spacing_px", 0)
    size = _measure(draw, text, font, letter_spacing=ls)
    pos = _resolve_anchor(ev_spec["anchor"], tuple(ev_spec["padding"]),
                          (width, height), size)
    _draw_text(draw, canvas, pos, text, font, _hex_to_rgba(ev_spec["color"]),
               letter_spacing=ls, shadow=ev_spec.get("shadow"))

    ts_spec = manifest["timestamp"]
    ts_text = _format_timestamp(source_seconds, ts_spec["format"])
    ts_font = _font(ts_spec)
    ts_size = _measure(draw, ts_text, ts_font)
    ts_pos = _resolve_anchor(ts_spec["anchor"], tuple(ts_spec["padding"]),
                             (width, height), ts_size)
    _draw_text(draw, canvas, ts_pos, ts_text, ts_font, _hex_to_rgba(ts_spec["color"]),
               shadow=ts_spec.get("shadow"))

    sl_spec = manifest.get("session_label")
    if sl_spec and session_label:
        sl_text = session_label.upper() if sl_spec.get("uppercase") else session_label
        sl_font = _font(sl_spec)
        sl_ls = sl_spec.get("letter_spacing_px", 0)
        sl_size = _measure(draw, sl_text, sl_font, letter_spacing=sl_ls)
        sl_pos = _resolve_anchor(sl_spec["anchor"], tuple(sl_spec["padding"]),
                                 (width, height), sl_size)
        _draw_text(draw, canvas, sl_pos, sl_text, sl_font,
                   _hex_to_rgba(sl_spec["color"]), letter_spacing=sl_ls)

    return canvas.convert("RGB")


def _cover_resize(img: Image.Image, target: tuple[int, int]) -> Image.Image:
    tw, th = target
    sw, sh = img.size
    src_aspect = sw / sh
    tgt_aspect = tw / th
    if src_aspect > tgt_aspect:
        new_h = th
        new_w = int(round(th * src_aspect))
    else:
        new_w = tw
        new_h = int(round(tw / src_aspect))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - tw) // 2
    top = (new_h - th) // 2
    return img.crop((left, top, left + tw, top + th))


_TRIPTYCH_DEFAULTS = {
    "size": [1920, 1080],
    "background": "#14161a",
    "header": {
        "height": 140,
        "bg": "#0c0e12",
        "title_padding": [80, 30],
        "title_font_index": 4,
        "title_size": 56,
        "subtitle_padding_right": 80,
        "subtitle_font_index": 10,
        "subtitle_size": 28,
        "subtitle_color": "#b4b4b4",
        "accent_height": 6,
    },
    "card": {
        "bg": "#20232a",
        "padding": 50,
        "image_height": 480,
        "accent_height": 5,
        "time_padding": [30, 35],
        "time_font_index": 10,
        "time_size": 22,
        "title_padding": [30, 75],
        "title_font_index": 4,
        "title_size": 38,
        "title_color": "#ffffff",
        "caption_padding": [30, 145],
        "caption_font_index": 0,
        "caption_size": 24,
        "caption_color": "#d7d7d7",
        "caption_line_spacing": 8,
    },
}


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int,
          draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        cand = " ".join(cur + [w])
        bbox = draw.textbbox((0, 0), cand, font=font)
        if bbox[2] - bbox[0] <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def render_triptych(events: list[dict], *, header_title: str,
                    subtitle: str, accent_hex: str,
                    keyframe_paths: list[Path],
                    cfg: dict | None = None) -> Image.Image:
    """Render a 3-panel summary card (layout G).

    `events` is a list of {"time": "H:MM:SS", "title": "...", "caption": "..."}
    `keyframe_paths` is a parallel list of frame JPGs.
    """
    cfg = cfg or _TRIPTYCH_DEFAULTS
    width, height = cfg["size"]
    canvas = Image.new("RGBA", (width, height), _hex_to_rgba(cfg["background"]))

    hd = cfg["header"]
    header_h = hd["height"]
    header_bar = Image.new("RGBA", (width, header_h), _hex_to_rgba(hd["bg"]))
    canvas.paste(header_bar, (0, 0))
    accent = Image.new("RGBA", (width, hd["accent_height"]), _hex_to_rgba(accent_hex))
    canvas.paste(accent, (0, header_h - hd["accent_height"]))

    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(
        "/System/Library/Fonts/HelveticaNeue.ttc",
        size=hd["title_size"], index=hd["title_font_index"],
    )
    draw.text(tuple(hd["title_padding"]), header_title.upper(),
              font=title_font, fill=_hex_to_rgba(accent_hex))

    sub_font = ImageFont.truetype(
        "/System/Library/Fonts/HelveticaNeue.ttc",
        size=hd["subtitle_size"], index=hd["subtitle_font_index"],
    )
    sub_bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
    sub_w = sub_bbox[2] - sub_bbox[0]
    draw.text((width - hd["subtitle_padding_right"] - sub_w, hd["title_padding"][1] + 20),
              subtitle, font=sub_font, fill=_hex_to_rgba(hd["subtitle_color"]))

    cd = cfg["card"]
    pad = cd["padding"]
    n = max(1, min(3, len(events)))
    card_w = (width - pad * (n + 1)) // n
    card_top = header_h + pad
    card_h = height - header_h - pad * 2

    image_h = cd["image_height"]
    title_font_card = ImageFont.truetype(
        "/System/Library/Fonts/HelveticaNeue.ttc",
        size=cd["title_size"], index=cd["title_font_index"],
    )
    time_font = ImageFont.truetype(
        "/System/Library/Fonts/HelveticaNeue.ttc",
        size=cd["time_size"], index=cd["time_font_index"],
    )
    cap_font = ImageFont.truetype(
        "/System/Library/Fonts/HelveticaNeue.ttc",
        size=cd["caption_size"], index=cd["caption_font_index"],
    )
    accent_rgba = _hex_to_rgba(accent_hex)

    for i, ev in enumerate(events[:n]):
        cx = pad + i * (card_w + pad)
        card = Image.new("RGBA", (card_w, card_h), _hex_to_rgba(cd["bg"]))

        kf = Image.open(keyframe_paths[i]).convert("RGB")
        kf = _cover_resize(kf, (card_w, image_h))
        card.paste(kf, (0, 0))
        ImageDraw.Draw(card).rectangle(
            [0, image_h, card_w, image_h + cd["accent_height"]], fill=accent_rgba,
        )

        cd_draw = ImageDraw.Draw(card)
        text_x_base = cd["time_padding"][0]
        text_y_base = image_h + cd["time_padding"][1]
        cd_draw.text((text_x_base, text_y_base), ev.get("time", ""),
                     font=time_font, fill=accent_rgba)
        cd_draw.text((cd["title_padding"][0], image_h + cd["title_padding"][1]),
                     ev.get("title", "").upper(),
                     font=title_font_card, fill=_hex_to_rgba(cd["title_color"]))

        cap_x = cd["caption_padding"][0]
        cap_y = image_h + cd["caption_padding"][1]
        cap_max_w = card_w - cap_x * 2
        for line in _wrap(ev.get("caption", ""), cap_font, cap_max_w, cd_draw):
            cd_draw.text((cap_x, cap_y), line,
                         font=cap_font, fill=_hex_to_rgba(cd["caption_color"]))
            cap_y += cap_font.size + cd["caption_line_spacing"]

        canvas.paste(card, (cx, card_top), card)

    return canvas.convert("RGB")


def main() -> None:
    if len(sys.argv) < 6:
        print("usage: render_card.py FRAME.jpg EVENT_LABEL SECONDS TEAM_ID OUT.png [YEAR] [SESSION_LABEL]",
              file=sys.stderr)
        sys.exit(2)
    frame_path = Path(sys.argv[1])
    event_label = sys.argv[2]
    seconds = float(sys.argv[3])
    team_id = sys.argv[4] or None
    out_path = Path(sys.argv[5])
    year = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None
    session_label = sys.argv[7] if len(sys.argv) > 7 else None

    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    accent = _load_team_color(
        team_id, year, manifest["accent"].get("team_color_fallback", "#ff1801")
    )
    img = render_card(frame_path, event_label, seconds, accent,
                      session_label=session_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
