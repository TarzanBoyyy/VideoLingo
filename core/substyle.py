"""Subtitle-overlay .ass builder.

Reads a subdetect recipe.json (original-subtitle band + font size, in pixels)
and a translated .srt, then writes a complete .ass where:

  * layer 0 "Cover" events draw an opaque black rectangle over the original
    subtitle band, timed to the UNION of the original-subtitle intervals
    (intervals.json) and the translated cue times — so the source text is
    covered even where no translation exists;
  * layer 0 "Cover" events also black out each extra scene label (extras.json:
    OCR-detected on-screen text outside the band — name tags, titles);
  * layer 1 "SubVi" events render each translated cue bottom-centred on the
    band. BorderStyle=3 makes libass draw the backing box hugging the REAL
    rendered lines — no estimated-line-count guesswork, no giant empty slab;
  * layer 2 "SubEx" events render the Vietnamese translation of each scene
    label centred on its original bbox.

Everything is pixel-literal: PlayResX/PlayResY equal the video dimensions, so
FontSize and \\pos values are real pixels.
"""

import json
import os
import platform
import re

__all__ = ["build_ass", "ass_filter_path", "parse_srt"]

# 00:01:02,345 --> 00:01:05,678   (comma or dot millis, 1-3 digits)
_SRT_TS = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _to_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int((ms + "000")[:3]) / 1000.0


def _ass_time(t):
    """seconds -> ASS h:mm:ss.cc (centiseconds)."""
    cs = max(0, int(round(t * 100)))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def parse_srt(path):
    """Minimal SRT parse -> [(start_s, end_s, text)]; no external deps."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        raw = f.read().replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", raw):
        m = _SRT_TS.search(block)
        if not m:
            continue
        text = block[m.end():].strip()
        if not text:
            continue
        g = m.groups()
        cues.append((_to_seconds(*g[:4]), _to_seconds(*g[4:]), text))
    return cues


def _load_intervals(path):
    """intervals.json -> [(start, end)]; missing/unreadable -> []."""
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("intervals", []) if isinstance(data, dict) else data
        return [(float(i["start"]), float(i["end"])) for i in items]
    except Exception:
        return []


def _load_extras(path):
    """extras.json -> [{text, vi, bbox, intervals}]; missing/bad -> []."""
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", []) if isinstance(data, dict) else []
        out = []
        for it in items:
            bbox = it.get("bbox") or []
            vi = str(it.get("vi") or "").strip()
            ivs = [(float(i["start"]), float(i["end"]))
                   for i in (it.get("intervals") or [])
                   if i.get("end", 0) > i.get("start", 0)]
            if len(bbox) == 4 and vi and ivs:
                out.append({"vi": vi, "bbox": [float(v) for v in bbox],
                            "intervals": ivs})
        return out
    except Exception:
        return []


def _merge_ranges(ranges, gap=0.2):
    """Union of (start, end) ranges; merges overlaps and gaps < `gap` s."""
    merged = []
    for s, e in sorted(ranges):
        if e <= s:
            continue
        if merged and s - merged[-1][1] < gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _font_name():
    """Same platform logic as _7_sub_into_vid.TRANS_FONT_NAME."""
    sys = platform.system()
    if sys == "Linux":
        return "DejaVu Sans"
    if sys == "Darwin":
        return "Arial Unicode MS"
    return "Arial"


def _esc_text(text):
    """Cue text -> ASS text: strip braces (override-block chars), nl -> \\N."""
    text = text.replace("{", "").replace("}", "")
    return re.sub(r"\s*\n\s*", r"\\N", text)


def ass_filter_path(path):
    """Escape a filesystem path for use inside an ffmpeg filter graph."""
    return (
        path.replace("\\", "/")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
    )


def build_ass(srt_path, recipe_path, intervals_path, out_ass, font_scale=None,
              extras_path=None):
    """Build the subtitle-overlay .ass file; returns out_ass."""
    with open(recipe_path, encoding="utf-8") as f:
        recipe = json.load(f)

    band = recipe["band_px"]
    vid = recipe.get("video") or {}
    play_w = int(round(float(vid["width"])))
    play_h = int(round(float(vid["height"])))
    pad = float(recipe.get("pad_px", 0) or 0)

    # recipe.font_px đã nhân VL_SUB_FONT_SCALE lúc detect — KHÔNG nhân lại ở đây.
    font_px = max(8, int(round(float(recipe.get("font_px", 24)) * (font_scale or 1.0))))

    # Cover rect: chặt đúng band gốc + pad — KHÔNG vươn theo số dòng Việt nữa
    # (BorderStyle=3 của SubVi tự ôm text; cover chỉ lo giấu sub Trung).
    rx = max(0.0, float(band["x"]) - pad)
    rw = min(float(band["w"]) + 2 * pad, play_w - rx)
    rt = max(0.0, float(band["y"]) - pad)
    rh = min(float(band["h"]) + 2 * pad, play_h - rt)
    rect_bottom = rt + rh

    inner_pad = 8.0
    # Wrap rộng ~84%W để câu Việt (dài hơn Trung) gom được ≤2 dòng; box ôm nên
    # rộng thêm cũng không để lại vùng đen trống.
    margin_lr = int(play_w * 0.08)
    margin_v = int(play_h - rect_bottom + inner_pad)
    box_pad = int(round(font_px * 0.22))   # Outline = padding box khi BorderStyle=3

    cues = parse_srt(srt_path)
    band_times = _merge_ranges(
        _load_intervals(intervals_path) + [(s, e) for s, e, _ in cues])
    extras = _load_extras(extras_path)

    def cover_draw(x: float, y: float, w: float, h: float) -> str:
        return (
            f"{{\\an7\\pos({x:.0f},{y:.0f})\\p1\\c&H000000&\\1a&H00&}}"
            f"m 0 0 l {w:.0f} 0 l {w:.0f} {h:.0f} l 0 {h:.0f}{{\\p0}}"
        )

    lines = [
        "[Script Info]\n",
        "; generated by core/substyle.py (subtitle-overlay recipe)\n",
        "ScriptType: v4.00+\n",
        f"PlayResX: {play_w}\n",
        f"PlayResY: {play_h}\n",
        "WrapStyle: 0\n",
        "ScaledBorderAndShadow: yes\n",
        "YCbCr Matrix: None\n",
        "\n",
        "[V4+ Styles]\n",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n",
        # SubVi: BorderStyle=3 → box BackColour ôm đúng từng dòng render thật;
        # Outline = padding quanh text. Bottom-margin neo đáy text vào đáy band.
        f"Style: SubVi,{_font_name()},{font_px},&HFFFFFF,&HFFFFFF,&H000000,"
        f"&H000000,0,0,0,0,100,100,0,0,3,{box_pad},0,2,"
        f"{margin_lr},{margin_lr},{margin_v},1\n",
        # SubEx: chữ Việt thay nhãn gốc — box ôm, canh giữa bằng \pos từng event.
        f"Style: SubEx,{_font_name()},{font_px},&HFFFFFF,&HFFFFFF,&H000000,"
        "&H000000,0,0,0,0,100,100,0,0,3,8,0,5,0,0,0,1\n",
        f"Style: Cover,{_font_name()},20,&HFFFFFF,&HFFFFFF,&H000000,"
        "&H000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n",
        "\n",
        "[Events]\n",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n",
    ]
    for s, e in band_times:
        lines.append(
            f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Cover,,0,0,0,,"
            f"{cover_draw(rx, rt, rw, rh)}\n"
        )
    for ex in extras:
        x1, y1, x2, y2 = ex["bbox"]
        # Pad rộng theo cỡ chữ: nhãn drama hay có marker trang trí (`*`, khung)
        # nằm NGOÀI bbox chữ mà OCR trả về — pad hẹp để lộ vệt vàng lẹt xẹt.
        px = max(36.0, (y2 - y1) * 1.4)
        py = max(12.0, (y2 - y1) * 0.6)
        bx = max(0.0, x1 - px)
        by = max(0.0, y1 - py)
        bw = min(play_w - bx, (x2 - x1) + 2 * px)
        bh = min(play_h - by, (y2 - y1) + 2 * py)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ex_font = max(8, int(round((y2 - y1) * 1.1)))
        # Giữ chữ không wrap lung tung: wrap ≤ ~1.8× bề rộng nhãn gốc NHƯNG phải
        # đối xứng quanh tâm nhãn — nhãn sát mép mà wrap quá rộng thì text canh
        # giữa tại cx sẽ tràn khung. Pos cũng clamp để text luôn trong frame.
        inset = 16.0
        half = max(60.0, min(cx - inset, play_w - cx - inset))
        wrap = max(120.0, min(bw * 1.8, 2 * half, play_w - 2 * inset))
        px_c = min(max(cx, inset + wrap / 2), play_w - inset - wrap / 2)
        ml = int(max(0.0, px_c - wrap / 2))
        mr = int(max(0.0, play_w - px_c - wrap / 2))
        for s, e in ex["intervals"]:
            lines.append(
                f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Cover,,0,0,0,,"
                f"{cover_draw(bx, by, bw, bh)}\n"
            )
            lines.append(
                f"Dialogue: 2,{_ass_time(s)},{_ass_time(e)},SubEx,,{ml},{mr},0,,"
                f"{{\\an5\\pos({px_c:.0f},{cy:.0f})\\fs{ex_font}}}"
                f"{_esc_text(ex['vi'])}\n"
            )
    for s, e, text in cues:
        lines.append(
            f"Dialogue: 1,{_ass_time(s)},{_ass_time(e)},SubVi,,0,0,0,,"
            f"{_esc_text(text)}\n"
        )

    os.makedirs(os.path.dirname(out_ass) or ".", exist_ok=True)
    with open(out_ass, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return out_ass
