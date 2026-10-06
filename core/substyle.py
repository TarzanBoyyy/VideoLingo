"""Subtitle-overlay .ass builder.

Reads a subdetect recipe.json (original-subtitle band + font size, in pixels)
and a translated .srt, then writes a complete .ass where:

  * layer 0 "Cover" events draw ONE slab per cue = union(original band rect,
    measured cue block), timed to the cue — the slab only exists while
    Vietnamese text sits on it; between cues the source frame shows through
    (no bare pre-cover slab). Slab style comes from SUB_TEMPLATES
    (VL_SUB_TEMPLATE): alpha, corner radius, text shadow;
  * layer 0 "Cover" events also black out each extra scene label (extras.json:
    OCR-detected on-screen text outside the band — name tags, titles);
  * layer 1 "SubVi" events render each translated cue bottom-centred on the
    band, ONE event per line at \\pos-controlled baseline pitch. Text is
    pre-measured with Pillow: font size solved so rendered ascent matches the
    measured Chinese glyph height, balanced wrap capped at VL_SUB_MAX_LINES
    lines, per-cue \\fs shrink when over budget, baseline pitch matching the
    source's measured line pitch (or template line-height factor);
  * layer 2 "SubEx" events render the Vietnamese translation of each scene
    label centred on its original bbox — same fit/per-line treatment.

Everything is pixel-literal: PlayResX/PlayResY equal the video dimensions, so
FontSize and \\pos values are real pixels.
"""

import itertools
import json
import os
import platform
import re
import unicodedata

__all__ = ["build_ass", "ass_filter_path", "parse_srt", "SUB_TEMPLATES"]

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


def _load_track(path):
    """intervals.json -> ([(start,end)], [change_t]); missing/unreadable -> ([],[]).
    `changes` = mốc nội dung band đổi (đổi câu/bắt/tắt) do subtrack đo."""
    if not path:
        return [], []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            items = data.get("intervals", [])
            chs = [float(c) for c in data.get("changes", [])]
        else:
            items, chs = data, []
        return [(float(i["start"]), float(i["end"])) for i in items], chs
    except Exception:
        return [], []


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


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def _band_items(recipe):
    """Detections nằm trong band sub (bbox = [x1,y1,x2,y2], có frame time t)."""
    band = recipe.get("band_px") or {}
    y0, y1 = float(band.get("y", 0)), float(band.get("y", 0)) + float(band.get("h", 0))
    out = []
    for d in recipe.get("detections") or []:
        bb = d.get("bbox") or []
        if len(bb) != 4:
            continue
        cy = (bb[1] + bb[3]) / 2
        if y0 - 20 <= cy <= y1 + 20:   # band theo cy, dung sai 20px
            out.append(d)
    return out


def _glyph_height(recipe):
    """Median chiều cao bbox glyph sub gốc — chiều cao chữ THẬT trên frame,
    chính xác hơn font_px (vốn là h·1.25 nhân lúc detect)."""
    hs = [d["bbox"][3] - d["bbox"][1] for d in _band_items(recipe)]
    return _median(hs)


def _band_rows(recipe):
    """{t: [tâm-y các HÀNG sub gốc trong band]} — gom cy gần nhau thành hàng."""
    items = _band_items(recipe)
    med_h = _median([d["bbox"][3] - d["bbox"][1] for d in items]) or 0
    if not med_h:
        return {}
    rows_by_t = {}
    for d in items:
        rows_by_t.setdefault(d.get("t", 0), []).append((d["bbox"][1] + d["bbox"][3]) / 2)
    out = {}
    for t, cys in rows_by_t.items():
        cys = sorted(cys)
        rows = [cys[0]]
        for cy in cys[1:]:
            if cy - rows[-1] > 0.6 * med_h:
                rows.append(cy)
        out[t] = rows
    return out


def _line_pitch(recipe):
    """Median khoảng cách tâm 2 hàng (baseline pitch) của sub gốc, đo trên các
    frame có ≥2 dòng. None nếu video chưa từng hiện sub 2 dòng — khi đó caller
    dùng line_height factor của template."""
    samples = []
    for rows in _band_rows(recipe).values():
        samples.extend(b - a for a, b in zip(rows, rows[1:]))
    return _median(samples)


def _src_rows(recipe):
    """Số dòng sub gốc nhiều nhất quan sát được — trần mềm cho budget dòng Việt
    (sub gốc 1 dòng thì Việt không nên vượt quá 2)."""
    rows = _band_rows(recipe)
    return max((len(r) for r in rows.values()), default=1)


def _cue_rows(band_rows, s, e):
    """Số hàng sub gốc quan sát được TRONG LÚC cue hiện (dung sai 0.15s) —
    wrap Việt theo đúng ngắt dòng này (sub Trung 1 hàng → ưu tiên 1 hàng Việt).
    Không có mẫu quanh cue → 0, caller fallback _src_rows."""
    return max((len(r) for t, r in band_rows.items()
                if s - 0.15 <= t <= e + 0.15), default=0)


def _split_long_cues(cues, changes, max_dur=2.5, min_piece=0.5):
    """Cue dài trải qua nhiều mốc đổi sub gốc → chữ Việt đứng im trong khi sub
    Trung đã sang câu mới (VD: WhisperX gộp 11s thoại = 1 cue). Chia cue tại các
    mốc `changes` nội khoa; phân bổ TỪ theo tỉ lệ thời lượng mỗi mảnh (xấp xỉ
    nhịp — post-hoc không dịch lại được). Mảnh <min_piece s gộp vào lân cận;
    mảnh rớt 0 từ thì bỏ mốc chia đó."""
    out = []
    for s, e, text in cues:
        words = unicodedata.normalize("NFC", text).split()
        pts = ([c for c in changes if s + 0.3 < c < e - 0.3]
               if e - s > max_dur else [])
        bounds = [s] + pts + [e]
        # Trần mảnh: mỗi mảnh ≥ ~3 từ thì mới đọc kịp — thừa mốc thì bỏ mốc
        # cạnh mảnh ngắn nhất (gộp lại), không xẻ vụn thành từng từ chớp.
        max_pieces = max(1, len(words) // 3)
        while len(bounds) - 1 > max_pieces:
            seg = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
            j = min(range(1, len(bounds) - 1),
                    key=lambda i: min(seg[i - 1], seg[i]))
            del bounds[j]
        # Mảnh <min_piece vẫn quá ngắn → gộp tiếp
        i = 1
        while i < len(bounds) - 1:
            if bounds[i + 1] - bounds[i] < min_piece:
                del bounds[i]
            else:
                i += 1
        if len(bounds) <= 2:
            out.append((s, e, text))
            continue
        cuts = [round((b - s) / (e - s) * len(words)) for b in bounds[1:]]
        pieces, prev_w = [], 0
        for i, cut in enumerate(cuts):
            sl = words[prev_w:cut]
            prev_w = cut
            if not sl:
                if pieces:
                    pieces[-1][1] = bounds[i + 1]   # mảnh 0 từ → gộp về trước
                continue
            pieces.append([bounds[i], bounds[i + 1], " ".join(sl)])
        if len(pieces) <= 1:
            out.append((s, e, text))
            continue
        pieces[0][0] = s                            # mảnh đầu luôn từ mốc gốc
        out.extend(tuple(p) for p in pieces)
    return out


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


# --- Layout theo model Hypit -------------------------------------------------
# caption-fine của Hypit không tự đo chữ: nó emit HTML/CSS và để Chromium wrap;
# kiểm soát tràn bằng budget ĐẾM ĐƯỢC (max-words-per-line / max-lines — vượt
# budget thì reject chứ không clip) và fitter shrink của text-flow
# (hyperframes/src/text.ts: đo → binary-search scale 20 vòng → áp, sàn
# minimum-scale). Bản port này thay Chromium bằng libass, đo bằng Pillow+RAQM
# trên đúng file font libass dùng: wrap cân bằng (minimax) trong vùng usable,
# shrink per-cue tới khi chui vào budget dòng, rồi vẽ 1 box ôm cả cue block
# (Hypit: cue box width:max-content + padding — không còn strip đen bậc thang
# theo từng dòng như BorderStyle=3 trần).

# Budget không set cứng: derive theo video (xem build_ass). Env chỉ là override;
# "0"/thiếu = auto. Mặc định suy ra từ font_px/band/frame của CHÍNH video đó.
_ENV_MAX_LINES = os.environ.get("VL_SUB_MAX_LINES", "")
_ENV_MIN_SCALE = os.environ.get("VL_SUB_MIN_SCALE", "")
_ENV_MAX_BLOCK_H = os.environ.get("VL_SUB_MAX_BLOCK_H", "")
_ENV_TEMPLATE = os.environ.get("VL_SUB_TEMPLATE", "")
_ENV_GLYPH_SCALE = os.environ.get("VL_SUB_GLYPH_SCALE", "")

# Template phụ đề — port trường style của Hypit caption recipe (size/line-height/
# padding/radius/background/fill/stroke/shadow). Slab vẫn BẮT BUỘC gần-đặc vì
# nhiệm vụ của nó là giấu sub Trung phía sau (Hypit dùng background:transparent
# vì nó không phải che gì). radius = hệ số × line_height; alpha: 0=đặc,255=tắt.
SUB_TEMPLATES = {
    "solid": {"desc": "khối đen đặc, góc vuông",
              "alpha": 0, "radius": 0.0, "shadow": 0, "lh": 1.15},
    "round": {"desc": "khối đen bo góc",
              "alpha": 0, "radius": 0.4, "shadow": 0, "lh": 1.12},
    "ghost": {"desc": "khối mờ ~88%, bo góc — slab mềm hơn",
              "alpha": 30, "radius": 0.4, "shadow": 0, "lh": 1.12},
    "echo":  {"desc": "khối đen bo góc + đổ bóng chữ",
              "alpha": 0, "radius": 0.4, "shadow": 3, "lh": 1.08},
}
DEFAULT_TEMPLATE = "round"


def _font_file(name):
    """File .ttf đúng với font libass sẽ resolve — đo sai font = wrap sai."""
    env = os.environ.get("VL_SUB_FONT_FILE")
    if env and os.path.isfile(env):
        return env
    cands = {
        "DejaVu Sans": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        ],
        "Arial Unicode MS": [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ],
        "Arial": [
            "/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        ],
    }
    for p in cands.get(name, []):
        if os.path.isfile(p):
            return p
    return None


class _Measurer:
    """Đo bề rộng/line-height bằng đúng font file (Pillow+RAQM). Thiếu PIL
    hoặc thiếu file font thì fallback ước lượng theo em — layout xấu đi chút
    chứ pipeline không chết."""

    def __init__(self, font_name):
        self._pil = None
        path = _font_file(font_name)
        if path:
            try:
                from PIL import ImageFont
                self._pil = (ImageFont, path)
            except Exception:
                self._pil = None
        self._cache = {}

    def _font(self, size):
        if not self._pil:
            return None
        key = max(8, int(round(size)))
        if key not in self._cache:
            try:
                ImageFont, path = self._pil
                self._cache[key] = ImageFont.truetype(
                    path, key, layout_engine=ImageFont.Layout.RAQM)
            except Exception:
                self._cache[key] = None
        return self._cache[key]

    def width(self, text, size):
        f = self._font(size)
        if f is not None:
            try:
                return float(f.getlength(text))
            except Exception:
                pass
        w = 0.0
        for ch in text:
            if ch == " ":
                w += 0.30
            elif ord(ch) >= 0x2E80:
                w += 1.0
            elif ch.isupper() or ch.isdigit():
                w += 0.65
            elif ch in ".,;:!?'»«":
                w += 0.30
            else:
                w += 0.50
        return w * size

    def space(self, size):
        f = self._font(size)
        if f is not None:
            try:
                return float(f.getlength(" "))
            except Exception:
                pass
        return 0.30 * size

    def line_h(self, size):
        f = self._font(size)
        if f is not None:
            try:
                a, d = f.getmetrics()
                return float(a + d)
            except Exception:
                pass
        return 1.2 * size

    def ink_h(self, size, ref="Việt"):
        """Chiều cao mực thật của chuỗi tham chiếu (đỉnh dấu -> đáy chữ). 'Việt'
        có cap + dấu trên + chấm dưới → đại diện đúng vùng hiển thị cảm nhận."""
        f = self._font(size)
        if f is not None:
            try:
                b = f.getbbox(ref)
                return float(b[3] - b[1])
            except Exception:
                pass
        return 0.98 * size

    def size_for_ink(self, target_h):
        """Cỡ font để chiều cao mực (ref 'Việt') == target_h px — tỉ lệ ~tuyến
        tính theo size nên đo ở 100 rồi scale là đủ chính xác cho libass."""
        h = self.ink_h(100)
        return max(8.0, target_h * 100.0 / h) if h > 0 else target_h


def _balanced(words, meas, size, usable_w, max_lines):
    """Chia words thành ≤max_lines dòng, minimax bề rộng dòng; None nếu không
    chia được. Ít dòng trước (n=1→max): câu ngắn giữ 1 dòng cỡ gốc, chỉ câu
    dài mới wrap — line break cân bằng nên 2 dòng không lệch hẳn nhau."""
    n_words = len(words)
    if not n_words:
        return []
    space = meas.space(size)
    if any(meas.width(wd, size) > usable_w for wd in words):
        return None   # một từ dài hơn cả dòng → chỉ shrink mới cứu được
    prefix = [0.0]
    for wd in words:
        prefix.append(prefix[-1] + meas.width(wd, size) + space)

    def lw(i, j):
        return prefix[j] - prefix[i] - space

    if lw(0, n_words) <= usable_w:
        return [" ".join(words)]
    for n in range(2, max_lines + 1):
        if n > n_words:
            break
        best = None
        for cuts in itertools.combinations(range(1, n_words), n - 1):
            pts = (0,) + cuts + (n_words,)
            widest = max(lw(pts[k], pts[k + 1]) for k in range(n))
            if best is None or widest < best[0]:
                best = (widest, pts)
        if best and best[0] <= usable_w:
            pts = best[1]
            return [" ".join(words[pts[k]:pts[k + 1]])
                    for k in range(n)]
    return None


def _fit_cue(text, meas, base_size, usable_w, want_lines, min_px, max_block_h,
             pitch_at=None, hard_cap=None):
    """Port text-flow shrink của Hypit (hyperframes/src/text.ts): thử layout ở
    size gốc; vượt budget (số dòng HOẶC chiều cao block) thì binary-search 20
    vòng lấy size lớn nhất còn vừa. Budget dòng leo từ `want_lines` (số dòng sub
    gốc) tới `hard_cap` — mỗi mức budget được shrink ĐẦY ĐỦ tới sàn `min_px`
    trước khi nới thêm dòng, nên sub gốc 1 dòng sẽ ép 1 dòng Việt hết cỡ, chỉ
    khi sàn cũng không vừa mới thêm dòng. Hụt cả trần → trả lines=None để
    libass tự wrap (cue quá dài, chấp nhận xấu hơn là mất chữ).

    `pitch_at(size)`: khoảng cách baseline giữa 2 dòng ở size đó — block cao
    (n-1)·pitch + line_h chứ không phải n·line_h (line-hệ thừa của dòng cuối
    không chiếm chỗ). Trả (lines, size): lines là list dòng đã ngắt sẵn.
    """
    if pitch_at is None:
        pitch_at = meas.line_h

    def block_h(ls, size):
        return (len(ls) - 1) * pitch_at(size) + meas.line_h(size)

    words = unicodedata.normalize("NFC", text).split()
    if not words:
        return None, base_size
    cap = max(want_lines, hard_cap if hard_cap else want_lines + 1)
    for budget in range(want_lines, cap + 1):

        def layout(size, budget=budget):
            ls = _balanced(words, meas, size, usable_w, budget)
            if ls is None or block_h(ls, size) > max_block_h:
                return None
            return ls

        if layout(base_size):
            return layout(base_size), base_size
        if layout(min_px) is None:
            continue                    # budget này sàn cũng không vừa → nới dòng
        lo, hi = min_px, float(base_size)
        for _ in range(20):
            mid = (lo + hi) / 2
            if layout(mid):
                lo = mid
            else:
                hi = mid
        return layout(lo), lo
    return None, min_px


def build_ass(srt_path, recipe_path, intervals_path, out_ass, font_scale=None,
              extras_path=None, template=None):
    """Build the subtitle-overlay .ass file; returns out_ass."""
    with open(recipe_path, encoding="utf-8") as f:
        recipe = json.load(f)

    tpl = SUB_TEMPLATES.get(template or _ENV_TEMPLATE or DEFAULT_TEMPLATE,
                            SUB_TEMPLATES[DEFAULT_TEMPLATE])
    band = recipe["band_px"]
    vid = recipe.get("video") or {}
    play_w = int(round(float(vid["width"])))
    play_h = int(round(float(vid["height"])))
    pad = float(recipe.get("pad_px", 0) or 0)

    meas = _Measurer(_font_name())

    # CHIỀU CAO CHỮ MATCH SUB TRUNG — nhưng tính từ BOX (band), không phải từ
    # glyph trần: slab đen đã che sub gốc rồi nên cỡ chữ Việt neo vào chiều cao
    # vùng box. size_for_ink giải cỡ sao cho mực chữ Việt ("Việt" = cap + dấu
    # trên + chấm dưới) cao đúng bằng box/glyph Trung đo được. Thiếu cả hai →
    # fallback font_px recipe (đã nhân VL_SUB_FONT_SCALE lúc detect).
    glyph_h = _glyph_height(recipe) or float(band.get("h") or 0) or None
    if glyph_h:
        gscale = float(_ENV_GLYPH_SCALE) if _ENV_GLYPH_SCALE else 1.0
        font_px = max(8, int(round(meas.size_for_ink(glyph_h * gscale *
                                                     (font_scale or 1.0)))))
    else:
        font_px = max(8, int(round(float(recipe.get("font_px", 24)) *
                                   (font_scale or 1.0))))

    # Cover rect: chặt đúng band gốc + pad.
    rx = max(0.0, float(band["x"]) - pad)
    rw = min(float(band["w"]) + 2 * pad, play_w - rx)
    rt = max(0.0, float(band["y"]) - pad)
    rh = min(float(band["h"]) + 2 * pad, play_h - rt)
    rect_bottom = rt + rh

    inner_pad = 8.0
    margin_v = int(play_h - rect_bottom + inner_pad)
    box_pad = int(round(font_px * 0.22))   # Outline = padding box khi BorderStyle=3

    # --- Budget derive theo video (Hypit: budget là authored per work) -------
    # Vùng wrap: 84%W, CANH THEO TÂM BAND chứ không phải tâm frame — band lệch
    # thì sub Việt + box bám đúng vùng sub gốc.
    usable_w = play_w * 0.84
    cue_cx = min(max(float(band["x"]) + float(band["w"]) / 2, usable_w / 2),
                 play_w - usable_w / 2)
    margin_l = int(round(cue_cx - usable_w / 2))
    margin_r = int(round(play_w - cue_cx - usable_w / 2))
    text_bottom = rect_bottom - inner_pad  # đáy khối text

    # ĐỘ CAO DÒNG MATCH SUB TRUNG: pitch đo được từ các frame 2-dòng của sub
    # gốc; video chỉ có sub 1 dòng (không đo được) thì dùng hệ số line-height
    # của template (Hypit recipe mẫu: 1.08). Sàn 0.85×(ascent+descent) — dưới
    # đó dấu/descender của 2 dòng chồng mực thô.
    src_pitch = _line_pitch(recipe)

    def pitch_at(size):
        p = float(src_pitch) if src_pitch else tpl["lh"] * size
        return max(p, meas.line_h(size) * 0.85)

    # Ngân sách diện tích: block Việt ≤ 15% chiều cao frame (VL_SUB_MAX_BLOCK_H
    # để override px tuyệt đối). Số dòng tối đa = bao nhiêu dòng cỡ font_px nhét
    # vừa ngân sách đó, NHƯNG không vượt (số dòng sub gốc + 1) — video sub gốc
    # chỉ 1 dòng thì Việt tối đa 2, không biến thành khối 3-4 dòng che giữa.
    max_block_h = float(_ENV_MAX_BLOCK_H) if _ENV_MAX_BLOCK_H else 0.15 * play_h
    auto_lines = max(2, min(3, int(max_block_h // meas.line_h(font_px))))
    src_rows = _src_rows(recipe)
    max_lines = (int(_ENV_MAX_LINES) if _ENV_MAX_LINES
                 else min(auto_lines, src_rows + 1))
    # Sàn shrink: không nhỏ hơn 55% cỡ gốc VÀ không dưới ngưỡng đọc được của
    # frame (~2.2%H ≈ 28px trên 1280) — scale mù sẽ ép font nhỏ thành vô đọc.
    min_scale = float(_ENV_MIN_SCALE) if _ENV_MIN_SCALE else 0.55
    min_px = max(8.0, font_px * min_scale, 0.022 * play_h)

    extras = _load_extras(extras_path)
    # intervals.json: không dùng cho timing cover, nhưng dùng để SNAP cue Việt
    # theo ranh giới sub gốc — `changes` là mốc đổi nội dung band do subtrack
    # đo (đổi câu/bắt/tắt). Việt hiện/tắt đúng nhịp Trung → không lộ sub gốc
    # chạy trước, cũng không đứng im che câu tiếp theo.
    orig_ivs, changes = _load_track(intervals_path)
    cues = _split_long_cues(parse_srt(srt_path), changes)

    starts = []
    prev = -1e9
    for s, _e, _t in cues:
        # Kéo start về mốc đổi sub gốc gần nhất trước cue (≤0.8s) NHƯNG sau
        # start cue trước — một mốc đổi chỉ thuộc về một cue, không thì 2 cue
        # liền nhau cùng kéo về một mốc → chồng sub. Cue đã bắt đầu đúng tại
        # một mốc đổi (mảnh của _split_long_cues) thì giữ nguyên. Fallback
        # interval start cho cache cũ chưa có `changes`.
        at_mark = any(abs(c - s) <= 0.05 for c in changes)
        cand = ([] if at_mark
                else [c for c in changes if prev < c < s and s - c <= 0.8])
        if cand:
            s = max(cand)
        else:
            for a, _b in orig_ivs:
                if prev < a < s and s - a <= 0.8:
                    s = a
        starts.append(s)
        prev = s
    ends = []
    for _s, e, _t in cues:
        nxt = [c for c in changes if e < c <= e + 0.6]
        ends.append(nxt[0] if nxt else e)   # Việt không tắt sớm lộ đuôi Trung
    for i in range(len(cues) - 1):
        if starts[i + 1] > starts[i]:      # handoff đúng mốc đổi câu: cue sau
            ends[i] = min(ends[i], starts[i + 1])   # kéo về sớm → cue trước nhường
    band_rows = _band_rows(recipe)

    kappa = 0.5523   # cubic-bezier xấp xỉ cung tròn cho bo góc

    def cover_draw(x: float, y: float, w: float, h: float,
                   radius: float = 0.0) -> str:
        """Rect/slab path ở gốc trái-trên; radius>0 → bo góc (Hypit radius)."""
        if radius >= 1.0:
            r = min(float(radius), w / 2, h / 2)
            k = kappa * r
            path = (
                f"m {r:.0f} 0 l {w - r:.0f} 0 "
                f"b {w - r + k:.0f} 0 {w:.0f} {r - k:.0f} {w:.0f} {r:.0f} "
                f"l {w:.0f} {h - r:.0f} "
                f"b {w:.0f} {h - r + k:.0f} {w - r + k:.0f} {h:.0f} "
                f"{w - r:.0f} {h:.0f} l {r:.0f} {h:.0f} "
                f"b {r - k:.0f} {h:.0f} 0 {h - r + k:.0f} 0 {h - r:.0f} "
                f"l 0 {r:.0f} b 0 {r - k:.0f} {r - k:.0f} 0 {r:.0f} 0"
            )
        else:
            path = f"m 0 0 l {w:.0f} 0 l {w:.0f} {h:.0f} l 0 {h:.0f}"
        return (
            f"{{\\an7\\pos({x:.0f},{y:.0f})\\p1\\c&H000000&"
            f"\\1a&H{int(tpl['alpha']):02X}&}}{path}{{\\p0}}"
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
        # SubVi: BorderStyle=3 → box BackColour ôm từng dòng (lưới an toàn dưới
        # union rect); MarginL/R lệch nhau để canh giữa quanh tâm band.
        f"Style: SubVi,{_font_name()},{font_px},&HFFFFFF,&HFFFFFF,&H000000,"
        f"&H000000,0,0,0,0,100,100,0,0,3,{box_pad},0,2,"
        f"{margin_l},{margin_r},{margin_v},1\n",
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
    shad = f"\\shad{int(tpl['shadow'])}" if tpl["shadow"] else ""
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
        # Fit như cue chính: ≤max_lines dòng trong vùng wrap, shrink nếu cần.
        xlines, xsize = _fit_cue(ex["vi"], meas, ex_font, wrap,
                                 max_lines, min_px, max_block_h, pitch_at)
        for s, e in ex["intervals"]:
            if xlines:
                # 1 slab union(cover nhãn gốc, block dịch) — bo góc template;
                # text emit per-line \\pos canh giữa cy, pitch giống cue band.
                xp = max(8.0, xsize * 0.15)
                xw = max(meas.width(ln, xsize) for ln in xlines)
                xn = len(xlines)
                xpitch = pitch_at(xsize)
                xlh = meas.line_h(xsize)
                xh = (xn - 1) * xpitch + xlh
                xbot = cy + xh / 2
                tx = max(0.0, px_c - xw / 2 - xp)
                ty = max(0.0, cy - xh / 2 - xp)
                tw = min(play_w - tx, xw + 2 * xp)
                th = min(play_h - ty, xh + 2 * xp)
                ux, uy = min(bx, tx), min(by, ty)
                ux2, uy2 = max(bx + bw, tx + tw), max(by + bh, ty + th)
                rect = cover_draw(ux, uy, ux2 - ux, uy2 - uy,
                                  tpl["radius"] * xlh)
            else:
                rect = cover_draw(bx, by, bw, bh,
                                  tpl["radius"] * meas.line_h(ex_font))
            lines.append(
                f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Cover,,0,0,0,,"
                f"{rect}\n"
            )
            if xlines:
                xfs = f"\\fs{int(round(xsize))}"
                for i, ln in enumerate(xlines):
                    y = xbot - (xn - 1 - i) * xpitch
                    lines.append(
                        f"Dialogue: 2,{_ass_time(s)},{_ass_time(e)},SubEx,,"
                        f"{ml},{mr},0,,{{\\an2\\pos({px_c:.0f},{y:.0f}){xfs}"
                        f"{shad}}}{_esc_text(ln)}\n"
                    )
            else:
                lines.append(
                    f"Dialogue: 2,{_ass_time(s)},{_ass_time(e)},SubEx,,"
                    f"{ml},{mr},0,,{{\\an5\\pos({px_c:.0f},{cy:.0f})"
                    f"\\fs{int(round(xsize))}{shad}}}{_esc_text(ex['vi'])}\n"
                )
    for ci, (s, e, text) in enumerate(cues):
        s = starts[ci]
        e = max(ends[ci], s + 0.15)        # guard: cue không thụt lùi
        # Layout đo-trước theo NGẮT DÒNG sub gốc: want = số hàng quan sát được
        # lúc cue hiện (fallback trần chung), hard_cap = max_lines. Mỗi dòng là
        # 1 event \\an2\\pos — pitch baseline do mình đặt (libass không cho set
        # line-spacing), match sub Trung.
        want = _cue_rows(band_rows, s, e) or src_rows
        fit_lines, size = _fit_cue(text, meas, font_px, usable_w,
                                   min(want, max_lines), min_px, max_block_h,
                                   pitch_at, hard_cap=max_lines)
        if not fit_lines:
            body = _esc_text(text)
            lines.append(
                f"Dialogue: 1,{_ass_time(s)},{_ass_time(e)},SubVi,,0,0,0,,"
                f"{{{shad}}}{body}\n"
            )
            continue
        n = len(fit_lines)
        pitch = pitch_at(size)
        lh = meas.line_h(size)
        fs = f"\\fs{int(round(size))}" if size < font_px - 0.6 else ""
        # Block MỌC XUỐNG DƯỚI band: dòng đầu nằm đúng chỗ sub gốc, dòng thừa
        # đổ xuống vùng dưới — không đẩy lên che giữa frame. Slab = union(band,
        # block) nên che liền từ band xuống hết block. Pad siết sát (0.10×size)
        # → box gọn, chữ nhìn to hơn trong cùng diện tích.
        top = text_bottom - lh                     # đỉnh box dòng đầu
        bot = text_bottom + (n - 1) * pitch        # đáy box dòng cuối
        pad_eff = size * 0.10
        widest = max(meas.width(ln, size) for ln in fit_lines)
        bx = max(0.0, min(play_w, cue_cx - widest / 2 - pad_eff))
        bw = min(play_w - bx, widest + 2 * pad_eff)
        by = max(0.0, top - pad_eff)
        bh = min(play_h - by, bot + pad_eff - by)
        ux, uy = min(rx, bx), min(rt, by)
        ux2, uy2 = max(rx + rw, bx + bw), max(rt + rh, by + bh)
        rect = cover_draw(ux, uy, ux2 - ux, uy2 - uy, tpl["radius"] * lh)
        lines.append(
            f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Cover,,0,0,0,,{rect}\n"
        )
        for i, ln in enumerate(fit_lines):
            y = text_bottom + i * pitch
            lines.append(
                f"Dialogue: 1,{_ass_time(s)},{_ass_time(e)},SubVi,,0,0,0,,"
                f"{{\\an2\\pos({cue_cx:.0f},{y:.0f}){fs}{shad}}}"
                f"{_esc_text(ln)}\n"
            )

    os.makedirs(os.path.dirname(out_ass) or ".", exist_ok=True)
    with open(out_ass, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return out_ass
