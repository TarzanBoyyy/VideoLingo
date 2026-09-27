"""Gán giọng (speaker) cho từng chữ và chia đoạn theo giọng — hàm thuần, không cần GPU.

Dùng chung cho bộ đo Giai đoạn 1 (`tools/speaker_bench/` của video-dashboard load file này
bằng đường dẫn, không qua `core/__init__.py` vốn import torch/demucs) và cho Giai đoạn 2.
Chỉ stdlib + numpy (pandas import lười trong `load_words_xlsx`).

Luật cốt lõi (docs/TASK-multi-speaker-dubbing.md mục 6.3): máy không chắc thì `None` —
KHÔNG BAO GIỜ mặc định người nói số 1. Người quyết định sau (trang duyệt).

Định dạng:
- word: `{"text", "start", "end", "punct"?}` theo đúng thứ tự dòng của cleaned_chunks.xlsx
  (với tiếng Trung mỗi word = 1 chữ Hán, dấu câu là dòng riêng).
- turn: `{"start", "end", "speaker"}`; overlap region: `{"start", "end", "speakers"}`.
"""
from __future__ import annotations

import math
import unicodedata
from typing import Iterable, NamedTuple, Sequence

import numpy as np

# Dấu kết câu dùng để tách đoạn — gồm cả dấu tiếng Trung (bản gốc whisper-diarization chỉ `.?!`).
SENTENCE_END = frozenset("。？！?!…")


def is_punct(text: str) -> bool:
    """Word chỉ gồm dấu câu/khoảng trắng/ký hiệu (Unicode P*, Z*, S*)."""
    t = (text or "").strip()
    return bool(t) and all(unicodedata.category(c)[0] in "PZS" for c in t)


def _ends_sentence(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and t[-1] in SENTENCE_END


def _dur(w: dict) -> float:
    return max(0.0, float(w["end"]) - float(w["start"]))


# --------------------------------------------------------------------------- turns


def normalize_labels(turns: Iterable[dict], *others: Iterable[dict]) -> tuple[list[dict], list[list[dict]], dict]:
    """Đổi nhãn thô của diarizer (SPEAKER_00, 0, spk1…) thành `S1, S2…` theo thứ tự xuất hiện.

    `others` (vd `exclusive_turns`) dùng chung bảng nhãn. Trả (turns, [others…], mapping thô→chuẩn).
    """
    turns = sorted((dict(t) for t in turns), key=lambda t: (float(t["start"]), float(t["end"])))
    mapping: dict[str, str] = {}
    for t in turns:
        raw = str(t["speaker"])
        if raw not in mapping:
            mapping[raw] = f"S{len(mapping) + 1}"
    out_others = []
    for group in others:
        g = sorted((dict(t) for t in group), key=lambda t: (float(t["start"]), float(t["end"])))
        for t in g:
            raw = str(t["speaker"])
            if raw not in mapping:
                mapping[raw] = f"S{len(mapping) + 1}"
            t["speaker"] = mapping[raw]
        out_others.append(g)
    for t in turns:
        t["speaker"] = mapping[str(t["speaker"])]
    return turns, out_others, mapping


def turns_overlaps(turns: Iterable[dict], min_dur: float = 0.0) -> list[dict]:
    """Vùng có ≥ 2 speaker KHÁC NHAU cùng nói (quét điểm đầu/cuối turn)."""
    events: list[tuple[float, int, str]] = []
    for t in turns:
        s, e = float(t["start"]), float(t["end"])
        if e > s:
            events.append((s, 1, str(t["speaker"])))
            events.append((e, -1, str(t["speaker"])))
    events.sort(key=lambda x: (x[0], x[1]))   # kết thúc (-1) trước bắt đầu (+1) tại cùng thời điểm
    active: dict[str, int] = {}
    out: list[dict] = []
    cur_start: float | None = None
    cur_spk: set[str] = set()
    for time, kind, spk in events:
        before = {k for k, v in active.items() if v > 0}
        active[spk] = active.get(spk, 0) + kind
        after = {k for k, v in active.items() if v > 0}
        if len(before) < 2 <= len(after):
            cur_start, cur_spk = time, set(after)
        elif len(before) >= 2 and len(after) >= 2:
            cur_spk |= after
        elif len(before) >= 2 > len(after) and cur_start is not None:
            if time - cur_start > min_dur:
                out.append({"start": round(cur_start, 3), "end": round(time, 3), "speakers": sorted(cur_spk)})
            cur_start, cur_spk = None, set()
    return out


# --------------------------------------------------------------------------- words


def assign_words(
    words: Sequence[dict],
    turns: Sequence[dict],
    exclusive_turns: Sequence[dict] | None = None,
    overlaps: Sequence[dict] | None = None,
    *,
    min_ratio: float = 0.5,
    zero_len_gap: float = 0.3,
    edge_collar: float | None = 0.3,
) -> list[dict]:
    """Gắn `speaker` cho từng chữ (không phải dấu câu) theo chồng lấn thời gian.

    Nguồn: https://github.com/CHANDRESH2002/ai-video-dubber@f036ae52e85d8cc6fdb5a9caa06530c8be3d0a08
    components/alignment.py:14-38 — MIT (speaker có tổng chồng lấn lớn nhất; không chồng lấn
    turn nào → `None`). Sửa: làm ở mức CHỮ thay vì mức segment; dùng `exclusive_turns` nếu có
    (mỗi lúc một người) còn không thì `turns`; tỉ lệ chồng lấn / độ dài chữ < `min_ratio` → `None`;
    chữ dài 0 → turn gần nhất chỉ khi cách ≤ `zero_len_gap` s; dấu câu nhận speaker của chữ trước;
    thêm `overlap_frac` (phần thời lượng chữ nằm trong vùng nói chồng). Không bao giờ mặc định S1.

    Luật "chữ mép" (`edge_collar`, None = tắt, đúng nguyên văn ngưỡng 0,5): WhisperX hay kéo dài
    chữ cuối câu vào khoảng lặng (đo trên vd-run1: trung vị 0,12 s/chữ nhưng có chữ 1–3 s) nên
    tỉ lệ < 0,5 dù chỉ MỘT giọng phủ chữ đó. Chỉ gán khi: đúng một speaker chồng lấn chữ VÀ không
    có turn nào của speaker khác trong ±`edge_collar` s quanh chữ (kể cả trong `turns` có chồng).
    Chữ ở ranh giới hai người vẫn là `None`. Chữ được gán kiểu này có `reasons: ["edge"]`.

    Trả bản sao từng word + `speaker`, `ratio`, `overlap_frac`, `reasons`.
    """
    base = list(exclusive_turns) if exclusive_turns else list(turns)
    bs = np.array([float(t["start"]) for t in base], dtype=float)
    be = np.array([float(t["end"]) for t in base], dtype=float)
    bspk = [str(t["speaker"]) for t in base]
    allt = list(base) + [t for t in turns if exclusive_turns]
    as_ = np.array([float(t["start"]) for t in allt], dtype=float)
    ae = np.array([float(t["end"]) for t in allt], dtype=float)
    aspk = np.array([str(t["speaker"]) for t in allt], dtype=object)
    ov = list(overlaps or [])
    os_ = np.array([float(o["start"]) for o in ov], dtype=float)
    oe = np.array([float(o["end"]) for o in ov], dtype=float)

    out: list[dict] = []
    prev_spk: str | None = None
    prev_ov = 0.0
    for w in words:
        d = dict(w)
        d["punct"] = bool(d.get("punct", is_punct(str(d.get("text", "")))))
        d["reasons"] = list(d.get("reasons") or [])
        if d["punct"]:
            d["speaker"], d["ratio"], d["overlap_frac"] = prev_spk, None, prev_ov
            out.append(d)
            continue
        ws, we = float(d["start"]), float(d["end"])
        dur = we - ws
        speaker: str | None = None
        ratio = 0.0
        if len(base):
            if dur <= 0:
                inside = np.nonzero((bs <= ws) & (be >= ws))[0]
                if len(inside):
                    j = int(inside[np.argmax(be[inside] - bs[inside])])
                    speaker, ratio = bspk[j], 1.0
                else:
                    dist = np.maximum(bs - ws, ws - be)
                    j = int(np.argmin(dist))
                    if dist[j] <= zero_len_gap:
                        speaker, ratio = bspk[j], 1.0
                    else:
                        d["reasons"].append("no_turn")
            else:
                inter = np.clip(np.minimum(be, we) - np.maximum(bs, ws), 0.0, None)
                per: dict[str, float] = {}
                for j in np.nonzero(inter > 0)[0]:
                    per[bspk[j]] = per.get(bspk[j], 0.0) + float(inter[j])
                if per:
                    best = max(sorted(per), key=lambda k: per[k])
                    ratio = min(1.0, per[best] / dur)
                    if ratio >= min_ratio:
                        speaker = best
                    elif edge_collar is not None and len(per) == 1 and not bool(np.any(
                            (aspk != best) & (as_ < we + edge_collar) & (ae > ws - edge_collar))):
                        speaker = best
                        d["reasons"].append("edge")
                    else:
                        d["reasons"].append("low_ratio")
                else:
                    d["reasons"].append("no_turn")
        else:
            d["reasons"].append("no_turn")
        if len(ov):
            if dur <= 0:
                ofrac = 1.0 if bool(np.any((os_ <= ws) & (oe >= ws))) else 0.0
            else:
                ofrac = float(np.clip(np.minimum(oe, we) - np.maximum(os_, ws), 0.0, None).sum()) / dur
        else:
            ofrac = 0.0
        d["speaker"], d["ratio"], d["overlap_frac"] = speaker, round(ratio, 4), round(min(1.0, ofrac), 4)
        d["_per"] = per if dur > 0 and len(base) else {}
        prev_spk, prev_ov = speaker, d["overlap_frac"]
        out.append(d)
    if edge_collar is not None:
        _punct_boundaries(out, bs, be, bspk)
    for d in out:
        d.pop("_per", None)
    _repropagate_punct(out)
    return out


def _punct_boundaries(out: list[dict], bs, be, bspk, max_gap: float = 0.1) -> None:
    """Pass 2 của luật "chữ mép": chữ `None` nằm ngay sau dấu câu (hoặc đầu bài) thì thuộc mệnh
    đề PHÍA SAU; nằm ngay trước dấu câu (hoặc cuối bài) thì thuộc mệnh đề PHÍA TRƯỚC — tiếng Trung
    đổi người nói gần như luôn ở dấu câu. Chỉ gán khi chữ có chồng lấn với turn của speaker đó
    (hoặc cách ≤ `max_gap` s). Chữ kẹp giữa hai dấu câu → vẫn `None`. Chỉ dùng nhãn của pass 1
    (không lan dây chuyền)."""
    labels = [d["speaker"] for d in out]
    n = len(out)
    for i, d in enumerate(out):
        if d["punct"] or labels[i] is not None:
            continue
        before_p = i == 0 or out[i - 1]["punct"]
        after_p = i == n - 1 or out[i + 1]["punct"]
        if before_p == after_p:
            continue
        j = i + 1 if before_p else i - 1
        if not (0 <= j < n) or out[j]["punct"] or labels[j] is None:
            continue
        cand = labels[j]
        per = d.get("_per") or {}
        if per and max(per, key=lambda k: per[k]) != cand:
            continue
        ok = per.get(cand, 0.0) > 0
        if not ok and len(bs):
            ws, we = float(d["start"]), float(d["end"])
            mask = np.array([k == cand for k in bspk])
            gaps = np.maximum(bs[mask] - we, ws - be[mask]) if mask.any() else np.array([np.inf])
            ok = float(gaps.min()) <= max_gap
        if ok:
            d["speaker"] = cand
            d["reasons"] = [r for r in d["reasons"] if r not in ("low_ratio", "no_turn")] + ["edge"]


def _repropagate_punct(words: list[dict]) -> None:
    prev_spk, prev_ov = None, 0.0
    for d in words:
        if d["punct"]:
            d["speaker"], d["overlap_frac"] = prev_spk, prev_ov
        else:
            prev_spk, prev_ov = d["speaker"], d.get("overlap_frac", 0.0)


def smooth_runs(
    words: Sequence[dict],
    *,
    max_words: int = 3,
    max_sec: float = 0.5,
    pause: float = 0.8,
    respect_boundaries: bool = True,
) -> list[dict]:
    """Làm mượt nhãn nhấp nháy: một dải chữ cùng nhãn có < `max_words` chữ (không tính dấu câu)
    VÀ < `max_sec` giây, kẹp giữa hai dải của CÙNG một speaker → đổi theo hàng xóm, ghi
    `reasons: ["smoothed"]` (+ `smoothed_from`).

    `respect_boundaries`: không làm mượt qua dấu kết câu hoặc khoảng nghỉ > `pause` — câu chen
    ngắn thật ("好。") giữa hai câu của người khác được giữ nguyên (docs mục 10: làm mượt có thể
    xoá câu chen ngắn thật). Chỉ nhìn cấu trúc dải GỐC (không lan truyền dây chuyền).
    """
    out = [dict(w, reasons=list(w.get("reasons") or [])) for w in words]
    idx = [i for i, w in enumerate(out) if not w["punct"]]
    runs: list[tuple[str | None, list[int]]] = []
    for i in idx:
        spk = out[i]["speaker"]
        if runs and runs[-1][0] == spk:
            runs[-1][1].append(i)
        else:
            runs.append((spk, [i]))

    def boundary_between(a_last: int, b_first: int) -> bool:
        if float(out[b_first]["start"]) - float(out[a_last]["end"]) > pause:
            return True
        return any(out[k]["punct"] and _ends_sentence(out[k]["text"]) for k in range(a_last + 1, b_first))

    for k in range(1, len(runs) - 1):
        (ps, pi), (cs, ci), (ns, ni) = runs[k - 1], runs[k], runs[k + 1]
        if ps is None or ps != ns or cs == ps:
            continue
        span = float(out[ci[-1]]["end"]) - float(out[ci[0]]["start"])
        if not (len(ci) < max_words and span < max_sec):
            continue
        if respect_boundaries and (boundary_between(pi[-1], ci[0]) or boundary_between(ci[-1], ni[0])):
            continue
        for i in ci:
            out[i]["smoothed_from"] = out[i]["speaker"]
            out[i]["speaker"] = ps
            out[i]["reasons"].append("smoothed")
    _repropagate_punct(out)
    return out


# --------------------------------------------------------------------------- segments


class CertainRule(NamedTuple):
    """Ngưỡng "chắc" (khởi điểm theo docs mục 6.3; evaluate.py hiệu chỉnh ở 6.6).

    NamedTuple (không dùng dataclass) để load được bằng spec_from_file_location mà không cần
    đăng ký sys.modules."""

    min_coverage: float = 0.8
    max_overlap: float = 0.2
    min_dur: float = 0.6
    allow_smoothed: bool = False
    min_purity: float = 0.0
    allow_edge: bool = True


def is_certain(seg: dict, rule: CertainRule = CertainRule()) -> bool:
    return (
        seg.get("speaker") is not None
        and float(seg.get("coverage", 0.0)) >= rule.min_coverage
        and float(seg.get("overlap_frac", 1.0)) <= rule.max_overlap
        and float(seg["end"]) - float(seg["start"]) >= rule.min_dur
        and (rule.allow_smoothed or "smoothed" not in (seg.get("reasons") or []))
        and float(seg.get("purity", 0.0)) >= rule.min_purity
        and (rule.allow_edge or "edge" not in (seg.get("reasons") or []))
    )


def _join(texts: list[str]) -> str:
    out = ""
    for t in texts:
        if out and t and out[-1].isascii() and out[-1].isalnum() and t[0].isascii() and t[0].isalnum():
            out += " "
        out += t
    return out


def _group_words(words: Sequence[dict], pause: float, max_len: float) -> list[list[int]]:
    """Chia chỉ số chữ thành nhóm: (a) đổi speaker, (b) sau dấu kết câu, (c) nghỉ > pause,
    (d) dài > max_len. Nhóm chỉ có dấu câu được dán vào nhóm trước (hoặc sau nếu đứng đầu)."""
    groups: list[list[int]] = []
    cur: list[int] = []
    cur_spk: str | None = None
    has_word = False
    seg_start = 0.0
    for i, w in enumerate(words):
        if cur and not w["punct"]:
            last_end = float(words[cur[-1]]["end"])
            if (has_word and w["speaker"] != cur_spk) \
                    or float(w["start"]) - last_end > pause \
                    or (has_word and float(w["end"]) - seg_start > max_len):
                groups.append(cur)
                cur, has_word = [], False
        cur.append(i)
        if not w["punct"] and not has_word:
            has_word, cur_spk, seg_start = True, w["speaker"], float(w["start"])
        if w["punct"] and _ends_sentence(w["text"]):
            groups.append(cur)
            cur, has_word = [], False
    if cur:
        groups.append(cur)
    merged: list[list[int]] = []
    pending: list[int] = []
    for g in groups:
        if all(words[i]["punct"] for i in g):
            if merged:
                merged[-1].extend(g)
            else:
                pending.extend(g)
            continue
        merged.append(pending + g)
        pending = []
    if pending:
        merged.append(pending)   # toàn dấu câu, không có chữ nào
    return merged


def build_segments(
    words: Sequence[dict],
    turns: Sequence[dict] | None = None,
    *,
    pause: float = 0.8,
    max_len: float = 15.0,
    rule: CertainRule = CertainRule(),
    id_prefix: str = "seg_",
) -> list[dict]:
    """Chia chữ đã gán giọng thành đoạn + đo purity/coverage/overlap_frac/status/certain.

    Nguồn: https://github.com/TachikomaRed/tachidubb@528e937adc14f07e78f2fa4c2e6367d7b9f77832
    pipeline/transcriber.py:188-308 — MIT (`_group_into_tts_segments`: gom chữ, tách khi đổi
    speaker / dấu kết câu / nghỉ / quá dài) và ý tưởng tách câu khi đổi speaker của
    https://github.com/MahmoudAshraf97/whisper-diarization@c6614d717764592c4261f2590c1eb29e87948c3e
    helpers.py:360-385 (`get_sentences_speaker_mapping`) — BSD-2-Clause.
    Sửa: nhãn `""` → `None` và `None` là một giá trị riêng (chữ không rõ giọng không bị nuốt vào
    đoạn của người trước — tachidubb bỏ qua đổi speaker khi nhãn rỗng); thêm dấu kết câu tiếng
    Trung (bản gốc chỉ `.?!`, helpers.py:275); không kéo dài `end` tới min_duration; không port
    "relabel theo đa số" (`get_realigned_ws_mapping_with_punctuation`) vì nó xoá câu chen ngắn thật.

    purity   = thời lượng `turns` của speaker đoạn trong [start, end] / tổng thời lượng mọi turn
               trong khoảng đó (không có `turns` → 1,0 nếu có speaker).
    coverage = thời lượng chữ có nhãn trực tiếp (không tính chữ được làm mượt) / tổng thời lượng
               chữ của đoạn. Không chia cho [start, end]: WhisperX gán cả thời lượng cho dòng dấu
               câu nên mẫu số theo khung thời gian bị phồng (đo trên vd-run1: câu gán đủ chữ vẫn
               chỉ đạt ~0,75).
    """
    ts = np.array([float(t["start"]) for t in turns or []], dtype=float)
    te = np.array([float(t["end"]) for t in turns or []], dtype=float)
    tspk = [str(t["speaker"]) for t in turns or []]
    segs: list[dict] = []
    for n, g in enumerate(_group_words(words, pause, max_len), start=1):
        nonp = [words[i] for i in g if not words[i]["punct"]]
        if not nonp:
            continue
        speaker = nonp[0]["speaker"]
        start = float(nonp[0]["start"])
        end = max(float(w["end"]) for w in nonp)
        dur = max(0.0, end - start)
        reasons: list[str] = []
        smoothed = any("smoothed" in (w.get("reasons") or []) for w in nonp)
        labeled = sum(_dur(w) for w in nonp if w["speaker"] is not None and "smoothed" not in (w.get("reasons") or []))
        word_time = sum(_dur(w) for w in nonp)
        if speaker is None:
            coverage = 0.0
        elif word_time > 0:
            coverage = min(1.0, labeled / word_time)
        else:
            coverage = 0.0 if smoothed else 1.0
        wd = [max(_dur(w), 1e-3) for w in nonp]
        overlap_frac = sum(float(w.get("overlap_frac") or 0.0) * x for w, x in zip(nonp, wd)) / sum(wd)
        if speaker is None:
            purity = 0.0
        elif len(ts) and dur > 0:
            inter = np.clip(np.minimum(te, end) - np.maximum(ts, start), 0.0, None)
            total = float(inter.sum())
            mine = float(sum(inter[j] for j in range(len(tspk)) if tspk[j] == speaker))
            purity = mine / total if total > 0 else 0.0
        else:
            purity = 1.0
        ratios = [float(w["ratio"]) for w in nonp if w.get("ratio") is not None]
        mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0

        if speaker is None:
            reasons.append("no_turn")
        if speaker is not None and coverage < rule.min_coverage:
            reasons.append("low_coverage")
        if overlap_frac > rule.max_overlap:
            reasons.append("overlap")
        if speaker is not None and purity < max(rule.min_purity, 0.5):
            reasons.append("low_purity")
        if dur < rule.min_dur:
            reasons.append("short")
        if smoothed:
            reasons.append("smoothed")
        if any("edge" in (w.get("reasons") or []) for w in nonp):
            reasons.append("edge")      # thông tin — không chặn "chắc" trừ khi rule.allow_edge=False

        if speaker is None:
            status = "no_turn"
        elif "overlap" in reasons:
            status = "overlap"
        elif "low_coverage" in reasons or "low_purity" in reasons:
            status = "low_purity"
        elif "short" in reasons:
            status = "short"
        else:
            status = "assigned"
        seg = {
            "id": f"{id_prefix}{n:04d}",
            "start": round(start, 3), "end": round(end, 3),
            "word_start": g[0], "word_end": g[-1],
            "text": _join([str(words[i]["text"]) for i in g]),
            "speaker": speaker, "status": status,
            "purity": round(purity, 4), "coverage": round(coverage, 4), "overlap_frac": round(overlap_frac, 4),
            "reasons": reasons, "disagree_arms": [],
        }
        seg["certain"] = is_certain(seg, rule)
        seg["confidence"] = round(mean_ratio * coverage * (1.0 - overlap_frac) * purity, 4) if speaker else 0.0
        segs.append(seg)
    # Đánh số lại liên tục sau khi bỏ nhóm rỗng.
    for n, s in enumerate(segs, start=1):
        s["id"] = f"{id_prefix}{n:04d}"
    return segs


# --------------------------------------------------------------------------- so khớp giữa các arm


def linear_sum_assignment(cost: Sequence[Sequence[float]]) -> tuple[list[int], list[int]]:
    """Hungarian (Kuhn–Munkres, O(n²m)) — tối thiểu tổng cost; ma trận chữ nhật được.

    Viết lại theo thuật toán chuẩn (không phụ thuộc scipy). Trả (rows, cols) như scipy.
    """
    c = np.asarray(cost, dtype=float)
    if c.size == 0:
        return [], []
    transposed = c.shape[0] > c.shape[1]
    if transposed:
        c = c.T
    n, m = c.shape
    INF = math.inf
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[j] = hàng (1-based) đang gán cho cột j
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = c[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = sorted((p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0)
    rows = [r for r, _ in pairs]
    cols = [col for _, col in pairs]
    if transposed:
        rows, cols = cols, rows
        order = sorted(range(len(rows)), key=lambda k: rows[k])
        rows, cols = [rows[k] for k in order], [cols[k] for k in order]
    return rows, cols


def label_map(words_ref: Sequence[dict], words_other: Sequence[dict]) -> dict[str, str]:
    """Ánh xạ nhãn của `words_other` → nhãn của `words_ref` (Hungarian, tối đa thời lượng chữ
    cùng nhãn). Hai danh sách phải cùng độ dài (cùng bộ chữ). Nhãn không ghép được thì vắng."""
    if len(words_ref) != len(words_other):
        raise ValueError("hai danh sách chữ phải cùng độ dài")
    ref_labels = sorted({w["speaker"] for w in words_ref if w["speaker"] is not None and not w["punct"]})
    oth_labels = sorted({w["speaker"] for w in words_other if w["speaker"] is not None and not w["punct"]})
    if not ref_labels or not oth_labels:
        return {}
    ri = {s: k for k, s in enumerate(ref_labels)}
    oi = {s: k for k, s in enumerate(oth_labels)}
    co = np.zeros((len(oth_labels), len(ref_labels)))
    for a, b in zip(words_ref, words_other):
        if a["punct"] or a["speaker"] is None or b["speaker"] is None:
            continue
        co[oi[b["speaker"]], ri[a["speaker"]]] += max(_dur(a), 1e-3)
    rows, cols = linear_sum_assignment(-co)
    return {oth_labels[r]: ref_labels[c] for r, c in zip(rows, cols) if co[r, c] > 0}


def disagreements(
    segments: list[dict],
    words: Sequence[dict],
    others: dict[str, Sequence[dict]],
    *,
    min_agree: float = 0.5,
) -> dict[str, dict[str, str]]:
    """Gắn `disagree_arms` cho từng đoạn (có speaker) của arm đang xem: arm khác bất đồng khi
    phần thời lượng chữ mà nhãn (đã ánh xạ Hungarian) trùng nhãn đoạn < `min_agree`.
    Sửa `segments` tại chỗ; trả bảng ánh xạ nhãn theo arm."""
    maps: dict[str, dict[str, str]] = {}
    for arm, ow in others.items():
        m = label_map(words, ow)
        maps[arm] = m
        for seg in segments:
            if seg["speaker"] is None:
                continue
            tot = agree = 0.0
            for i in range(seg["word_start"], seg["word_end"] + 1):
                if words[i]["punct"]:
                    continue
                x = max(_dur(words[i]), 1e-3)
                tot += x
                if m.get(ow[i]["speaker"]) == seg["speaker"]:
                    agree += x
            if tot > 0 and agree / tot < min_agree and arm not in seg["disagree_arms"]:
                seg["disagree_arms"].append(arm)
    return maps


def merge_suggestions(embeddings: dict[str, "np.ndarray | Sequence[float]"], threshold: float = 0.75) -> list[dict]:
    """Cặp cụm giọng có cosine giữa centroid ≥ `threshold` → gợi ý "có thể là cùng một người".

    Chỉ là gợi ý cho người duyệt — không tự gộp."""
    cents: dict[str, np.ndarray] = {}
    for spk, emb in embeddings.items():
        e = np.asarray(emb, dtype=float)
        if e.ndim == 1:
            e = e[None, :]
        e = e[np.all(np.isfinite(e), axis=1)]
        if not len(e):
            continue
        e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
        c = e.mean(axis=0)
        n = np.linalg.norm(c)
        if n > 0:
            cents[spk] = c / n
    keys = sorted(cents)
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            cos = float(np.dot(cents[a], cents[b]))
            if cos >= threshold:
                out.append({"a": a, "b": b, "cosine": round(cos, 4)})
    return sorted(out, key=lambda x: -x["cosine"])


# --------------------------------------------------------------------------- IO


def load_words_xlsx(path: str) -> list[dict]:
    """cleaned_chunks.xlsx → words (bỏ ngoặc kép bọc text của audio_preprocess.save_results)."""
    import pandas as pd

    df = pd.read_excel(path)
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        text = str(row.text)
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            text = text[1:-1]
        out.append({"i": i, "text": text, "start": float(row.start), "end": float(row.end), "punct": is_punct(text)})
    return out
