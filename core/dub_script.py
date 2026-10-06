"""Kịch bản lồng tiếng: dịch + sắc thái + vừa ngân sách thời lượng trong MỘT lượt LLM.

Chạy sau `_8_2_dub_chunks` (mỗi dòng tts_tasks đã có timestamp, câu gốc, bản dịch phụ đề):
  - ngân sách dòng = thời lượng câu gốc theo timestamp (`duration`);
  - `dub_emotion.enabled`: emotion2vec+ (core/emotion.py) cho nhãn thô → LLM chốt `style`
    theo ngữ cảnh. `style` chỉ là metadata, chưa chèn vào text TTS (chờ chốt engine);
  - `dub_fit.enabled`: câu Việt LLM viết (≤ max_syllables) thay `lines` để TTS đọc và ra
    dub.srt. Cột `text` giữ bản phụ đề cũ để `_8_2` chạy lại (--keep) vẫn khớp trans.srt.

Sau TTS, `_10_gen_audio` gọi `fit_row` (đo → viết ngắn hơn, tối đa `max_rewrites`), rồi mới
tăng tốc có trần `max_speed`; cuối stage ghi output/log/dub_report.json bằng `write_report`.
"""
import concurrent.futures
import json
import math
import os
import re
from collections import Counter

import pandas as pd

from core.prompts import get_dub_script_prompt, get_dub_shorten_prompt
from core.utils import ask_gpt, check_cancel, load_key, rprint
from core.utils.models import _4_1_TERMINOLOGY, _8_1_AUDIO_TASK

REPORT_FILE = "output/log/dub_report.json"
CONTEXT_LINES = 2


def _cfg(key, default):
    try:
        return load_key(key)
    except KeyError:  # config.yaml cũ chưa có khối dub_* → coi như tắt
        return default


def emotion_on() -> bool:
    return bool(_cfg("dub_emotion.enabled", False))


def fit_on() -> bool:
    return bool(_cfg("dub_fit.enabled", False))


def syllables(text: str) -> int:
    # Tiếng Việt: mỗi âm tiết là một "từ" cách nhau bởi khoảng trắng
    return len(re.findall(r"\w+", text or ""))


_VI_MARKS = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)


def vi_ok(text: str) -> bool:
    """LLM đôi khi trả tiếng Việt KHÔNG DẤU — TTS đọc sai hẳn; câu ≥ 3 âm tiết phải có dấu."""
    return syllables(text) < 3 or bool(_VI_MARKS.search(text))


def as_list(v) -> list:
    return eval(v) if isinstance(v, str) else list(v)


def _max_syllables(budget: float) -> int:
    return max(1, math.floor(budget * float(load_key("dub_fit.syllables_per_sec"))))


# ---------------------------------------------------------------- lượt gọi LLM gộp

def _write_batch(batch, before, after, theme, terms, styles):
    zh_all = " ".join(it["zh"] for it in batch)
    batch_terms = [t for t in terms if t.get("src") and t["src"] in zh_all]
    prompt = get_dub_script_prompt(batch, theme, batch_terms, before, after, styles)
    want = {it["id"] for it in batch}

    def valid(resp):
        lines = resp.get("lines") if isinstance(resp, dict) else None
        if not isinstance(lines, list):
            return {"status": "error", "message": "no lines"}
        got = {l.get("id") for l in lines if isinstance(l, dict)}
        if got != want:
            return {"status": "error", "message": f"ids mismatch: want {sorted(want)} got {sorted(got, key=str)}"}
        if any(not str(l.get("vi") or "").strip() for l in lines):
            return {"status": "error", "message": "empty vi"}
        if any(not vi_ok(str(l["vi"])) for l in lines):
            return {"status": "error", "message": "vi without diacritics"}
        return {"status": "success", "message": ""}

    resp = ask_gpt(prompt, resp_type="json", valid_def=valid, log_title="dub_script")
    out = {}
    for l in resp["lines"]:
        style = str(l.get("style") or "neutral").strip().lower()
        out[l["id"]] = {"vi": str(l["vi"]).strip(),
                        "style": style if style in styles else "neutral",
                        "note": str(l.get("note") or "").strip()}
    return out


def build_dub_script() -> None:
    emo, fit = emotion_on(), fit_on()
    if not (emo or fit):
        return
    df = pd.read_excel(_8_1_AUDIO_TASK)

    hints = {}
    if emo:
        from core.emotion import detect_emotions
        hints = detect_emotions(df)
    styles = [str(s) for s in _cfg("dub_emotion.styles", ["neutral"])] if emo else ["neutral"]

    with open(_4_1_TERMINOLOGY, encoding="utf-8") as f:
        terminology = json.load(f)
    theme, terms = terminology.get("theme", ""), terminology.get("terms", [])

    items = []
    for _, r in df.iterrows():
        n = int(r["number"])
        hint = hints.get(n, {}).get("hint")
        items.append({
            "id": n,
            "zh": str(r["origin"]),
            "vi_draft": " ".join(as_list(r["lines"])),
            "budget_sec": round(float(r["duration"]), 2),
            "max_syllables": _max_syllables(float(r["duration"])),
            "emotion_hint": hint if hint and hint != "neutral" else None,
        })

    bs = int(_cfg("dub_fit.batch_size", 12))
    batches = [(i, items[i:i + bs]) for i in range(0, len(items), bs)]
    ctx = lambda it: f"{it['zh']} / {it['vi_draft']}"  # noqa: E731
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as ex:
        futures = [ex.submit(_write_batch, b,
                             [ctx(it) for it in items[max(0, i - CONTEXT_LINES):i]],
                             [ctx(it) for it in items[i + len(b):i + len(b) + CONTEXT_LINES]],
                             theme, terms, styles)
                   for i, b in batches]
        for fut in concurrent.futures.as_completed(futures):
            check_cancel()
            results.update(fut.result())

    df["budget"] = df["duration"]
    df["vi_draft"] = [it["vi_draft"] for it in items]
    if emo:
        df["emotion_label"] = [hints[int(n)]["label"] for n in df["number"]]
        df["emotion_score"] = [hints[int(n)]["score"] for n in df["number"]]
        df["emotion_hint"] = [hints[int(n)]["hint"] for n in df["number"]]
        df["style"] = [results[int(n)]["style"] for n in df["number"]]
        df["style_note"] = [results[int(n)]["note"] for n in df["number"]]
    if fit:
        df["lines"] = [str([results[int(n)]["vi"]]) for n in df["number"]]
        df["src_lines"] = [str([" ".join(as_list(s))]) for s in df["src_lines"]]
        # Tính lại est_dur / cut_off theo câu mới (chunk tăng tốc dựa trên các cột này)
        from core._8_2_dub_chunks import analyze_subtitle_timing_and_speed, process_cutoffs
        tmp = df.copy()
        tmp["text"] = [results[int(n)]["vi"] for n in df["number"]]
        tmp = process_cutoffs(analyze_subtitle_timing_and_speed(tmp))
        tmp["text"] = df["text"]
        df = tmp

    df.to_excel(_8_1_AUDIO_TASK, index=False)
    rprint(f"[green]dub_script: {len(df)} dòng (fit={fit}, emotion={emo})[/green]")


# ---------------------------------------------------------------- đo → viết ngắn hơn

def _neighbour(tasks_df, number, step):
    idx = tasks_df.index[tasks_df["number"] == number][0] + step
    if 0 <= idx < len(tasks_df):
        return " ".join(as_list(tasks_df.loc[idx, "lines"]))
    return ""


def _shorten(row, cur, measured, budget, target, tasks_df):
    prompt = get_dub_shorten_prompt(row["origin"], cur, measured, budget, target,
                                    _neighbour(tasks_df, row["number"], -1),
                                    _neighbour(tasks_df, row["number"], 1))

    def valid(resp):
        vi = str(resp.get("vi") or "").strip() if isinstance(resp, dict) else ""
        ok = bool(vi) and vi_ok(vi)
        return {"status": "success" if ok else "error", "message": "" if ok else "empty vi / no diacritics"}

    try:
        return str(ask_gpt(prompt, resp_type="json", valid_def=valid, log_title="dub_shorten")["vi"]).strip()
    except Exception as e:  # noqa: BLE001 — hết lượt LLM thì để bước tăng tốc lo
        rprint(f"[yellow]dub_shorten dòng {row['number']} lỗi: {e}[/yellow]")
        return None


def fit_row(row, lines, real_dur, synth, tasks_df):
    """TTS dài hơn ngân sách quá `overshoot_pct` → LLM viết ngắn hơn rồi đọc lại.
    `synth(lines) -> real_dur`. Trả (lines, real_dur, số lần viết lại)."""
    budget = float(row["duration"])
    limit = budget * (1 + float(load_key("dub_fit.overshoot_pct")))
    max_rewrites = int(load_key("dub_fit.max_rewrites"))
    rewrites = 0
    while real_dur > limit and rewrites < max_rewrites:
        cur = " ".join(lines)
        n = syllables(cur)
        target = min(n - 1, math.floor(n * budget / real_dur))
        if target < 1:
            break
        new = _shorten(row, cur, real_dur, budget, target, tasks_df)
        if not new or syllables(new) >= n:
            break
        rewrites += 1
        rprint(f"[cyan]✂️ dòng {row['number']}: {real_dur:.2f}s > {budget:.2f}s — viết lại lần {rewrites}: {new}[/cyan]")
        lines = [new]
        real_dur = synth(lines)
    return lines, real_dur, rewrites


# ---------------------------------------------------------------- báo cáo

def write_report(tasks_df) -> dict | None:
    emo, fit = emotion_on(), fit_on()
    if not (emo or fit):
        return None
    df = tasks_df
    col = lambda c, d=None: df[c] if c in df.columns else pd.Series([d] * len(df))  # noqa: E731
    pct = float(_cfg("dub_fit.overshoot_pct", 0))
    speed = col("speed", 1.0).fillna(1.0).astype(float)
    rewrites = col("rewrites", 0).fillna(0).astype(int)
    truncated = col("truncated", 0).fillna(0).astype(float)
    over = df["real_dur"].astype(float) > df["duration"].astype(float) * (1 + pct)

    report = {
        "rows": len(df),
        "fit_enabled": fit,
        "emotion_enabled": emo,
        "rewritten_rows": int((rewrites > 0).sum()),
        "rewrites_total": int(rewrites.sum()),
        "over_budget_after_rewrites": int(over.sum()),
        "sped_up_rows": int((speed > 1.001).sum()),
        "speed_max": round(float(speed.max()), 3) if len(df) else 1.0,
        "truncated_rows": int((truncated > 0).sum()),
        "truncated_sec_total": round(float(truncated.sum()), 2),
    }
    if emo:
        raw = Counter(str(x) for x in col("emotion_label").fillna("(quá ngắn)"))
        report["emotion2vec_top_label"] = dict(raw.most_common())
        report["emotion_hint"] = dict(Counter(col("emotion_hint", "neutral")).most_common())
        report["style_final"] = dict(Counter(col("style", "neutral")).most_common())
        pairs = Counter(f"{h} → {s}" for h, s in zip(col("emotion_hint", "neutral"), col("style", "neutral")))
        report["hint_to_style"] = dict(pairs.most_common())

    detail_cols = [c for c in ("number", "start_time", "duration", "origin", "vi_draft", "lines", "real_dur",
                               "rewrites", "speed", "truncated", "emotion_label", "emotion_score",
                               "emotion_hint", "style", "style_note") if c in df.columns]
    detail = df[detail_cols].copy()
    detail["lines"] = [" ".join(as_list(v)) for v in detail["lines"]]
    report["lines"] = json.loads(detail.to_json(orient="records", force_ascii=False))

    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    summary = {k: v for k, v in report.items() if k != "lines"}
    rprint(f"[bold green]dub_report: {json.dumps(summary, ensure_ascii=False)}[/bold green]")
    return report
