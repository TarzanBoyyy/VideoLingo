"""Nhận diện cảm xúc từng câu thoại bằng emotion2vec+ large (FunASR).

Chạy trên vocal đã tách (demucs), cắt theo timestamp từng dòng trong tts_tasks.xlsx,
resample 16 kHz mono. Kết quả chỉ là GỢI Ý THÔ cho LLM (core/dub_script.py): score cao
nhất dưới `dub_emotion.min_score`, câu quá ngắn, hoặc nhãn neutral/other/unk → "neutral".

Cache ra output/audio/emotion.json (key = số dòng) để chạy lại stage không phải load
model lần nữa. Model được giải phóng ngay sau khi dùng — VieNeu cần VRAM cho TTS.
"""
import gc
import json
import os
import subprocess

import numpy as np
import soundfile as sf

from core.asr_backend.demucs_vl import demucs_audio
from core.utils import load_key, rprint
from core.utils.models import _AUDIO_DIR, _VOCAL_AUDIO_FILE

EMOTION_FILE = os.path.join(_AUDIO_DIR, "emotion.json")
VOCAL_16K = os.path.join(_AUDIO_DIR, "vocal_16k.wav")
SR = 16000
_NOT_A_HINT = {"neutral", "other", "unk", "<unk>", "unknown"}


def _seconds(t: str) -> float:
    h, m, s = str(t).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s.replace(",", "."))


def _english(label: str) -> str:
    # emotion2vec+ trả "生气/angry" — giữ phần tiếng Anh
    return label.split("/")[-1].strip().lower()


def _vocal_16k() -> np.ndarray:
    demucs_audio()  # bỏ qua nếu vocal.mp3 đã có (stage 9 sẽ dùng lại)
    if not os.path.exists(VOCAL_16K):
        subprocess.run(["ffmpeg", "-y", "-i", _VOCAL_AUDIO_FILE, "-ac", "1", "-ar", str(SR), VOCAL_16K],
                       check=True, capture_output=True)
    data, sr = sf.read(VOCAL_16K, dtype="float32")
    assert sr == SR, sr
    return data


def detect_emotions(df) -> dict[int, dict]:
    """{number: {"hint", "label", "score"}} cho mọi dòng của tts_tasks."""
    if os.path.exists(EMOTION_FILE):
        cached = {int(k): v for k, v in json.load(open(EMOTION_FILE, encoding="utf-8")).items()}
        if set(cached) >= set(int(n) for n in df["number"]):
            rprint(f"[blue]emotion: dùng cache {EMOTION_FILE}[/blue]")
            return cached

    min_score = float(load_key("dub_emotion.min_score"))
    min_clip = float(load_key("dub_emotion.min_clip_sec"))
    data = _vocal_16k()

    from funasr import AutoModel
    import torch
    # str(): load_key trả kiểu ruamel, OmegaConf trong FunASR không merge được
    model = AutoModel(model=str(load_key("dub_emotion.model")), hub=str(load_key("dub_emotion.hub")),
                      disable_update=True, log_level="ERROR")
    out: dict[int, dict] = {}
    try:
        for _, row in df.iterrows():
            n = int(row["number"])
            a, b = int(_seconds(row["start_time"]) * SR), int(_seconds(row["end_time"]) * SR)
            clip = data[a:b]
            if len(clip) < min_clip * SR:
                out[n] = {"hint": "neutral", "label": None, "score": None}
                continue
            r = model.generate(clip, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
            i = int(np.argmax(r["scores"]))
            label, score = _english(r["labels"][i]), float(r["scores"][i])
            hint = label if score >= min_score and label not in _NOT_A_HINT else "neutral"
            out[n] = {"hint": hint, "label": label, "score": round(score, 3)}
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(EMOTION_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    hinted = sum(1 for v in out.values() if v["hint"] != "neutral")
    rprint(f"[green]emotion: {len(out)} dòng, {hinted} dòng có gợi ý (score ≥ {min_score})[/green]")
    return out
