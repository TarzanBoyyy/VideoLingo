"""VieNeu-TTS v3 Turbo integration for VideoLingo via local OpenAI-compatible server.

Server: ~/project/vieneu-server (uvicorn on 127.0.0.1:4123)
Endpoint: POST /v1/audio/speech  {model, input, voice, response_format}
Output: 48kHz PCM WAV -> converted to 24kHz mono WAV for VideoLingo pipeline.

Voice resolution order:
  1. env `VIENEU_VOICE`  — set per job by the dashboard worker (queue nhiều giọng,
     chạy tuần tự, không phải sửa config.yaml giữa các job)
  2. config.yaml key `custom_tts_voice`
  3. DEFAULT_VOICE (Trúc Ly)
"""
from pathlib import Path
import json
import os
import subprocess
import tempfile

import requests

try:
    from core.utils import load_key
except Exception:  # standalone test
    def load_key(k, default=None):
        return default


VIENEU_URL = "http://127.0.0.1:4123/v1/audio/speech"
DEFAULT_VOICE = "Trúc Ly"


def resolve_voice():
    env_voice = (os.environ.get("VIENEU_VOICE") or "").strip()
    if env_voice:
        return env_voice
    try:
        v = load_key("custom_tts_voice")
        if v:
            return v
    except Exception:
        pass
    return DEFAULT_VOICE


def custom_tts(text, save_path):
    voice = resolve_voice()

    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": "vieneu-v3-turbo",
        "input": text,
        "voice": voice,
        "response_format": "wav",
    }

    last_err = None
    for attempt in range(3):
        try:
            resp = requests.post(VIENEU_URL, json=payload, timeout=120)
            if resp.status_code == 200:
                # VieNeu outputs 48kHz WAV; VideoLingo expects easy-to-handle wav -> resample to 24kHz mono
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_path, "-ar", "24000", "-ac", "1", str(speech_file_path)],
                    check=True, capture_output=True,
                )
                Path(tmp_path).unlink(missing_ok=True)
                print(f"[VieNeu] voice={voice} saved: {speech_file_path}")
                return
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        print(f"[VieNeu] attempt {attempt+1} failed: {last_err}")
    raise RuntimeError(f"VieNeu TTS failed after 3 attempts: {last_err}")


if __name__ == "__main__":
    custom_tts("Xin chào, đây là bản thử nghiệm lồng tiếng từ VieNeu.", "custom_tts_test.wav")
