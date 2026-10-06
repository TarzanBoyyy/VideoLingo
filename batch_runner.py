"""Persistent pipeline runner for the video-dashboard worker.

Worker spawn một tiến trình này rồi đẩy job qua stdin (mỗi dòng một JSON) thay vì
gọi `run_pipeline.py` mỗi job — torch/whisper/demucs load MỘT lần và giữ ấm trên
GPU giữa các job (whisperX_local + demucs_vl cache model ở module level).

Protocol (stdin, một JSON mỗi dòng):
  {"video": "/abs/src.mp4", "sub_only": false, "env": {"VIENEU_VOICE": "x", "VL_API_KEY": null}}
    → chạy pipeline đủ (hoặc 7 stage khi sub_only). env: giá trị null = unset.
  {"op": "demucs", "src": "/abs/in.wav", "out": "/abs/vocal.wav"}
    → tách giọng bằng separator đã cache (clone ref của dashboard).

Sau mỗi job in sentinel `== [VD-DONE] rc=N ==` (rc=0 ok, 1 lỗi — traceback đã in).
Runner thoát (rc 0) khi stdin đóng, idle quá VL_RUNNER_IDLE_S giây để trả VRAM,
hoặc — khi IDLE_S <= 0 — ngay sau mỗi job (one-shot, VD_WARM_MODELS=0).
"""
import json
import os
import select
import sys
import time
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

DONE_RE = "== [VD-DONE] rc=%d =="
IDLE_S = float(os.environ.get("VL_RUNNER_IDLE_S", "600"))


def _apply_env(env: dict) -> None:
    for k, v in (env or {}).items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


def _demucs(src: str, out: str) -> None:
    from core.asr_backend.demucs_vl import get_separator
    from demucs.audio import save_audio
    sep, model = get_separator()
    _, outputs = sep.separate_audio_file(src)
    save_audio(outputs["vocals"].cpu(), out, samplerate=model.samplerate,
               bitrate=128, preset=2, clip="rescale", as_float=False,
               bits_per_sample=16)


def _handle(spec: dict) -> int:
    try:
        _apply_env(spec.get("env"))
        if spec.get("op") == "demucs":
            _demucs(spec["src"], spec["out"])
        else:
            from run_pipeline import run
            run(spec["video"], sub_only=bool(spec.get("sub_only")))
        return 0
    except Exception:
        traceback.print_exc()
        return 1


def main() -> int:
    # IDLE_S <= 0 = one-shot: thoát sau MỖI job (VD_WARM_MODELS=0 — hành vi cũ).
    # Không dùng làm timeout select: select(0) thoát trước khi đọc được spec.
    wait_s = IDLE_S if IDLE_S > 0 else 86400
    print(f"[batch_runner] sẵn sàng · idle-timeout {IDLE_S:.0f}s", flush=True)
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], wait_s)
        if not ready:
            print("[batch_runner] idle timeout — trả VRAM, thoát", flush=True)
            return 0
        line = sys.stdin.readline()
        if not line:  # worker đóng stdin / chết
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            spec = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[batch_runner] spec hỏng: {exc}", flush=True)
            print(DONE_RE % 1, flush=True)
            continue
        t0 = time.time()
        rc = _handle(spec)
        print(f"[batch_runner] job xong trong {time.time()-t0:.0f}s", flush=True)
        print(DONE_RE % rc, flush=True)
        if IDLE_S <= 0:
            return 0


if __name__ == "__main__":
    sys.exit(main())
