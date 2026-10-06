"""Headless driver for VideoLingo pipeline (no Streamlit).

Usage:
  cd ~/project/VideoLingo
  .venv/bin/python run_pipeline.py <input_video> [--sub-only] [--keep]

Produces in output/:
  - src.srt / trans.srt / subtitle video (sub_only or full run)
  - dubbed video (full run)
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

DONE = "output/.text_done"


def sh(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def run(vid: str, sub_only: bool = False, keep: bool = False) -> None:
    """Chạy pipeline cho một video. Tách khỏi main() để batch_runner.py gọi
    in-process nhiều lần (model whisper/demucs được cache ở module level)."""
    vid = os.path.abspath(vid)
    assert os.path.exists(vid), vid

    if not keep:
        shutil.rmtree("output", ignore_errors=True)
    os.makedirs("output", exist_ok=True)

    # VideoLingo finds media via output/input_manifest.json or a single video in output/
    from core._1_ytdlp import write_input_manifest
    dst = os.path.join("output", "source.mp4")
    shutil.copy2(vid, dst)
    write_input_manifest("output/source.mp4", "video")

    t0 = time.time()

    # ---- Text stage ----
    from core import _2_asr, _3_1_split_nlp, _3_2_split_meaning
    from core import _4_1_summarize, _4_2_translate, _5_split_sub, _6_gen_sub, _7_sub_into_vid

    print("== [1/7] WhisperX transcribe ==", flush=True)
    _2_asr.transcribe()

    print("== [2/7] Split NLP ==")
    _3_1_split_nlp.split_by_spacy()
    print("== [3/7] Split by meaning (LLM) ==")
    _3_2_split_meaning.split_sentences_by_meaning()

    print("== [4/7] Summarize ==")
    _4_1_summarize.get_summary()
    print("== [5/7] Translate ==")
    _4_2_translate.translate_all()

    print("== [6/7] Split subs + align timestamps ==")
    _5_split_sub.split_for_sub_main()
    _6_gen_sub.align_timestamp_main()

    print("== [7/7] Burn subtitles ==")
    _7_sub_into_vid.merge_subtitles_to_video()

    if sub_only:
        print(f"SUB-ONLY done in {time.time()-t0:.0f}s")
        return

    # ---- Audio stage ----
    from core import _8_1_audio_task, _8_2_dub_chunks, _9_refer_audio
    from core import _10_gen_audio, _11_merge_audio, _12_dub_to_vid

    print("== [8] Audio tasks + chunks ==")
    _8_1_audio_task.gen_audio_task_main()
    _8_2_dub_chunks.gen_dub_chunks()
    # dub_emotion / dub_fit (config.yaml) — tắt cả hai thì không làm gì
    from core import dub_script
    dub_script.build_dub_script()

    print("== [9] Reference audio ==")
    _9_refer_audio.extract_refer_audio_main()

    print("== [10] Generate TTS audio ==")
    _10_gen_audio.gen_audio()

    print("== [11] Merge audio ==")
    _11_merge_audio.merge_full_audio()

    print("== [12] Mux into video ==")
    _12_dub_to_vid.merge_video_audio()

    print(f"FULL PIPELINE done in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--sub-only", action="store_true", help="stop after subtitle burn-in")
    ap.add_argument("--keep", action="store_true", help="don't wipe output/ first")
    args = ap.parse_args()
    run(args.video, sub_only=args.sub_only, keep=args.keep)


if __name__ == "__main__":
    main()
