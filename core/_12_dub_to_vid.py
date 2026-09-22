import os
import platform
import subprocess

import cv2
import numpy as np
from rich.console import Console

from core._1_ytdlp import find_video_files
from core.asr_backend.audio_preprocess import normalize_audio_volume
from core.utils import *
from core.utils.models import *

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = 'output/dub.srt'
DUB_AUDIO = 'output/dub.mp3'

TRANS_FONT_SIZE = 17
TRANS_FONT_NAME = 'Arial'
if platform.system() == 'Linux':
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    TRANS_FONT_NAME = 'Arial Unicode MS'

TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'

def merge_video_audio():
    """Merge video and audio, and reduce video volume"""
    from core._1_ytdlp import is_audio_only_input
    if is_audio_only_input():
        rprint("[bold green]🎵 Audio-only input: skipping dubbing video merge. Dubbed audio is in the `output` directory.[/bold green]")
        return

    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE
    
    if not load_key("burn_subtitles"):
        rprint("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(DUB_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    # Normalize dub audio
    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)
    
    # Merge video and audio with translated subtitles
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    
    # Subtitle-overlay recipe (subdetect): burn dub subs via a generated .ass
    # that covers the original subtitle band; else fixed fallback style.
    recipe_path = os.environ.get("VL_SUB_RECIPE")
    if recipe_path and os.path.exists(recipe_path):
        from core import substyle
        ass_path = substyle.build_ass(
            DUB_SUB_FILE,
            recipe_path,
            os.environ.get("VL_SUB_INTERVALS"),
            "output/dub_recipe.ass",
            extras_path=os.environ.get("VL_SUB_EXTRAS"),
        )
        rprint(f"[bold green]substyle: built {ass_path} from recipe {recipe_path}[/bold green]")
        subtitle_filter = f"ass={substyle.ass_filter_path(ass_path)}"
    else:
        subtitle_filter = (
            f"subtitles={DUB_SUB_FILE}:force_style='FontSize={int(TARGET_HEIGHT*0.035)},"
            f"FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
            f"OutlineColour={TRANS_OUTLINE_COLOR},Outline=2,"
            f"Alignment=2,MarginV={int(TARGET_HEIGHT*0.07)},"
            f"PlayResX={TARGET_WIDTH},PlayResY={TARGET_HEIGHT}'"
        )
    
    cmd = [
        'ffmpeg', '-y', '-i', VIDEO_FILE, '-i', background_file, '-i', normalized_dub_audio,
        '-filter_complex',
        f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
        f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2,'
        # normalize=0: amix mặc định chia mỗi nguồn cho 2 ⇒ giọng lồng mất ~6dB.
        # BGM chạy 0.9 và bị sidechaincompress duck khi có giọng Việt — nhạc rõ
        # ở khoảng trống, tự nhỏ dưới lời thoại; alimiter chống clip.
        f'{subtitle_filter}[v];'
        f'[1:a]volume=0.9[bgm];'
        f'[2:a]asplit=2[dubmix][dubkey];'
        f'[bgm][dubkey]sidechaincompress=threshold=0.02:ratio=8:attack=15:release=400:makeup=1[bgduck];'
        f'[bgduck][dubmix]amix=inputs=2:duration=first:dropout_transition=3:normalize=0[mix];'
        f'[mix]alimiter=limit=0.891[a]'
    ]

    if load_key("ffmpeg_gpu"):
        rprint("[bold green]Using GPU acceleration...[/bold green]")
        cmd.extend(['-map', '[v]', '-map', '[a]', '-c:v', 'h264_nvenc'])
    else:
        cmd.extend(['-map', '[v]', '-map', '[a]'])
    
    cmd.extend(['-c:a', 'aac', '-b:a', '96k', DUB_VIDEO])
    
    subprocess.run(cmd)
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")

if __name__ == '__main__':
    merge_video_audio()
