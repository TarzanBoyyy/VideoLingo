import os
import torch
from rich.console import Console
from rich import print as rprint
from demucs.pretrained import get_model
from demucs.audio import save_audio
from torch.cuda import is_available as is_cuda_available
from typing import Optional
from demucs.api import Separator
from demucs.apply import BagOfModels
import gc
from core.utils.models import *

class PreloadedSeparator(Separator):
    def __init__(self, model: BagOfModels, shifts: int = 1, overlap: float = 0.25,
                 split: bool = True, segment: Optional[int] = None, jobs: int = 0):
        self._model, self._audio_channels, self._samplerate = model, model.audio_channels, model.samplerate
        device = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                            segment=segment, jobs=jobs, progress=True, callback=None, callback_arg=None)

_SEP_CACHE = None

def get_separator():
    """Separator htdemucs_ft cache ở module level — batch_runner giữ tiến trình
    sống giữa các job nên model chỉ load một lần (và tái dùng cho op demucs
    của clone ref)."""
    global _SEP_CACHE
    if _SEP_CACHE is None:
        console = Console()
        # htdemucs_ft: bản fine-tune của htdemucs — stem sạch hơn (đặc biệt giọng nói
        # đè nhạc), chậm ~4x nhưng vẫn nhanh trên GPU.
        console.print("🤖 Loading <htdemucs_ft> model...")
        model = get_model('htdemucs_ft')
        _SEP_CACHE = (PreloadedSeparator(model=model, shifts=1, overlap=0.25), model)
    return _SEP_CACHE

def demucs_audio():
    if os.path.exists(_VOCAL_AUDIO_FILE) and os.path.exists(_BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} and {_BACKGROUND_AUDIO_FILE} already exist, skip Demucs processing.[/yellow]")
        return

    console = Console()
    os.makedirs(_AUDIO_DIR, exist_ok=True)

    separator, model = get_separator()
    
    console.print("🎵 Separating audio...")
    _, outputs = separator.separate_audio_file(_RAW_AUDIO_FILE)

    kwargs = {"samplerate": model.samplerate, "bitrate": 128, "preset": 2,
             "clip": "rescale", "as_float": False, "bits_per_sample": 16}

    console.print("🎤 Saving vocals track...")
    save_audio(outputs['vocals'].cpu(), _VOCAL_AUDIO_FILE, **kwargs)

    console.print("🎹 Saving background music...")
    background = sum(audio for source, audio in outputs.items() if source != 'vocals')
    save_audio(background.cpu(), _BACKGROUND_AUDIO_FILE, **kwargs)
    
    # outputs/background là tensor lớn — giải phóng; model+separator giữ trong _SEP_CACHE.
    del outputs, background
    gc.collect()
    
    console.print("[green]✨ Audio separation completed![/green]")

if __name__ == "__main__":
    demucs_audio()
