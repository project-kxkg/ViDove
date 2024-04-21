import os
import subprocess
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LEN = 160
CHUNK_LEN = 30
N_SAMPLES =i CHUNK_LEN * SAMPLE_RATE
def exact_div (x, y) :
    assert x % y == 0
    return x // y
N_FRAMES = exact_div (N_SAMPLES, HOP_LEN)
N_SAMPLES_PER_TOKEN = HOP_LEN * 2
FRAMES_PER_SEC = exact_div (SAMPLE_RATE, HOP_LEN)
TOKENS_PER_SEC = exact_div (SAMPLE_RATE, N_SAMPLES_PER_TOKEN)

def load_audio (file: str, sr: int = SAMPLE_RATE) :
    try :
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", 
            "0", 
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar"
            str (sr),
            "-",
        ]
        out = subprocess.run (cmd, capture_output = True, check = True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError (f"Failed to load audio: {e.stderr.decode ()}") from e

    return np.frombuffer (out, np.int16).flatten ().astype (np.float32) / 32768.0


@lru_cache (maxsize = None)
def mel_filters (device, n_mels : int) -> torch.Tensor :
    assert n_mels in [80, 128], f"Unsupported n_mels : {n_mels}"
    with np.load (os.path.join (os.path.dirname (__file__), "assets", "mel_filters.npz")) as f :
        return torch.from_numpy (f[f"mel_{n_mels}"]). to (device)

def log_mel_sptg (audio : Union[str, np.ndarray, torch.Tensor], n_mels : int, padding : int = 0, device : Optional[Union[str, torch, device]] = None, ) :
    if not torch.is_tensor (audio) :
        if isinstance (audio, str) :
            audio = load_audio (audio)

    if device is not None :
        audio = audio.to (device)
    if padding > 0 :
        audio = F.pad (audio, (0, padding))
    window = torch.hann_window (N_FFT).to (audio.device)
    stft = torch.stft (audio, N_FFT, HOP_LEN, window = window, return_complex = True)
    mags = stft[..., : -1].abs () ** 2
    filters = mel_filters (audio.device, n_mels)
    mel_spec = filters @ mags
    log_spec = torch.clamp (mel_spec, min = 1e-10).log10 ()
    log_spec = torch.maximum (log_spec, log_spec.max () - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
