"""Shared pieces: the pinned public DeepFilterNet3 checkpoint, audio loading, and the libDF feature state."""
from __future__ import annotations

from pathlib import Path
import sys
import types
import urllib.request
import zipfile

import numpy as np
import soundfile as sf
from scipy import signal

RATE = 48000
CACHE = Path(__file__).resolve().parent / ".cache"
BASE_DIR = CACHE / "DeepFilterNet3"
# The released DeepFilterNet3 checkpoint, pinned to a commit of github.com/Rikorose/DeepFilterNet.
BASE_URL = ("https://github.com/Rikorose/DeepFilterNet/raw/d375b2d8309e0935d165700c91da9de862a99c31/"
            "models/DeepFilterNet3.zip")

# deepfilternet 0.5.6 imports torchaudio.backend.common (only for a type hint); newer torchaudio dropped it.
if "torchaudio.backend.common" not in sys.modules:
    _backend, _common = types.ModuleType("torchaudio.backend"), types.ModuleType("torchaudio.backend.common")
    _common.AudioMetaData = type("AudioMetaData", (), {})
    _backend.common = _common
    sys.modules["torchaudio.backend"], sys.modules["torchaudio.backend.common"] = _backend, _common


def ensure_base() -> Path:
    """Download and unpack the public checkpoint once (about 8 MB)."""
    if not (BASE_DIR / "config.ini").is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        archive = CACHE / "DeepFilterNet3.zip"
        print(f"downloading the public DeepFilterNet3 checkpoint to {CACHE}")
        urllib.request.urlretrieve(BASE_URL, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(CACHE)
        archive.unlink()
    return BASE_DIR


def fresh_state():
    """libDF's analysis keeps streaming state, so every batch and every file gets a new one."""
    from libdf import DF
    return DF(sr=RATE, fft_size=960, hop_size=480, nb_bands=32, min_nb_erb_freqs=2)


def load(path, rate=RATE) -> np.ndarray:
    """Mono float32 at `rate`, with digital silence (from mic gates) trimmed off both ends."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != rate:
        x = signal.resample_poly(x, rate, sr).astype(np.float32)
    sound = np.flatnonzero(np.abs(x) > 1e-5)
    return x[sound[0]:sound[-1] + 1] if sound.size else x


def load_folder(folder) -> dict[str, np.ndarray]:
    files = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in (".wav", ".flac"))
    if not files:
        raise SystemExit(f"no .wav or .flac files in {folder}")
    return {p.stem: load(p) for p in files}


def split(x, held_out=0.25):
    """The last part of every take is kept out of training and used to compare before and after."""
    cut = int(len(x) * (1 - held_out))
    return x[:cut], x[cut:]
