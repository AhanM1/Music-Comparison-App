"""Load any audio file into a standard AudioClip: mono, float32, fixed sample rate.

Why standardize? Every later step (pitch, loudness, alignment) assumes both
recordings share the same sample rate and channel layout. Doing it once, here,
means no other module ever has to worry about it.

Decoding strategy:
1. soundfile (libsndfile) via librosa: fast, handles WAV / FLAC / OGG / MP3.
2. FFmpeg fallback: converts anything else (e.g. M4A from phone recorders)
   to a temporary WAV, which then goes through the same librosa.load call.
"""

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf

from musiccompare import config
from musiccompare.models import AudioClip

AudioSource = str | Path | bytes

FFMPEG_TIMEOUT_S = 120


class AudioLoadError(Exception):
    """Raised when a file can't be decoded into usable audio."""


def load_audio(
    source: AudioSource,
    name: str | None = None,
    sr: int = config.SAMPLE_RATE,
) -> AudioClip:
    """Load ``source`` (a file path, or the raw bytes of a file) as a mono AudioClip.

    Args:
        source: Path to an audio file, or its contents as bytes (e.g. an upload).
        name: Label for the clip. Defaults to the file name for paths.
        sr: Target sample rate; the audio is resampled to this.

    Raises:
        AudioLoadError: if the audio can't be decoded or is empty.
    """
    if name is None:
        name = Path(source).name if isinstance(source, str | Path) else "uploaded audio"

    try:
        y, _ = _librosa_load(source, sr)
    except sf.LibsndfileError:
        # libsndfile doesn't understand this format; let FFmpeg convert it.
        y = _load_with_ffmpeg(source, sr)

    if y.size == 0:
        raise AudioLoadError(f"{name!r} contains no audio samples.")

    return AudioClip(y=np.asarray(y, dtype=np.float32), sr=sr, name=name)


def _librosa_load(source: AudioSource, sr: int) -> tuple[npt.NDArray[np.float32], float]:
    """Decode + downmix to mono + resample, all in one librosa call."""
    file = io.BytesIO(source) if isinstance(source, bytes) else source
    y, out_sr = librosa.load(file, sr=sr, mono=True)
    return y, out_sr


def _load_with_ffmpeg(source: AudioSource, sr: int) -> npt.NDArray[np.float32]:
    """Convert ``source`` to a temporary 32-bit float WAV with FFmpeg, then load it."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioLoadError(
            "This format needs FFmpeg, which wasn't found on PATH. "
            "Install it (Windows: `winget install Gyan.FFmpeg`) and restart your terminal."
        )

    # A temporary folder is deleted automatically (with everything in it) when the
    # `with` block ends, even if an error happens inside it.
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(source, bytes):
            # FFmpeg needs a real file: some formats (like M4A) store their index at
            # the end, so they can't be decoded from a one-way stream.
            input_path = Path(tmp) / "input"
            input_path.write_bytes(source)
        else:
            input_path = Path(source)
        wav_path = Path(tmp) / "decoded.wav"

        cmd = [
            ffmpeg,
            "-nostdin",  # never wait for keyboard input
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(input_path),
            "-vn",  # ignore video / embedded cover art
            "-c:a", "pcm_f32le",  # lossless 32-bit float, no extra rounding
            str(wav_path),
        ]  # fmt: skip
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=FFMPEG_TIMEOUT_S)
        except subprocess.CalledProcessError as e:
            detail = e.stderr.decode(errors="replace").strip() or "unknown error"
            raise AudioLoadError(f"Could not decode audio: {detail}") from e
        except subprocess.TimeoutExpired as e:
            raise AudioLoadError("Decoding took too long; is the file very large?") from e

        y, _ = _librosa_load(wav_path, sr)
        return y
