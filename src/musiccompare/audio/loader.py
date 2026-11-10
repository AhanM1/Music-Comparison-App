"""Load any audio file into a standard AudioClip: mono, float32, fixed sample rate.

Why standardize? Every later step (pitch, loudness, alignment) assumes both
recordings share the same sample rate and channel layout. Doing it once, here,
means no other module ever has to worry about it.

Decoding strategy:
1. soundfile (libsndfile) via librosa: fast, handles WAV / FLAC / OGG / MP3.
2. FFmpeg fallback: converts anything else (e.g. M4A from phone recorders)
   to a temporary WAV, which then goes through the same librosa.load call.

Only the first ``config.MAX_DURATION_S`` (+ a little) is decoded, so a huge
file can't tie the app up for minutes before we can say it's too long.
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
from librosa.util.exceptions import ParameterError

from musiccompare import config
from musiccompare.models import AudioClip

AudioSource = str | Path | bytes

FFMPEG_TIMEOUT_S = 120


class AudioLoadError(Exception):
    """Raised when a file can't be decoded into usable audio.

    ``str(error)`` is a message for the player; ``detail`` holds the technical
    reason (e.g. FFmpeg's own error output), if any, for anyone debugging.
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


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
        AudioLoadError: if the audio can't be decoded, is empty, contains
            invalid samples, or is longer than ``config.MAX_DURATION_S``.
    """
    if name is None:
        name = Path(source).name if isinstance(source, str | Path) else "uploaded audio"

    try:
        y = _librosa_load(source, sr)
    except sf.LibsndfileError:
        # libsndfile doesn't understand this format; let FFmpeg convert it.
        y = _load_with_ffmpeg(source, name, sr)

    if y.size == 0:
        raise AudioLoadError(f"{name!r} contains no audio samples.")
    if y.size > config.MAX_DURATION_S * sr:
        minutes = config.MAX_DURATION_S / 60
        raise AudioLoadError(
            f"{name!r} is longer than {minutes:g} minutes. Trim it to the passage you "
            "want to compare (shorter clips also give more focused feedback)."
        )
    # A float WAV can hold NaN ("not a number") or infinity, e.g. from a crashed
    # recorder or a buggy editor. Every later calculation would turn into NaN.
    if not np.all(np.isfinite(y)):
        raise AudioLoadError(
            f"{name!r} contains invalid samples (NaN or infinity), so it's probably "
            "damaged. Try exporting it again from your recording app."
        )

    return AudioClip(y=np.asarray(y, dtype=np.float32), sr=sr, name=name)


def _decode_limit_s() -> float:
    """Decode a little past the cap, so "too long" can be told apart from "exactly at it"."""
    return config.MAX_DURATION_S + 1.0


def _librosa_load(source: AudioSource, sr: int) -> npt.NDArray[np.float32]:
    """Decode + downmix to mono + resample, all in one librosa call."""
    file = io.BytesIO(source) if isinstance(source, bytes) else source
    try:
        y, _ = librosa.load(file, sr=sr, mono=True, duration=_decode_limit_s())
    except ParameterError as e:  # e.g. resampling a signal that contains NaN
        raise AudioLoadError(
            "This file contains invalid audio data, so it's probably damaged.", detail=str(e)
        ) from e
    return np.asarray(y, dtype=np.float32)


def _load_with_ffmpeg(source: AudioSource, name: str, sr: int) -> npt.NDArray[np.float32]:
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
            "-t", str(_decode_limit_s()),  # stop decoding past the length cap
            "-vn",  # ignore video / embedded cover art
            "-c:a", "pcm_f32le",  # lossless 32-bit float, no extra rounding
            str(wav_path),
        ]  # fmt: skip
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=FFMPEG_TIMEOUT_S)
        except subprocess.CalledProcessError as e:
            # FFmpeg's own message ("[in#0 @ 0x...] Invalid data found ...") means
            # nothing to a musician, so it goes in ``detail``, not the message.
            raise AudioLoadError(
                f"Couldn't read {name!r}: it isn't an audio format this app understands, "
                "or the file is damaged. Try exporting it as WAV, MP3, or FLAC.",
                detail=e.stderr.decode(errors="replace").strip() or None,
            ) from e
        except subprocess.TimeoutExpired as e:
            raise AudioLoadError("Decoding took too long; is the file very large?") from e

        return _librosa_load(wav_path, sr)
