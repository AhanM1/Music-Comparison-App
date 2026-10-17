"""Tests for musiccompare.audio.loader and the AudioClip model."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musiccompare import config
from musiccompare.audio.loader import AudioLoadError, load_audio
from musiccompare.models import AudioClip

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not on PATH")


@pytest.fixture
def stereo_wav(tmp_path: Path, make_tone) -> Path:
    """A 1-second stereo WAV at 44.1 kHz, i.e. a typical CD-quality file.

    ``tmp_path`` is a built-in pytest fixture: a fresh temporary folder per test.
    """
    sr = 44_100
    left, right = make_tone(440, sr=sr), make_tone(660, sr=sr)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack([left, right]), sr)  # columns = channels
    return path


def test_load_path_gives_mono_at_target_rate(stereo_wav: Path) -> None:
    clip = load_audio(stereo_wav)

    assert clip.y.ndim == 1  # stereo was mixed down to mono
    assert clip.y.dtype == np.float32
    assert clip.sr == config.SAMPLE_RATE  # 44.1 kHz was resampled to 22.05 kHz
    assert clip.duration_s == pytest.approx(1.0, abs=0.01)  # resampling keeps length
    assert clip.name == "stereo.wav"


def test_load_bytes_matches_load_path(stereo_wav: Path) -> None:
    from_path = load_audio(stereo_wav)
    from_bytes = load_audio(stereo_wav.read_bytes())

    assert from_bytes.name == "uploaded audio"
    np.testing.assert_allclose(from_bytes.y, from_path.y)


@requires_ffmpeg
def test_m4a_uses_ffmpeg_fallback(stereo_wav: Path, tmp_path: Path) -> None:
    # libsndfile can't read M4A, so this only passes if the FFmpeg path works.
    m4a = tmp_path / "phone_recording.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(stereo_wav), str(m4a)],
        check=True,
    )

    clip = load_audio(m4a.read_bytes(), name="phone")

    assert clip.sr == config.SAMPLE_RATE
    # AAC encoding adds a few ms of padding, so allow a small tolerance.
    assert clip.duration_s == pytest.approx(1.0, abs=0.05)


def test_garbage_bytes_raise_audio_load_error() -> None:
    # With FFmpeg: it fails to decode. Without: "FFmpeg not found". Either way, a clear error.
    with pytest.raises(AudioLoadError):
        load_audio(b"this is definitely not audio")


def test_audio_clip_rejects_stereo() -> None:
    with pytest.raises(ValueError, match="mono"):
        AudioClip(y=np.zeros((2, 100), dtype=np.float32), sr=22_050, name="bad")
