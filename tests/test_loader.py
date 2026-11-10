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


@requires_ffmpeg
def test_decode_errors_keep_ffmpeg_details_out_of_the_message() -> None:
    with pytest.raises(AudioLoadError) as info:
        load_audio(b"this is definitely not audio", name="junk.wav")

    message = str(info.value)
    assert "junk.wav" in message
    assert "isn't an audio format" in message
    assert "[in#" not in message  # FFmpeg internals...
    assert info.value.detail  # ...are kept for debugging instead


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("sr", [config.SAMPLE_RATE, 44_100])  # with and without resampling
def test_invalid_samples_raise_audio_load_error(
    tmp_path: Path, make_tone, bad: float, sr: int
) -> None:
    """A float WAV can store NaN/infinity; that used to crash deep inside librosa."""
    y = make_tone(sr=sr)
    y[1000] = bad
    path = tmp_path / "damaged.wav"
    sf.write(path, y, sr, subtype="FLOAT")

    with pytest.raises(AudioLoadError, match="damaged"):
        load_audio(path)


def test_too_long_raises_audio_load_error(
    tmp_path: Path, make_tone, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MAX_DURATION_S", 2.0)
    path = tmp_path / "long.wav"
    sf.write(path, make_tone(duration_s=4.0), config.SAMPLE_RATE)

    with pytest.raises(AudioLoadError, match="longer than"):
        load_audio(path)


def test_exactly_the_maximum_length_is_fine(
    tmp_path: Path, make_tone, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "MAX_DURATION_S", 2.0)
    path = tmp_path / "ok.wav"
    sf.write(path, make_tone(duration_s=2.0), config.SAMPLE_RATE)
    assert load_audio(path).duration_s == pytest.approx(2.0)


@requires_ffmpeg
def test_ffmpeg_stops_decoding_past_the_cap(
    stereo_wav: Path, tmp_path: Path, make_tone, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only about MAX_DURATION_S + 1 s is decoded, however long the file is."""
    long_wav = tmp_path / "long.wav"
    sf.write(long_wav, make_tone(duration_s=8.0), config.SAMPLE_RATE)
    m4a = tmp_path / "long.m4a"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(long_wav), str(m4a)], check=True
    )
    monkeypatch.setattr(config, "MAX_DURATION_S", 2.0)
    calls: list[list[str]] = []
    real_run = subprocess.run

    def spy(cmd: list[str], **kwargs: object) -> object:
        calls.append(cmd)
        return real_run(cmd, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(subprocess, "run", spy)
    with pytest.raises(AudioLoadError, match="longer than"):
        load_audio(m4a)
    assert calls[0][calls[0].index("-t") + 1] == "3.0"


def test_audio_clip_rejects_stereo() -> None:
    with pytest.raises(ValueError, match="mono"):
        AudioClip(y=np.zeros((2, 100), dtype=np.float32), sr=22_050, name="bad")
