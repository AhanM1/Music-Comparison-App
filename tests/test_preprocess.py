"""Tests for musiccompare.audio.preprocess."""

import numpy as np
import pytest

from musiccompare import config
from musiccompare.audio.preprocess import normalize_peak, trim_silence
from musiccompare.models import AudioClip

SR = config.SAMPLE_RATE


def clip_of(y: np.ndarray) -> AudioClip:
    return AudioClip(y=y, sr=SR, name="test")


def test_trim_removes_leading_and_trailing_silence(make_tone, make_silence) -> None:
    # 1 s silence + 1 s tone + 1 s silence  ->  should trim to ~1 s
    y = np.concatenate([make_silence(1.0), make_tone(duration_s=1.0), make_silence(1.0)])

    trimmed = trim_silence(clip_of(y))

    # librosa trims in whole frames (~23 ms each), so allow a little slack.
    assert trimmed.duration_s == pytest.approx(1.0, abs=0.1)


def test_trim_keeps_silence_in_the_middle(make_tone, make_silence) -> None:
    # tone + rest + tone: the rest is part of the music and must stay.
    y = np.concatenate([make_tone(duration_s=1.0), make_silence(1.0), make_tone(duration_s=1.0)])

    trimmed = trim_silence(clip_of(y))

    assert trimmed.duration_s == pytest.approx(3.0, abs=0.1)


def test_normalize_sets_peak_and_preserves_shape(make_tone) -> None:
    quiet = clip_of(make_tone(amplitude=0.1))

    loud = normalize_peak(quiet, peak=0.9)

    assert np.max(np.abs(loud.y)) == pytest.approx(0.9)
    # Every sample scaled by the same factor -> dynamics shape unchanged.
    np.testing.assert_allclose(loud.y, quiet.y * 9, rtol=1e-5)
    assert loud.y.dtype == np.float32


def test_normalize_does_not_modify_original(make_tone) -> None:
    original = clip_of(make_tone(amplitude=0.1))
    before = original.y.copy()

    normalize_peak(original)

    np.testing.assert_array_equal(original.y, before)


def test_normalize_silent_clip_raises(make_silence) -> None:
    with pytest.raises(ValueError, match="silent"):
        normalize_peak(clip_of(make_silence()))
