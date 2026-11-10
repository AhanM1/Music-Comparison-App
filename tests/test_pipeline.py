"""Tests for musiccompare.pipeline: the end-to-end analysis every front end uses."""

import numpy as np
import pytest

from musiccompare import config
from musiccompare.analysis.alignment import align
from musiccompare.analysis.compare import compare
from musiccompare.audio.loader import AudioLoadError
from musiccompare.pipeline import compare_takes, prepare

SR = config.SAMPLE_RATE


def test_prepare_trims_normalizes_and_measures(make_tone, make_silence, to_wav_bytes) -> None:
    y = np.concatenate([make_silence(0.5), make_tone(440, duration_s=1.5, amplitude=0.2)])

    clip, features = prepare(to_wav_bytes(y), name="take.wav")

    assert clip.name == "take.wav"
    assert clip.duration_s == pytest.approx(1.5, abs=0.05)  # leading silence trimmed
    assert np.abs(clip.y).max() == pytest.approx(0.9, abs=1e-3)  # normalized
    assert features.n_frames == pytest.approx(clip.duration_s * SR / config.HOP_LENGTH, abs=2)


@pytest.mark.parametrize("seconds", [1 / SR, 0.05, 0.5])
def test_prepare_rejects_too_short_recordings(make_tone, to_wav_bytes, seconds: float) -> None:
    """A click or a 1-sample file used to produce scores, tips, and saved sessions."""
    y = make_tone(440, duration_s=seconds)
    with pytest.raises(AudioLoadError, match="at least 1 s"):
        prepare(to_wav_bytes(np.maximum(y, 1e-3)), name="click.wav")


def test_prepare_rejects_a_short_sound_in_a_long_silence(
    make_tone, make_silence, to_wav_bytes
) -> None:
    y = np.concatenate([make_silence(2.0), make_tone(440, duration_s=0.3), make_silence(2.0)])
    # Trimming works in whole frames, so a little of the silence stays.
    with pytest.raises(AudioLoadError, match=r"only 0\.[34] s of sound"):
        prepare(to_wav_bytes(y))


def test_prepare_rejects_silence(make_silence, to_wav_bytes) -> None:
    with pytest.raises(AudioLoadError, match="completely silent"):
        prepare(to_wav_bytes(make_silence(3.0)), name="quiet.wav")


def test_compare_takes_aligns_then_compares(make_melody) -> None:
    from musiccompare.features.extract import extract_features
    from musiccompare.models import AudioClip

    notes = [60, 64, 67, 72]
    ref = extract_features(AudioClip(make_melody(notes), SR, "ref"))
    user = extract_features(AudioClip(make_melody(notes, durations=0.5), SR, "me"))

    result = compare_takes(ref, user)

    expected = compare(ref, user, align(ref, user))
    assert result.scores == expected.scores
    np.testing.assert_array_equal(result.alignment.path, expected.alignment.path)
