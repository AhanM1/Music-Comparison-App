"""Tests for musiccompare.analysis.quality: spotting "busy" recordings.

A busy recording has more than one note sounding at a time (accompaniment, a
drone, heavy reverb), which the note-by-note comparison can't handle well.
"""

import numpy as np
import pytest

from musiccompare import config
from musiccompare.analysis.quality import busy_reference, busy_take, clarity, is_busier
from musiccompare.features.extract import extract_features
from musiccompare.models import AudioClip

SR = config.SAMPLE_RATE
TUNE = [60, 64, 67, 72, 71, 67, 65, 69, 74, 72, 67, 64]


def features_of(y: np.ndarray):  # type: ignore[no-untyped-def]
    return extract_features(AudioClip(y=y.astype(np.float32), sr=SR, name="test"))


def sine(freq: float, n: int, amplitude: float) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * np.arange(n) / SR)


@pytest.fixture(scope="module")
def solo() -> np.ndarray:
    """A plain 12-note melody with overtones (built here, once per module, since
    conftest's make_melody fixture can't be used by a module-scoped fixture)."""
    notes = []
    for midi in TUNE:
        t = np.arange(int(0.4 * SR)) / SR
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        tone = sum(a * np.sin(2 * np.pi * f0 * (h + 1) * t) for h, a in enumerate((1, 0.5, 0.3)))
        notes.append(0.3 * tone * np.minimum(1, t / 0.02) * np.minimum(1, (0.4 - t) / 0.03))
    return np.concatenate(notes)


def test_a_solo_melody_is_clear(solo: np.ndarray) -> None:
    assert clarity(features_of(solo)).pitch_confidence > 0.8


def test_silence_has_no_clarity() -> None:
    assert clarity(features_of(np.zeros(SR))).pitch_confidence == 0.0


@pytest.mark.parametrize("accompaniment", ["drone", "chords"])
def test_accompaniment_under_the_reference_is_busy(solo: np.ndarray, accompaniment: str) -> None:
    n = solo.size
    under = (
        sine(110, n, 0.25) + sine(165, n, 0.15)  # a drone on A and E
        if accompaniment == "drone"
        else sine(130.81, n, 0.2) + sine(164.81, n, 0.2) + sine(196.0, n, 0.2)  # C major
    )
    ref, user = features_of(solo + under), features_of(solo)
    assert busy_reference(ref, user)
    assert not busy_take(ref, user)


def test_a_backing_track_under_your_take_is_busy(solo: np.ndarray) -> None:
    n = solo.size
    ref, user = features_of(solo), features_of(solo + sine(110, n, 0.25) + sine(165, n, 0.15))
    assert busy_take(ref, user)
    assert not busy_reference(ref, user)


def overtone_rich(solo_notes: list[int]) -> np.ndarray:
    """The same tune with eight strong harmonics: a very bright solo instrument."""
    notes = []
    for midi in solo_notes:
        t = np.arange(int(0.4 * SR)) / SR
        f0 = 440.0 * 2 ** ((midi - 69) / 12)
        weights = (1.0, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.5)
        tone = sum(a * np.sin(2 * np.pi * f0 * (h + 1) * t) for h, a in enumerate(weights))
        notes.append(0.1 * tone * np.minimum(1, t / 0.02) * np.minimum(1, (0.4 - t) / 0.03))
    return np.concatenate(notes)


@pytest.mark.parametrize("variant", ["hiss", "quieter", "same", "much brighter timbre"])
def test_clean_variants_are_not_busy(solo: np.ndarray, variant: str) -> None:
    """Timbre especially: a bright solo isn't "busy" (which a chroma-based check got wrong)."""
    other = {
        "hiss": solo + 0.01 * np.random.default_rng(0).standard_normal(solo.size),
        "quieter": 0.3 * solo,
        "same": solo,
        "much brighter timbre": overtone_rich(TUNE),
    }[variant]
    ref, user = features_of(other), features_of(solo)
    assert not busy_reference(ref, user)
    assert not busy_take(ref, user)


def test_is_busier_compares_pitch_confidence() -> None:
    from musiccompare.analysis.quality import Clarity

    clear = Clarity(pitch_confidence=0.95)
    assert is_busier(Clarity(0.3), clear)
    assert not is_busier(Clarity(0.9), clear)  # small differences are normal
    assert not is_busier(clear, Clarity(0.3))  # it's the *other* one that's busy
