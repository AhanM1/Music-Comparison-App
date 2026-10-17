"""Shared test fixtures: synthetic audio where the "right answer" is known exactly.

pytest loads this file automatically. Any test can use a fixture below simply by
naming it as a function parameter, e.g. ``def test_x(make_tone): ...``.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from musiccompare import config

ToneFactory = Callable[..., npt.NDArray[np.float32]]


@pytest.fixture
def make_tone() -> ToneFactory:
    """Return a function that generates a pure sine tone.

    This is a "factory fixture": instead of one fixed tone, tests get a function
    they can call with whatever frequency / duration / amplitude they need.
    """

    def _make_tone(
        freq_hz: float = 440.0,
        duration_s: float = 1.0,
        sr: int = config.SAMPLE_RATE,
        amplitude: float = 0.5,
    ) -> npt.NDArray[np.float32]:
        t = np.arange(int(duration_s * sr)) / sr  # sample times in seconds
        return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)

    return _make_tone


@pytest.fixture
def make_silence() -> Callable[..., npt.NDArray[np.float32]]:
    """Return a function that generates digital silence (all zeros)."""

    def _make_silence(
        duration_s: float = 1.0, sr: int = config.SAMPLE_RATE
    ) -> npt.NDArray[np.float32]:
        return np.zeros(int(duration_s * sr), dtype=np.float32)

    return _make_silence
