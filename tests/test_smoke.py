"""Smoke tests: prove the package installs, imports, and its config makes sense.

pytest finds this file because its name starts with ``test_``, and runs every
function whose name starts with ``test_``. A plain ``assert`` is all you need.
"""

import librosa

import musiccompare
from musiccompare import config


def test_package_imports() -> None:
    assert musiccompare.__version__ == "0.1.0"


def test_pitch_range_is_below_nyquist() -> None:
    # A sampled signal can only represent frequencies up to half its sample rate
    # (the Nyquist frequency). If FMAX were above that, pitch tracking would break.
    nyquist_hz = config.SAMPLE_RATE / 2
    fmin_hz = librosa.note_to_hz(config.FMIN)
    fmax_hz = librosa.note_to_hz(config.FMAX)

    assert 0 < fmin_hz < fmax_hz < nyquist_hz
