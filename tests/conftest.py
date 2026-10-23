"""Shared test fixtures: synthetic audio where the "right answer" is known exactly.

pytest loads this file automatically. Any test can use a fixture below simply by
naming it as a function parameter, e.g. ``def test_x(make_tone): ...``.
"""

import io
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

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
def make_melody() -> Callable[..., npt.NDArray[np.float32]]:
    """Return a function that synthesizes a melody of instrument-like notes.

    Each note has a fundamental plus overtones (like a bowed or plucked string)
    and a short attack/release so note boundaries are audible. Every property can
    be set per note, so tests can make a "student" version that is out of tune
    (``cents``), uneven in loudness (``amplitudes``), or rushed (``durations``).
    ``vibrato_cents`` > 0 adds vibrato: the pitch wobbles ± that many cents,
    ``vibrato_hz`` times per second, as string players do.
    """

    def _make_melody(
        midi_notes: list[float],
        durations: float | list[float] = 0.4,
        cents: float | list[float] = 0.0,
        amplitudes: float | list[float] = 0.5,
        overtones: tuple[float, ...] = (1.0, 0.5, 0.3, 0.2),
        vibrato_cents: float = 0.0,
        vibrato_hz: float = 5.5,
        sr: int = config.SAMPLE_RATE,
    ) -> npt.NDArray[np.float32]:
        n = len(midi_notes)
        durs = np.broadcast_to(durations, n)
        offs = np.broadcast_to(cents, n)
        amps = np.broadcast_to(amplitudes, n)
        notes = []
        for midi, dur, off, amp in zip(midi_notes, durs, offs, amps, strict=True):
            t = np.arange(int(dur * sr)) / sr
            f0 = 440.0 * 2 ** ((midi - 69) / 12 + off / 1200)
            # Phase = 2π x (running total of the frequency); with no vibrato it's 2π f0 t.
            wobble = 2 ** (vibrato_cents * np.sin(2 * np.pi * vibrato_hz * t) / 1200)
            phase = 2 * np.pi * f0 * (np.cumsum(wobble) / sr if vibrato_cents else t)
            tone = sum(a * np.sin(phase * (h + 1)) for h, a in enumerate(overtones))
            envelope = np.minimum(1, t / 0.02) * np.minimum(1, (dur - t) / 0.03)
            notes.append(amp * tone * envelope / sum(overtones))
        return np.concatenate(notes).astype(np.float32)

    return _make_melody


@pytest.fixture
def to_wav_bytes() -> Callable[..., bytes]:
    """Return a function that encodes samples as WAV-file bytes (like an upload)."""

    def _to_wav_bytes(y: npt.NDArray[np.float32], sr: int = config.SAMPLE_RATE) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, y, sr, format="WAV")
        return buffer.getvalue()

    return _to_wav_bytes


@pytest.fixture
def make_silence() -> Callable[..., npt.NDArray[np.float32]]:
    """Return a function that generates digital silence (all zeros)."""

    def _make_silence(
        duration_s: float = 1.0, sr: int = config.SAMPLE_RATE
    ) -> npt.NDArray[np.float32]:
        return np.zeros(int(duration_s * sr), dtype=np.float32)

    return _make_silence
