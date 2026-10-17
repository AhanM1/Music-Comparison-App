"""Typed data containers shared by every layer (the "contract" between modules).

Each model is a frozen dataclass: once created it can't be modified, so a clip
passed into one function can never be secretly changed by another. Functions
that "change" a clip return a new one instead.

More models (FeatureSet, ComparisonResult, ...) are added in the phase that
first needs them; see PLAN.md section 2.4.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class AudioClip:
    """A mono audio signal plus the info needed to interpret it.

    ``y`` is a 1-D array of samples, each a float roughly in [-1, 1], where 0 is
    silence. ``sr`` (sample rate) says how many samples make up one second.
    """

    y: npt.NDArray[np.float32]
    sr: int
    name: str

    def __post_init__(self) -> None:
        # Validate once, at creation, so every other function can trust the data.
        if self.y.ndim != 1:
            raise ValueError(f"AudioClip must be mono (1-D), got shape {self.y.shape}")
        if self.sr <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sr}")

    @property
    def duration_s(self) -> float:
        """Length in seconds = number of samples / samples per second."""
        return len(self.y) / self.sr


FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class FeatureSet:
    """Everything we measure about one recording, one value per analysis frame.

    A "frame" is a short window of audio (2048 samples ≈ 93 ms) and frames start
    every HOP_LENGTH samples (≈ 23 ms). Index ``i`` in every array refers to the
    same moment, ``times[i]`` seconds into the clip.
    """

    times: FloatArray  # frame centre times (s)
    f0_hz: FloatArray  # fundamental frequency (pitch); NaN where no pitch (unvoiced)
    voiced_prob: FloatArray  # 0..1 confidence that a pitch is present
    rms_db: FloatArray  # loudness in dB relative to full scale (0 dB = max)
    centroid_hz: FloatArray  # spectral centroid: "brightness"
    flatness: FloatArray  # spectral flatness 0..1: tone-like (0) vs noise-like (1)
    mfcc: FloatArray  # timbre fingerprint, shape (n_mfcc, n_frames)
    extractor_version: str  # bump when feature code changes, so stored data stays consistent

    def __post_init__(self) -> None:
        n = len(self.times)
        for field_name in ("f0_hz", "voiced_prob", "rms_db", "centroid_hz", "flatness"):
            length = len(getattr(self, field_name))
            if length != n:
                raise ValueError(f"{field_name} has {length} frames, expected {n}")
        if self.mfcc.ndim != 2 or self.mfcc.shape[1] != n:
            raise ValueError(f"mfcc must have shape (n_mfcc, {n}), got {self.mfcc.shape}")

    @property
    def n_frames(self) -> int:
        return len(self.times)
