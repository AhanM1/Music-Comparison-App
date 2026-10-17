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
