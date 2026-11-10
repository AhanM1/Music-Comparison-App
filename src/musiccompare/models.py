"""Typed data containers shared by every layer (the "contract" between modules).

Each model is a frozen dataclass: once created it can't be modified, so a clip
passed into one function can never be secretly changed by another. Functions
that "change" a clip return a new one instead.

More models (FeatureSet, ComparisonResult, ...) are added in the phase that
first needs them; see PLAN.md section 2.4.
"""

from dataclasses import dataclass, field
from typing import Literal

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
    chroma: FloatArray  # pitch-class profile (C, C#, ... B), shape (12, n_frames)
    extractor_version: str  # bump when feature code changes, so stored data stays consistent

    def __post_init__(self) -> None:
        n = len(self.times)
        for field_name in ("f0_hz", "voiced_prob", "rms_db", "centroid_hz", "flatness"):
            length = len(getattr(self, field_name))
            if length != n:
                raise ValueError(f"{field_name} has {length} frames, expected {n}")
        if self.mfcc.ndim != 2 or self.mfcc.shape[1] != n:
            raise ValueError(f"mfcc must have shape (n_mfcc, {n}), got {self.mfcc.shape}")
        if self.chroma.shape != (12, n):
            raise ValueError(f"chroma must have shape (12, {n}), got {self.chroma.shape}")

    @property
    def n_frames(self) -> int:
        return len(self.times)


IntArray = npt.NDArray[np.intp]
BoolArray = npt.NDArray[np.bool_]


def no_spans() -> IntArray:
    """An empty list of (start, stop) frame spans, shape (0, 2)."""
    return np.empty((0, 2), dtype=np.intp)


@dataclass(frozen=True)
class Alignment:
    """How the user's recording lines up in time with the reference.

    ``path[i] = (ref_frame, user_frame)``: at reference frame ``ref_frame`` the
    user was at ``user_frame``. There is exactly one row per reference frame, in
    order, so ``path[:, 1]`` maps the reference timeline onto the user's.

    Not everything can be matched. ``extra_user`` lists stretches of the user's
    take with nothing like them in the reference (a count-in, a repeat, extra
    notes), and ``skipped_ref`` stretches of the reference with nothing like
    them in the take (skipped, or different notes). Both are (start, stop)
    frame spans, stop exclusive: user frames and reference frames respectively.
    """

    path: IntArray  # shape (n_ref_frames, 2)
    cost: float  # mean chroma distance along the path: 0 = identical, ~1 = unrelated
    resolution: int  # DTW ran on frames averaged in groups of this size (1 = full)
    extra_user: IntArray = field(default_factory=no_spans)  # shape (k, 2), user frames
    skipped_ref: IntArray = field(default_factory=no_spans)  # shape (k, 2), reference frames

    def __post_init__(self) -> None:
        if self.path.ndim != 2 or self.path.shape[1] != 2:
            raise ValueError(f"path must have shape (n, 2), got {self.path.shape}")
        if np.any(np.diff(self.path, axis=0) < 0):
            raise ValueError("path must never go backwards in time")
        for name in ("extra_user", "skipped_ref"):
            spans = getattr(self, name)
            if spans.ndim != 2 or spans.shape[1] != 2:
                raise ValueError(f"{name} must have shape (k, 2), got {spans.shape}")

    @property
    def ref_frames(self) -> IntArray:
        return self.path[:, 0]

    @property
    def user_frames(self) -> IntArray:
        return self.path[:, 1]


# The three things we compare. A Literal type means mypy rejects typos like "pich".
Dimension = Literal["pitch", "dynamics", "tone"]
DIMENSIONS: tuple[Dimension, ...] = ("pitch", "dynamics", "tone")


@dataclass(frozen=True)
class FrameDifferences:
    """How the user differs from the reference, at every frame of the reference.

    Positive = the user is sharp / louder / brighter; negative = flat / quieter /
    darker. NaN = nothing to compare at that moment (a rest, or no clear pitch).

    ``missing`` marks moments where the reference plays a note but the take has
    nothing to compare it with: no clear note was heard (for longer than a
    brief dropout), or that part of the reference was skipped. Those count as
    0 for pitch and tone, instead of quietly being left out of the scores.
    """

    times: FloatArray  # reference timeline (s)
    pitch_cents: FloatArray  # pitch difference, whole octaves removed
    dynamics_db: FloatArray  # loudness-contour difference, overall level offset removed
    tone_octaves: FloatArray  # brightness (spectral centroid) difference in octaves
    level_offset_db: float  # how much louder the user was overall (removed from dynamics_db)
    octave_shift: int  # whole octaves the user played above the reference (removed from tone)
    missing: BoolArray  # reference notes with nothing from the user to compare them to

    def __post_init__(self) -> None:
        n = len(self.times)
        for name in ("pitch_cents", "dynamics_db", "tone_octaves", "missing"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} frames, expected {n}")

    def errors(self, dimension: Dimension) -> FloatArray:
        """The difference curve for one dimension."""
        return {
            "pitch": self.pitch_cents,
            "dynamics": self.dynamics_db,
            "tone": self.tone_octaves,
        }[dimension]


@dataclass(frozen=True)
class ProblemRegion:
    """A stretch of time where one dimension was consistently off in one direction."""

    dimension: Dimension
    start_s: float  # on the reference's timeline
    end_s: float
    user_start_s: float  # the same stretch on the user's timeline
    user_end_s: float
    mean_error: float  # signed, in cents / dB / octaves (see FrameDifferences)
    severity: float  # 0 = just past the tolerance ... 1 = far past it

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class ComparisonResult:
    """Everything Phase 4 learns from comparing two aligned recordings."""

    alignment: Alignment
    differences: FrameDifferences
    scores: dict[Dimension, float | None]  # 0-100 per dimension; None = nothing to compare
    regions: list[ProblemRegion]  # sorted by start time
    # Share (0..1) of the reference's notes that had something from the user to
    # compare against. 1.0 = the whole piece was played.
    coverage: float = 1.0


Instrument = Literal["viola", "acoustic_guitar", "electric_guitar", "other"]
INSTRUMENTS: tuple[Instrument, ...] = ("viola", "acoustic_guitar", "electric_guitar", "other")
SkillLevel = Literal["beginner", "intermediate", "advanced"]
SKILL_LEVELS: tuple[SkillLevel, ...] = ("beginner", "intermediate", "advanced")


@dataclass(frozen=True)
class PlayerContext:
    """Who is playing what. Rules use the instrument to pick tips.

    Skill level and piece barely matter to the rules yet, but a future ML model
    will want them, and they can't be collected after the fact.
    """

    instrument: Instrument = "other"
    skill_level: SkillLevel = "intermediate"
    piece: str = ""


@dataclass(frozen=True)
class FeedbackItem:
    """One piece of advice: what happened, and what to try."""

    dimension: Dimension
    message: str  # what we measured, e.g. "At 0:01.6 – 0:03.2 you were 35 cents sharp."
    tip: str  # what to try, e.g. "Practise this passage slowly against a drone..."
    importance: float  # 0..1: how much fixing this would improve the take
    confidence: float  # 0..1: how sure we are that the measurement is right
    source: str  # which provider wrote it, e.g. "rules-v1" (later "ml-v1")
    region: ProblemRegion | None = None  # None = a habit across the whole take

    def __post_init__(self) -> None:
        for name in ("importance", "confidence"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in 0..1, got {getattr(self, name)}")

    @property
    def rank(self) -> float:
        """Sort key: worth fixing *and* likely real comes first."""
        return self.importance * self.confidence
