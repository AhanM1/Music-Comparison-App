"""Compare two aligned recordings, frame by frame, on pitch, dynamics, and tone.

Alignment (Phase 3) tells us, for every reference frame, which user frame was
playing "the same moment". Here we read both feature sets at those matching
frames and subtract. Everything is on the reference's timeline.

Each dimension is measured in a unit that matches how we hear it:

- Pitch in **cents** (1200 * log2 of the frequency ratio). Whole octaves are
  removed, because a pitch tracker sometimes jumps an octave on a note that
  was actually played correctly.
- Dynamics in **dB**, but as a *contour*: different microphones and rooms make
  whole recordings louder or quieter, so we first subtract the median
  difference (the overall level offset) and compare only the shape. Did you
  get louder and softer where the reference did?
- Tone as **brightness** in octaves: log2 of the spectral-centroid ratio.
  +0.25 means the centroid was 2^0.25 ≈ 19% higher (brighter).

Coverage. Comparing "the frames where both have a pitch" would let a take
that skips half the piece score 100: the skipped half is simply never
compared. So reference notes with nothing from the user to compare them to
(``FrameDifferences.missing``) count as 0 for pitch and tone, and are kept out
of every dimension's problem regions; the feedback reports them once, as
missing notes, instead of as a pile of "43 dB softer" spots.

Known limit: alignment matches whatever *sounds* most alike, so a wrong note
right next to a repeat of the right note (C C D played as C D D) can be
partly "explained away" by stretching the matching notes over it.
"""

import numpy as np

from musiccompare import config
from musiccompare.analysis.quality import busy_reference
from musiccompare.analysis.scoring import score
from musiccompare.analysis.segments import (
    FRAME_S,
    SMOOTHING_WIDTH,
    find_regions,
    find_runs,
    median_smooth,
)
from musiccompare.features.pitch import fold_octaves, hz_to_cents
from musiccompare.models import (
    DIMENSIONS,
    Alignment,
    BoolArray,
    ComparisonResult,
    Dimension,
    FeatureSet,
    FloatArray,
    FrameDifferences,
    ProblemRegion,
)

# Dimensions where a missing note counts as 0. Dynamics is left out: the
# loudness of a note that wasn't played isn't a separate mistake.
SCORED_AS_MISSED: tuple[Dimension, ...] = ("pitch", "tone")

# (tolerance, difference that scores 0) per dimension.
LIMITS: dict[Dimension, tuple[float, float]] = {
    "pitch": (config.PITCH_TOLERANCE_CENTS, config.PITCH_ZERO_SCORE_CENTS),
    "dynamics": (config.DYNAMICS_TOLERANCE_DB, config.DYNAMICS_ZERO_SCORE_DB),
    "tone": (config.TONE_TOLERANCE_OCTAVES, config.TONE_ZERO_SCORE_OCTAVES),
}


def compare(ref: FeatureSet, user: FeatureSet, alignment: Alignment) -> ComparisonResult:
    """Differences, a 0-100 score per dimension, and the problem regions."""
    differences = frame_differences(ref, user, alignment)
    user_times = user.times[alignment.user_frames]

    missing = differences.missing

    scores: dict[Dimension, float | None] = {}
    regions: list[ProblemRegion] = []
    for dimension in DIMENSIONS:
        smoothed = median_smooth(differences.errors(dimension), SMOOTHING_WIDTH)
        smoothed[missing] = np.nan  # reported once, as missing notes, not per dimension
        tolerance, zero_at = LIMITS[dimension]
        missed = int(missing.sum()) if dimension in SCORED_AS_MISSED else 0
        scores[dimension] = score(smoothed, tolerance, zero_at, missed_frames=missed)
        regions += find_regions(dimension, differences.times, smoothed, tolerance, user_times)

    regions.sort(key=lambda r: r.start_s)
    ref_notes = int(np.isfinite(ref.f0_hz).sum())
    coverage = 1.0 - (missing & np.isfinite(ref.f0_hz)).sum() / ref_notes if ref_notes else 1.0
    return ComparisonResult(alignment, differences, scores, regions, coverage=float(coverage))


def frame_differences(ref: FeatureSet, user: FeatureSet, alignment: Alignment) -> FrameDifferences:
    """User minus reference at every reference frame (NaN where there's nothing to compare)."""
    u = alignment.user_frames  # the user frame matching each reference frame

    # Pitch: NaN wherever either side has no pitch (hz_to_cents passes NaN through).
    raw_cents = hz_to_cents(user.f0_hz[u], ref.f0_hz)
    pitch_cents = fold_octaves(raw_cents)
    octave_shift = overall_octave_shift(raw_cents)
    if busy_reference(ref, user):
        # With accompaniment under the reference, its pitch tracker may follow a
        # bass line or a drone instead of the melody. Then its octave can't be
        # trusted, and a "wrong note" (a semitone or more off) is more likely the
        # tracker reading a harmony note than you: only intonation is judged.
        octave_shift = 0
        with np.errstate(invalid="ignore"):  # NaN compares as False
            pitch_cents[np.abs(pitch_cents) >= config.WRONG_NOTE_CENTS] = np.nan

    # Dynamics: compare wherever at least one side is playing. Silence is floored
    # at SOUNDING_DB, so "you stopped while the reference kept playing" counts as
    # a big but finite difference.
    ref_db = np.maximum(ref.rms_db, config.SOUNDING_DB).astype(np.float64)
    user_db = np.maximum(user.rms_db[u], config.SOUNDING_DB).astype(np.float64)
    ref_on = ref.rms_db > config.SOUNDING_DB
    user_on = user.rms_db[u] > config.SOUNDING_DB
    both_on = ref_on & user_on
    level_offset = float(np.median(user_db[both_on] - ref_db[both_on])) if both_on.any() else 0.0
    dynamics_db = np.where(ref_on | user_on, user_db - ref_db - level_offset, np.nan)

    # Tone: only meaningful while both are playing (a centroid of silence is noise).
    # Every instrument's centroid rises with pitch, so playing the whole piece an
    # octave up would read as "an octave brighter"; subtracting octave_shift
    # compares brightness for the same notes instead.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log2(user.centroid_hz[u] / ref.centroid_hz) - octave_shift
    tone_octaves = np.where(both_on & np.isfinite(ratio), ratio, np.nan)

    return FrameDifferences(
        times=ref.times,
        pitch_cents=pitch_cents,
        dynamics_db=dynamics_db,
        tone_octaves=tone_octaves,
        level_offset_db=level_offset,
        octave_shift=octave_shift,
        missing=missing_notes(ref, user, alignment),
    )


def overall_octave_shift(raw_cents: FloatArray) -> int:
    """Whole octaves the user played above the reference, e.g. -1 for a guitar
    playing a viola line an octave down.

    Only counted when most compared notes really are a whole number of octaves
    apart (within a semitone): then the most common shift wins, ignoring the
    pitch tracker's occasional single-note octave mistakes. When notes aren't
    octave-related at all (a reference whose tracker follows a drone), the
    shift stays 0 instead of making the whole comparison an octave off.
    """
    pitched = raw_cents[np.isfinite(raw_cents)]
    if pitched.size == 0:
        return 0
    octaves = np.round(pitched / 1200)
    near_octave = np.abs(pitched - 1200 * octaves) < 100
    if near_octave.mean() < config.OCTAVE_AGREEMENT:
        return 0
    values, counts = np.unique(octaves[near_octave], return_counts=True)
    return int(values[np.argmax(counts)])


def missing_notes(ref: FeatureSet, user: FeatureSet, alignment: Alignment) -> BoolArray:
    """Reference frames playing a note that the take has nothing to compare with.

    Two causes: no clear note was heard from the user (a rest where there
    shouldn't be one, a stopped or noisy passage) for at least
    ``config.MISSING_MIN_S``; or the alignment found that stretch of the
    reference skipped altogether.
    """
    ref_note = np.isfinite(ref.f0_hz)
    unheard = ref_note & ~np.isfinite(user.f0_hz[alignment.user_frames])
    # Majority vote over a short window: a pitch tracker drops out for a frame
    # or two at every note change, and that isn't a missed note.
    unheard = median_smooth(unheard.astype(np.float64), SMOOTHING_WIDTH) > 0.5
    missing = np.zeros(ref.n_frames, dtype=bool)
    min_frames = round(config.MISSING_MIN_S / FRAME_S)
    for start, stop in find_runs(unheard):
        if stop - start >= min_frames:
            missing[start:stop] = True
    for start, stop in alignment.skipped_ref:
        missing[start:stop] = True
    return missing & ref_note
