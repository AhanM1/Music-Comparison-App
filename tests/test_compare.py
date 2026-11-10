"""Tests for comparison, problem regions, and scoring.

The end-to-end tests build a "student" take with a known mistake (a sharp
passage, a missing crescendo, a darker tone, ...) and check that the mistake,
and only the mistake, is found in the right place.
"""

from dataclasses import replace

import numpy as np
import pytest

from musiccompare import config
from musiccompare.analysis.alignment import align
from musiccompare.analysis.compare import compare
from musiccompare.analysis.scoring import score
from musiccompare.analysis.segments import (
    FRAME_S,
    find_regions,
    find_runs,
    median_smooth,
    merge_runs,
    severity,
)
from musiccompare.features.extract import extract_features
from musiccompare.models import AudioClip, ComparisonResult, FeatureSet, FrameDifferences

SR = config.SAMPLE_RATE
NOTE_S = 0.4  # make_melody's default note length
# A tune without repeated notes, 12 notes x 0.4 s = 4.8 s.
TUNE = [60, 64, 67, 72, 71, 67, 65, 69, 74, 72, 67, 64]


def features_of(y: np.ndarray) -> FeatureSet:
    return extract_features(AudioClip(y=y, sr=SR, name="test"))


def compare_takes(ref_y: np.ndarray, user_y: np.ndarray) -> ComparisonResult:
    ref, user = features_of(ref_y), features_of(user_y)
    return compare(ref, user, align(ref, user))


def regions_of(result: ComparisonResult, dimension: str) -> list:
    return [r for r in result.regions if r.dimension == dimension]


# --- median_smooth -----------------------------------------------------------


def test_median_smooth_removes_short_spikes_but_keeps_real_steps() -> None:
    x = np.zeros(40)
    x[10:12] = 500.0  # 2-frame glitch
    x[25:] = 30.0  # sustained change
    smoothed = median_smooth(x, width=11)

    assert np.all(smoothed[:20] == 0)  # glitch gone
    assert np.all(smoothed[:24] == 0) and np.all(smoothed[26:] == 30)  # step stays sharp


def test_median_smooth_skips_nan_and_never_fills_it_in() -> None:
    x = np.array([10.0, 10.0, np.nan, 10.0, np.nan, np.nan, 10.0])
    smoothed = median_smooth(x, width=3)

    np.testing.assert_array_equal(np.isnan(smoothed), np.isnan(x))
    assert np.all(smoothed[~np.isnan(x)] == 10.0)


def test_median_smooth_width_one_is_identity() -> None:
    x = np.array([1.0, 5.0, 2.0])
    np.testing.assert_array_equal(median_smooth(x, width=1), x)


# --- runs and regions ---------------------------------------------------------


def test_find_runs() -> None:
    mask = np.array([False, True, True, False, True])
    assert find_runs(mask) == [(1, 3), (4, 5)]
    assert find_runs(np.array([True, True])) == [(0, 2)]
    assert find_runs(np.array([False])) == []


def test_merge_runs_joins_only_small_gaps() -> None:
    times = np.arange(100) * 0.1  # 10 frames per second
    runs = [(0, 10), (11, 20), (40, 50)]  # gaps of 0.1 s and 2.0 s

    assert merge_runs(runs, times, max_gap_s=0.2) == [(0, 20), (40, 50)]


def test_severity_grows_with_error() -> None:
    assert severity(15.0, tolerance=15.0) == 0.0
    assert severity(-30.0, tolerance=15.0) == 0.5
    assert severity(60.0, tolerance=15.0) == 0.75
    assert severity(0.0, tolerance=15.0) == 0.0


def test_find_regions_splits_sharp_from_flat_and_maps_user_time() -> None:
    n = 200
    times = np.arange(n) * FRAME_S
    errors = np.zeros(n)
    errors[20:60] = 40.0  # sharp
    errors[60:100] = -40.0  # immediately followed by flat
    errors[150:153] = 40.0  # too short to report
    user_times = times * 1.5  # the user played 1.5x slower

    regions = find_regions("pitch", times, errors, tolerance=15.0, user_times=user_times)

    assert [round(r.mean_error) for r in regions] == [40, -40]
    sharp = regions[0]
    assert sharp.start_s == pytest.approx(times[20])
    assert sharp.end_s == pytest.approx(times[60])
    assert sharp.user_start_s == pytest.approx(1.5 * times[20])
    assert sharp.severity == pytest.approx(1 - 15 / 40)


def test_find_regions_ignores_nan() -> None:
    times = np.arange(50) * FRAME_S
    assert find_regions("tone", times, np.full(50, np.nan), 0.25, times) == []


# --- scoring ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("errors", "expected"),
    [
        ([0.0, 10.0, -15.0], 100.0),  # all within tolerance
        ([50.0, -80.0, 500.0], 0.0),  # all at or past zero_at; capped, not negative
        ([32.5, -32.5], 50.0),  # halfway between 15 and 50
        ([0.0, np.nan, 80.0], 50.0),  # NaN ignored: one perfect, one zero
    ],
)
def test_score(errors: list[float], expected: float) -> None:
    assert score(np.array(errors), tolerance=15.0, zero_at=50.0) == pytest.approx(expected)


def test_missed_frames_count_as_zero() -> None:
    """Two perfect frames and two missed ones: half the piece matched."""
    assert score(np.zeros(2), tolerance=15.0, zero_at=50.0, missed_frames=2) == 50.0
    assert score(np.array([np.nan]), tolerance=15.0, zero_at=50.0, missed_frames=3) == 0.0


def test_score_is_none_when_nothing_to_compare() -> None:
    assert score(np.array([np.nan, np.nan]), tolerance=15.0, zero_at=50.0) is None


# --- FrameDifferences -------------------------------------------------------


def test_frame_differences_checks_lengths() -> None:
    with pytest.raises(ValueError, match="tone_octaves has 3 frames, expected 4"):
        FrameDifferences(
            np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3), 0.0, 0, np.zeros(4, bool)
        )


# --- End to end: known mistakes ---------------------------------------------


def test_identical_takes_score_100_with_no_problems(make_melody) -> None:
    y = make_melody(TUNE)
    result = compare_takes(y, y)

    assert result.scores == {"pitch": 100.0, "dynamics": 100.0, "tone": 100.0}
    assert result.regions == []


def test_different_tempo_alone_is_not_a_problem(make_melody) -> None:
    durations = list(np.resize([0.3, 0.55, 0.4, 0.5], len(TUNE)))
    result = compare_takes(make_melody(TUNE), make_melody(TUNE, durations=durations))

    assert all(s is not None and s >= 95 for s in result.scores.values()), result.scores
    assert result.regions == []


def test_out_of_step_vibrato_is_not_out_of_tune(make_melody) -> None:
    # Same notes, both with ±25-cent vibrato, but at different speeds, so at
    # any instant one may be 50 cents above the other. On average both are in tune.
    ref = make_melody(TUNE, vibrato_cents=25, vibrato_hz=5.0)
    user = make_melody(TUNE, vibrato_cents=25, vibrato_hz=6.3)
    result = compare_takes(ref, user)

    assert result.scores["pitch"] is not None and result.scores["pitch"] >= 90
    assert regions_of(result, "pitch") == []


def test_sharp_passage_is_found_in_the_right_place(make_melody) -> None:
    cents = [0.0] * len(TUNE)
    cents[4:8] = [35.0] * 4  # notes 5-8 (1.6 s - 3.2 s) played 35 cents sharp
    result = compare_takes(make_melody(TUNE), make_melody(TUNE, cents=cents))

    [region] = regions_of(result, "pitch")
    assert region.start_s == pytest.approx(4 * NOTE_S, abs=0.1)
    assert region.end_s == pytest.approx(8 * NOTE_S, abs=0.1)
    assert region.mean_error == pytest.approx(35, abs=3)
    # A third of the frames are 35 cents off, losing (35 - 15) / (50 - 15) of their credit.
    assert result.scores["pitch"] == pytest.approx(100 - 100 / 3 * (35 - 15) / (50 - 15), abs=5)


def test_flat_then_sharp_gives_two_regions(make_melody) -> None:
    cents = [0.0] * len(TUNE)
    cents[2:5] = [-40.0] * 3
    cents[5:8] = [40.0] * 3
    result = compare_takes(make_melody(TUNE), make_melody(TUNE, cents=cents))

    signs = [np.sign(r.mean_error) for r in regions_of(result, "pitch")]
    assert signs == [-1, 1]


def test_octave_slip_is_not_a_pitch_or_tone_error(make_melody) -> None:
    # The same tune an octave up: musically the same notes, and the same tone
    # (its centroid doubles, but so does its pitch).
    result = compare_takes(make_melody(TUNE), make_melody([m + 12 for m in TUNE]))

    assert result.differences.octave_shift == 1
    assert result.regions == []
    assert result.scores["tone"] == pytest.approx(100, abs=2)


def test_pitch_tracker_octave_mistake_does_not_fake_a_tone_change(make_melody) -> None:
    ref, user = features_of(make_melody(TUNE)), features_of(make_melody(TUNE))
    # Simulate the tracker reading notes 4-6 of the user's take an octave low.
    f0 = user.f0_hz.copy()
    f0[(user.times >= 4 * NOTE_S) & (user.times < 7 * NOTE_S)] /= 2
    user = replace(user, f0_hz=f0)

    result = compare(ref, user, align(ref, user))

    assert result.differences.octave_shift == 0
    assert result.regions == []


def test_quieter_recording_overall_is_not_a_dynamics_problem(make_melody) -> None:
    # 10 dB quieter everywhere, like a mic further away: same dynamic *shape*.
    result = compare_takes(make_melody(TUNE), make_melody(TUNE, amplitudes=0.5 * 10 ** (-10 / 20)))

    assert result.differences.level_offset_db == pytest.approx(-10, abs=0.5)
    assert result.scores["dynamics"] == pytest.approx(100, abs=1)
    assert regions_of(result, "dynamics") == []


def test_missing_soft_passage_is_a_dynamics_problem(make_melody) -> None:
    # The reference plays notes 7-10 softly (-12 dB); the user plays everything loud.
    amps = [0.5] * len(TUNE)
    amps[6:10] = [0.5 * 10 ** (-12 / 20)] * 4
    result = compare_takes(make_melody(TUNE, amplitudes=amps), make_melody(TUNE))

    [region] = regions_of(result, "dynamics")
    assert region.mean_error == pytest.approx(12, abs=1.5)  # user was louder
    assert region.start_s == pytest.approx(6 * NOTE_S, abs=0.1)
    assert region.end_s == pytest.approx(10 * NOTE_S, abs=0.1)
    assert regions_of(result, "pitch") == []


def test_darker_tone_lowers_tone_score_only(make_melody) -> None:
    ref = make_melody(TUNE, overtones=(1.0, 0.8, 0.8, 0.7, 0.6))
    darker = make_melody(TUNE, overtones=(1.0, 0.2, 0.05))
    result = compare_takes(ref, darker)

    tone_regions = regions_of(result, "tone")
    assert tone_regions and all(r.mean_error < -config.TONE_TOLERANCE_OCTAVES for r in tone_regions)
    assert result.scores["tone"] is not None and result.scores["tone"] < 60
    assert result.scores["pitch"] == pytest.approx(100, abs=2)


def test_regions_are_sorted_by_time(make_melody) -> None:
    cents = [0.0] * len(TUNE)
    cents[8:11] = [45.0] * 3
    amps = [0.5] * len(TUNE)
    amps[1:4] = [0.1] * 3
    result = compare_takes(make_melody(TUNE), make_melody(TUNE, cents=cents, amplitudes=amps))

    starts = [r.start_s for r in result.regions]
    assert {r.dimension for r in result.regions} >= {"pitch", "dynamics"}
    assert starts == sorted(starts)


# --- Coverage: missing notes count against you (QA report H3) -----------------


def test_stopping_for_half_the_piece_halves_pitch_and_tone(make_melody) -> None:
    """Silence for six notes used to score 100: those notes were never compared."""
    ref = make_melody(TUNE)
    user = ref.copy()
    user[int(3 * NOTE_S * SR) : int(9 * NOTE_S * SR)] = 0  # stops for notes 4-9

    result = compare_takes(ref, user)

    assert result.scores["pitch"] == pytest.approx(50, abs=8)
    assert result.scores["tone"] == pytest.approx(50, abs=8)
    assert result.coverage == pytest.approx(0.5, abs=0.08)
    missing_s = result.differences.times[result.differences.missing]
    assert missing_s.min() == pytest.approx(3 * NOTE_S, abs=0.15)
    assert missing_s.max() == pytest.approx(9 * NOTE_S, abs=0.15)
    # Reported once (as missing notes), not as a "43 dB softer" dynamics spot.
    assert not [r for r in result.regions if r.start_s < 9 * NOTE_S and r.end_s > 3 * NOTE_S]


def test_a_complete_take_has_full_coverage(make_melody) -> None:
    result = compare_takes(make_melody(TUNE), make_melody(TUNE, durations=0.5, vibrato_cents=20))
    assert result.coverage == 1.0
    assert not result.differences.missing.any()


def test_a_brief_pitch_dropout_is_not_a_missing_note(make_melody) -> None:
    ref = make_melody(TUNE)
    user = ref.copy()
    gap = int(1.0 * SR)
    user[gap : gap + int(0.15 * SR)] = 0  # 0.15 s hole: shorter than MISSING_MIN_S
    assert compare_takes(ref, user).coverage == 1.0


# --- Octave shift is only trusted when the notes agree --------------------------


def test_octave_shift_needs_most_notes_to_agree() -> None:
    from musiccompare.analysis.compare import overall_octave_shift

    assert overall_octave_shift(np.full(100, -1200.0)) == -1  # everything an octave down
    # One octave mistake by the pitch tracker in an otherwise same-octave take:
    assert overall_octave_shift(np.r_[np.full(90, 10.0), np.full(10, 1200.0)]) == 0
    # Mostly not octave-related (e.g. a tracker following a drone): no shift.
    assert overall_octave_shift(np.r_[np.full(70, -498.0), np.full(30, 1200.0)]) == 0
    assert overall_octave_shift(np.array([np.nan])) == 0


def test_busy_reference_is_not_compared_on_harmony_readings(make_melody) -> None:
    """A drone under the melody: the tracker reads it, not the melody. No fake
    octave, and no "wrong note" readings (only intonation is judged)."""
    melody = make_melody(TUNE)
    t = np.arange(melody.size) / SR
    drone = 0.25 * np.sin(2 * np.pi * 110 * t) + 0.15 * np.sin(2 * np.pi * 165 * t)

    result = compare_takes((melody + drone).astype(np.float32), melody)

    assert result.differences.octave_shift == 0
    cents = result.differences.pitch_cents
    assert np.all(np.abs(cents[np.isfinite(cents)]) < config.WRONG_NOTE_CENTS)
    assert result.scores["pitch"] == pytest.approx(100, abs=5)
