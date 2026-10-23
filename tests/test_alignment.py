"""Tests for DTW alignment, using melodies whose timing we control exactly."""

import librosa
import numpy as np
import pytest

from musiccompare import config
from musiccompare.analysis.alignment import (
    align,
    downsample_chroma,
    dtw_resolution,
    is_probably_same_piece,
    moving_average,
    upsample_path,
)
from musiccompare.features.extract import extract_features
from musiccompare.models import Alignment, AudioClip, FeatureSet

SR = config.SAMPLE_RATE
FRAMES_PER_S = SR / config.HOP_LENGTH  # ≈ 43 frames per second

# "Twinkle Twinkle": a tune with repeated notes, which makes alignment harder.
TUNE = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]
OTHER_TUNE = [71, 66, 63, 70, 61, 68, 59, 72, 64, 57, 69, 62, 67, 58]


def features_of(y: np.ndarray) -> FeatureSet:
    return extract_features(AudioClip(y=y, sr=SR, name="test"))


def note_starts_s(durations: list[float]) -> np.ndarray:
    """Time (s) at which each note starts, given the note durations."""
    return np.concatenate([[0.0], np.cumsum(durations)[:-1]])


def note_midpoints_s(durations: list[float]) -> np.ndarray:
    """Time (s) of the middle of each note, given the note durations."""
    return note_starts_s(durations) + np.asarray(durations) / 2


def test_identical_recordings_align_on_the_diagonal(make_melody) -> None:
    fs = features_of(make_melody(TUNE))

    alignment = align(fs, fs)

    np.testing.assert_array_equal(alignment.ref_frames, np.arange(fs.n_frames))
    np.testing.assert_array_equal(alignment.user_frames, np.arange(fs.n_frames))
    assert alignment.cost == pytest.approx(0.0, abs=1e-6)


def test_uneven_tempo_maps_each_note_to_the_same_note(make_melody) -> None:
    ref_durs = [0.4] * len(TUNE)
    # The student drags the first half (0.6 s notes) then rushes the end (0.3 s).
    user_durs = [0.6] * 7 + [0.3] * 7
    ref = features_of(make_melody(TUNE, durations=ref_durs))
    user = features_of(make_melody(TUNE, durations=user_durs))

    alignment = align(ref, user)

    ref_mid_frames = np.round(note_midpoints_s(ref_durs) * FRAMES_PER_S).astype(int)
    expected_user = note_midpoints_s(user_durs) * FRAMES_PER_S
    mapped_user = alignment.user_frames[ref_mid_frames]
    # Within 3 frames (~70 ms) of the true middle of the same note.
    np.testing.assert_allclose(mapped_user, expected_user, atol=3)


def test_random_tempo_maps_note_starts_to_note_starts(make_melody) -> None:
    ref_durs = [0.4] * len(TUNE)
    user_durs = list(np.random.default_rng(7).uniform(0.2, 0.8, len(TUNE)))
    ref = features_of(make_melody(TUNE, durations=ref_durs))
    user = features_of(make_melody(TUNE, durations=user_durs))

    alignment = align(ref, user)

    # Skip the first note: it starts at frame 0 in both, which is trivially right.
    ref_start_frames = np.round(note_starts_s(ref_durs)[1:] * FRAMES_PER_S).astype(int)
    expected_user = note_starts_s(user_durs)[1:] * FRAMES_PER_S
    # Within 2 frames (~46 ms) of the real note start: about the limit of what
    # listeners notice as "late".
    np.testing.assert_allclose(alignment.user_frames[ref_start_frames], expected_user, atol=2)


def test_time_stretched_copy_aligns_along_a_straight_line(make_melody) -> None:
    y = make_melody(TUNE)
    slower = librosa.effects.time_stretch(y, rate=0.8)  # 1.25x longer
    ref, user = features_of(y), features_of(slower)

    alignment = align(ref, user)

    middle = slice(int(0.1 * ref.n_frames), int(0.9 * ref.n_frames))
    expected = alignment.ref_frames[middle] * 1.25
    np.testing.assert_allclose(alignment.user_frames[middle], expected, atol=4)


def test_coarse_resolution_still_aligns(make_melody) -> None:
    ref_durs, user_durs = [0.4] * len(TUNE), [0.6] * 7 + [0.3] * 7
    ref = features_of(make_melody(TUNE, durations=ref_durs))
    user = features_of(make_melody(TUNE, durations=user_durs))

    # A tiny memory budget forces DTW to run on frames averaged in groups.
    alignment = align(ref, user, max_cells=2_000)

    assert alignment.resolution >= 4
    ref_mid_frames = np.round(note_midpoints_s(ref_durs) * FRAMES_PER_S).astype(int)
    expected_user = note_midpoints_s(user_durs) * FRAMES_PER_S
    tolerance = 3 + alignment.resolution  # coarser grid -> proportionally looser
    np.testing.assert_allclose(alignment.user_frames[ref_mid_frames], expected_user, atol=tolerance)


def test_same_piece_detection(make_melody) -> None:
    ref = features_of(make_melody(TUNE))
    same_but_slower = features_of(make_melody(TUNE, durations=0.5, cents=20))
    different = features_of(make_melody(OTHER_TUNE))

    same_alignment = align(ref, same_but_slower)
    different_alignment = align(ref, different)

    assert is_probably_same_piece(same_alignment)
    assert not is_probably_same_piece(different_alignment)
    assert different_alignment.cost > 2 * same_alignment.cost


def test_path_has_one_row_per_reference_frame_and_never_goes_back(make_melody) -> None:
    ref = features_of(make_melody(TUNE))
    user = features_of(make_melody(TUNE, durations=0.55))

    alignment = align(ref, user)

    assert alignment.path.shape == (ref.n_frames, 2)
    assert np.all(np.diff(alignment.user_frames) >= 0)
    assert alignment.user_frames.max() <= user.n_frames - 1


# --- Helpers ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_ref", "n_user", "max_cells", "expected"),
    [
        (100, 100, 10_000, 1),
        (100, 100, 2_500, 2),
        (1_000, 4_000, 1_000_000, 2),
        (101, 101, 2_500, 3),
    ],
)
def test_dtw_resolution(n_ref: int, n_user: int, max_cells: int, expected: int) -> None:
    k = dtw_resolution(n_ref, n_user, max_cells)

    assert k == expected
    assert -(-n_ref // k) * -(-n_user // k) <= max_cells  # -(-a // b) is ceil(a / b)


def test_downsample_chroma_keeps_unit_columns() -> None:
    rng = np.random.default_rng(0)
    chroma = rng.random((12, 10))

    out = downsample_chroma(chroma, 3)

    assert out.shape == (12, 4)  # ceil(10 / 3)
    np.testing.assert_allclose(np.linalg.norm(out, axis=0), 1.0)


def test_moving_average_leaves_a_straight_line_unchanged() -> None:
    line = 3.0 + 1.25 * np.arange(50)

    np.testing.assert_allclose(moving_average(line, 15), line)


def test_moving_average_smooths_a_step_and_keeps_it_rising() -> None:
    step = np.concatenate([np.zeros(20), np.full(20, 10.0)])

    out = moving_average(step, 5)

    assert out.shape == step.shape
    assert np.all(np.diff(out) >= 0)
    np.testing.assert_allclose(out[17:23], [0, 2, 4, 6, 8, 10])


def test_moving_average_handles_short_input() -> None:
    np.testing.assert_allclose(moving_average(np.array([4.0]), 15), [4.0])
    np.testing.assert_allclose(moving_average(np.array([1.0, 2.0, 3.0]), 15), [1.0, 2.0, 3.0])


def test_upsample_path_identity_at_full_resolution() -> None:
    path = np.array([[0, 0], [1, 1], [1, 2], [2, 3]])

    full = upsample_path(path, k=1, n_ref=3, n_user=4)

    # ref frame 1 was matched to user frames 1 and 2 -> average 1.5 rounds to 2
    np.testing.assert_array_equal(full, [[0, 0], [1, 2], [2, 3]])


def test_alignment_rejects_backwards_path() -> None:
    with pytest.raises(ValueError, match="backwards"):
        Alignment(path=np.array([[0, 2], [1, 1]]), cost=0.0, resolution=1)
