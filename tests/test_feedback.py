"""Tests for the feedback layer: wording, the provider interface, and the rules.

Like test_compare.py, the end-to-end tests build a "student" take with one
known mistake and check that the advice names that mistake, and only that.
"""

import numpy as np
import pytest

from musiccompare import config
from musiccompare.analysis.alignment import align
from musiccompare.analysis.compare import compare
from musiccompare.features.extract import extract_features
from musiccompare.feedback.base import FeedbackProvider
from musiccompare.feedback.rules import (
    ELECTRIC_TONE_NOTE,
    SOURCE,
    SPOT_TIPS,
    RuleBasedFeedback,
    explains_spots,
    family,
    importance,
)
from musiccompare.feedback.wording import describe, length_change, time_span
from musiccompare.models import (
    AudioClip,
    FeedbackItem,
    Instrument,
    PlayerContext,
    ProblemRegion,
)

SR = config.SAMPLE_RATE
TUNE = [60, 64, 67, 72, 71, 67, 65, 69, 74, 72, 67, 64]  # 12 notes x 0.4 s = 4.8 s
VIOLA = PlayerContext(instrument="viola")


def feedback(
    ref_y: np.ndarray, user_y: np.ndarray, instrument: Instrument = "viola"
) -> list[FeedbackItem]:
    ref = extract_features(AudioClip(y=ref_y, sr=SR, name="ref"))
    user = extract_features(AudioClip(y=user_y, sr=SR, name="me"))
    result = compare(ref, user, align(ref, user))
    return RuleBasedFeedback().generate(ref, user, result, PlayerContext(instrument=instrument))


def with_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    """``y`` plus white noise ``snr_db`` dB below it (fixed seed, so tests repeat)."""
    noise_rms = np.sqrt(np.mean(y**2)) * 10 ** (-snr_db / 20)
    noise = np.random.default_rng(0).normal(0, noise_rms, len(y))
    return (y + noise).astype(np.float32)


def region(dimension: str = "pitch", mean_error: float = 30.0) -> ProblemRegion:
    return ProblemRegion(dimension, 1.0, 2.0, 1.1, 2.3, mean_error, severity=0.5)  # type: ignore[arg-type]


# --- Wording -----------------------------------------------------------------


def test_time_span() -> None:
    assert time_span(3.2, 65.0) == "0:03.2 – 1:05.0"


@pytest.mark.parametrize(
    ("user_s", "ref_s", "expected"),
    [
        (5.0, 4.0, "+25% length"),
        (3.0, 4.0, "-25% length"),
        (8.0, 4.0, "+100% length"),
        (10.5, 4.0, "2.6x as long"),
        (1.0, 4.0, "4.0x shorter"),
    ],
)
def test_length_change(user_s: float, ref_s: float, expected: str) -> None:
    """Percentages for tempo differences, ratios once they'd become absurd."""
    assert length_change(user_s, ref_s) == expected


@pytest.mark.parametrize(
    ("dimension", "error", "expected"),
    [
        ("pitch", 32.4, "32 cents sharp"),
        ("pitch", -20.0, "20 cents flat"),
        ("dynamics", 3.04, "3.0 dB louder"),
        ("dynamics", -6.0, "6.0 dB softer"),
        ("tone", 1.0, "100% brighter"),  # +1 octave of centroid = twice as bright
        ("tone", -1.0, "50% darker"),
    ],
)
def test_describe(dimension: str, error: float, expected: str) -> None:
    assert describe(region(dimension, error)) == expected


def test_explains_spots() -> None:
    assert explains_spots(0, "sharp") == ""
    assert explains_spots(1, "sharp") == " This explains the sharp spot in the problem-spot list."
    assert explains_spots(3, "sharp") == " This explains 3 sharp spots in the problem-spot list."


# --- Models and ranking ------------------------------------------------------


def test_feedback_item_checks_its_ranges() -> None:
    with pytest.raises(ValueError, match="importance"):
        FeedbackItem("pitch", "m", "t", importance=1.5, confidence=0.5, source="x")
    with pytest.raises(ValueError, match="confidence"):
        FeedbackItem("pitch", "m", "t", importance=0.5, confidence=-0.1, source="x")


def test_rank_is_importance_times_confidence() -> None:
    item = FeedbackItem("pitch", "m", "t", importance=0.5, confidence=0.8, source="x")
    assert item.rank == pytest.approx(0.4)


def test_whole_piece_habit_counts_twice_a_short_spot() -> None:
    assert importance("pitch", 0.5, reach=1.0) == pytest.approx(2 * importance("pitch", 0.5, 0.0))
    assert importance("tone", 0.5, 1.0) < importance("pitch", 0.5, 1.0)  # weighted by dimension


def test_player_context_defaults() -> None:
    assert PlayerContext() == PlayerContext("other", "intermediate", "")


def test_instrument_families() -> None:
    assert family("viola") == "bowed"
    assert family("acoustic_guitar") == family("electric_guitar") == "guitar"
    assert family("other") == "other"


def test_rule_based_feedback_fits_the_provider_socket() -> None:
    assert isinstance(RuleBasedFeedback(), FeedbackProvider)
    assert RuleBasedFeedback.source == SOURCE == "rules-v1"


# --- Rules, end to end --------------------------------------------------------


def test_identical_takes_get_no_advice(make_melody) -> None:
    assert feedback(make_melody(TUNE), make_melody(TUNE)) == []


def test_problem_spots_get_tips_for_their_direction(make_melody) -> None:
    cents = [0, 0, 40, 40, 0, 0, 0, -40, -40, 0, 0, 0]  # sharp at 0.8-1.6 s, flat at 2.8-3.6 s
    items = feedback(make_melody(TUNE), make_melody(TUNE, cents=cents))

    assert len(items) == 2
    by_time = sorted(items, key=lambda i: i.region.start_s if i.region else 0)
    sharp, flat = by_time
    assert sharp.message == "At 0:00.8 – 0:01.6 you were 39 cents sharp."
    assert sharp.tip == SPOT_TIPS[("pitch", True)]["bowed"]
    assert flat.tip == SPOT_TIPS[("pitch", False)]["bowed"]
    assert all(i.source == SOURCE and i.region is not None for i in items)


def test_tips_depend_on_the_instrument(make_melody) -> None:
    ref, user = make_melody(TUNE), make_melody(TUNE, cents=[0, 0, 40, 40] + [0] * 8)

    viola, guitar = feedback(ref, user, "viola"), feedback(ref, user, "acoustic_guitar")

    assert viola[0].message == guitar[0].message  # same measurement...
    assert guitar[0].tip == SPOT_TIPS[("pitch", True)]["guitar"]  # ...different advice
    assert viola[0].tip != guitar[0].tip


def test_electric_guitar_tone_advice_is_less_certain(make_melody) -> None:
    ref = make_melody(TUNE, overtones=(1.0, 0.8, 0.8, 0.7, 0.6))
    darker = make_melody(TUNE, overtones=(1.0, 0.2, 0.05))

    (viola,) = feedback(ref, darker, "viola")
    (electric,) = feedback(ref, darker, "electric_guitar")

    assert viola.dimension == electric.dimension == "tone"
    assert "darker" in viola.message
    assert viola.confidence == config.DIMENSION_CONFIDENCE["tone"]
    assert electric.confidence == config.ELECTRIC_TONE_CONFIDENCE
    assert electric.tip.endswith(ELECTRIC_TONE_NOTE)


def test_sharp_throughout_is_one_tuning_tip(make_melody) -> None:
    (item,) = feedback(make_melody(TUNE), make_melody(TUNE, cents=30))

    # One habit instead of a "you were sharp at ..." for the whole piece.
    assert item.region is None
    assert item.message.startswith("Your whole take is about 30 cents sharp.")
    assert "problem-spot list" in item.message
    assert "A = 442 Hz" in item.tip


def test_one_bad_passage_is_not_a_tuning_problem(make_melody) -> None:
    cents = [40] * 6 + [0] * 6  # sharp for half the piece
    items = feedback(make_melody(TUNE), make_melody(TUNE, cents=cents))

    assert not any("whole take" in i.message for i in items)
    assert [i.region is not None for i in items] == [True]


# Notes 6-7 (2.4-3.2 s) differ from the rest of a take that is 20 cents sharp.
# (Kept under 70 cents: a note nearly a semitone off can be aligned to a
# neighbouring note instead; see the known limit in compare.py.)
@pytest.mark.parametrize(
    ("notes_6_7", "expected"),
    [
        (65, "65 cents sharp, even allowing for your tuning"),  # much sharper still
        (-25, "25 cents flat, even allowing for your tuning"),  # the other way
        (0, None),  # in tune: different from the rest, but not wrong
    ],
)
def test_tuning_keeps_spots_that_are_off_beyond_the_tuning(
    make_melody, notes_6_7: float, expected: str | None
) -> None:
    cents = [20] * 6 + [notes_6_7] * 2 + [20] * 4
    items = feedback(make_melody(TUNE), make_melody(TUNE, cents=cents))

    habits = [i for i in items if i.region is None]
    spots = [i for i in items if i.region is not None]
    assert len(habits) == 1
    assert habits[0].message.startswith("Your whole take is about 20 cents sharp.")
    if expected is None:
        assert spots == []
    else:
        (spot,) = spots
        assert spot.message == f"At 0:02.4 – 0:03.2 you were {expected}."


def test_octave_shift_is_a_note_not_a_mistake(make_melody) -> None:
    (item,) = feedback(make_melody(TUNE), make_melody([m - 12 for m in TUNE]))

    assert item.message == "You played 1 octave lower than the reference."
    assert item.importance < 0.1


def test_going_sharp_on_loud_notes_is_a_habit(make_melody) -> None:
    loud_soft = [0.5, 0.1] * 6
    user = make_melody(TUNE, amplitudes=loud_soft, cents=[30, 0] * 6)  # loud notes sharp
    items = feedback(make_melody(TUNE, amplitudes=loud_soft), user, "acoustic_guitar")

    habit = items[0]  # ranked above the individual sharp notes
    assert habit.region is None
    assert (
        habit.message == "On your loudest notes you're 30 cents sharper than on your softest ones."
    )
    assert "picking" in habit.tip


def test_flat_dynamics_get_exaggerate_advice(make_melody) -> None:
    ref = make_melody(TUNE, amplitudes=[0.5] * 4 + [0.05] * 4 + [0.5] * 4)  # p in the middle
    items = feedback(ref, make_melody(TUNE, amplitudes=0.25))

    habit = items[0]
    assert habit.dimension == "dynamics" and habit.region is None
    assert (
        habit.message
        == "Your loud and soft playing are only 2 dB apart; the reference spans 20 dB."
    )
    assert habit.tip.startswith("Exaggerate the contrast")


def test_wider_dynamics_than_a_reference_are_trusted_less(make_melody) -> None:
    user = make_melody(TUNE, amplitudes=[0.5] * 4 + [0.05] * 4 + [0.5] * 4)
    items = feedback(make_melody(TUNE, amplitudes=0.25), user)

    (habit,) = [i for i in items if i.region is None]
    assert habit.message.startswith("Your dynamics are more extreme than the reference")
    assert habit.confidence == config.DYNAMICS_WIDER_CONFIDENCE


def test_noisy_tone_replaces_the_brighter_spot_it_causes(make_melody) -> None:
    items = feedback(make_melody(TUNE), with_noise(make_melody(TUNE), snr_db=15))

    (item,) = items  # no separate "you were brighter" spot
    assert item.dimension == "tone" and item.region is None
    assert item.message.startswith("Your sound has more noise in it")
    assert "brighter spot" in item.message
    assert "quiet" in item.tip  # warns that the room/mic could be the cause
    assert item.confidence == config.NOISE_CONFIDENCE


@pytest.mark.parametrize(
    ("ref_snr", "user_snr"),
    [(None, 35.0), (20.0, 20.0)],  # faint hiss; equally noisy recordings
)
def test_small_or_shared_noise_is_not_flagged(make_melody, ref_snr, user_snr) -> None:
    y = make_melody(TUNE)
    ref = y if ref_snr is None else with_noise(y, ref_snr)

    items = feedback(ref, with_noise(y, user_snr))

    assert not any("noise" in i.message for i in items)


def test_electric_guitar_habits_mention_the_amp(make_melody) -> None:
    ref = make_melody(TUNE, amplitudes=[0.5] * 4 + [0.05] * 4 + [0.5] * 4)
    flat_dynamics = feedback(ref, make_melody(TUNE, amplitudes=0.25), "electric_guitar")
    noisy = feedback(make_melody(TUNE), with_noise(make_melody(TUNE), 15), "electric_guitar")

    assert "Amp compression" in flat_dynamics[0].tip
    (noise,) = noisy
    assert "Amp gain" in noise.tip
    assert noise.confidence == config.ELECTRIC_TONE_CONFIDENCE


def test_too_short_a_take_makes_no_habit_claims(make_melody) -> None:
    # Under a second: sharp, noisy, and flatter dynamics, but too little evidence.
    ref = make_melody([60, 67], amplitudes=[0.5, 0.05])
    user = with_noise(make_melody([60, 67], cents=30, amplitudes=0.25), 15)

    items = feedback(ref, user)

    assert all(i.region is not None for i in items)  # spots only


def test_advice_is_ranked(make_melody) -> None:
    cents = [0, 0, 40, 40, 0, 0, 0, -20, -20, 0, 0, 0]
    amps = [0.5] * 8 + [0.15] * 4
    items = feedback(make_melody(TUNE), make_melody(TUNE, cents=cents, amplitudes=amps))

    assert len(items) >= 3
    ranks = [i.rank for i in items]
    assert ranks == sorted(ranks, reverse=True)
