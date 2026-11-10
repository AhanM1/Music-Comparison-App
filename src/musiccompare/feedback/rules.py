"""Rule-based feedback (v1): hand-written rules that turn a comparison into advice.

Two kinds of advice:

- **Habits**: patterns across the whole take ("sharp throughout", "your
  dynamics are too flat"). Often the most useful advice: one fix improves
  everything.
- **Spots**: one item per problem region ("at 0:03.2 you were 30 cents flat"),
  with a tip for that dimension and direction.

The same symptom has different causes on different instruments (sharp on a
viola: the left hand lands high; sharp on a guitar: pressing too hard), so tips
are chosen by instrument family.

Ranking. Every item gets
- importance = dimension weight x severity x reach, where
  reach = 0.5 + 0.5 x (share of the piece affected), so a habit across the
  whole piece counts twice as much as an equally bad short spot;
- confidence = how far we trust the measurement.
Items are sorted by importance x confidence (FeedbackItem.rank).
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import numpy as np

from musiccompare import config
from musiccompare.analysis.quality import busy_reference
from musiccompare.analysis.segments import (
    FRAME_S,
    SMOOTHING_WIDTH,
    find_regions,
    find_runs,
    median_smooth,
    severity,
)
from musiccompare.feedback.wording import describe, span_list, time_span
from musiccompare.models import (
    ComparisonResult,
    Dimension,
    FeatureSet,
    FeedbackItem,
    FloatArray,
    Instrument,
    IntArray,
    PlayerContext,
    ProblemRegion,
)

# v2: missing / skipped / extra material, wrong notes, busy references.
SOURCE = "rules-v2"

Family = Literal["bowed", "guitar", "other"]


def family(instrument: Instrument) -> Family:
    if instrument == "viola":
        return "bowed"
    if instrument in ("acoustic_guitar", "electric_guitar"):
        return "guitar"
    return "other"


def importance(dimension: Dimension, severity: float, reach: float) -> float:
    """0..1; ``reach`` is the share of the piece affected (0..1)."""
    return config.DIMENSION_WEIGHTS[dimension] * severity * (0.5 + 0.5 * min(reach, 1.0))


def confidence(dimension: Dimension, context: PlayerContext) -> float:
    if dimension == "tone" and context.instrument == "electric_guitar":
        return config.ELECTRIC_TONE_CONFIDENCE
    return config.DIMENSION_CONFIDENCE[dimension]


# --- Spots -------------------------------------------------------------------

# Tip per (dimension, above the reference?) and instrument family.
# "Above" = sharp / louder / brighter.
SPOT_TIPS: dict[tuple[Dimension, bool], dict[Family, str]] = {
    ("pitch", True): {
        "bowed": "Your fingers are landing a little high (toward the bridge). Play the "
        "passage slowly against a drone or tuner and listen for the beats to disappear.",
        "guitar": "Pressing too hard, or pushing the string sideways, bends notes sharp. "
        "Fret with just enough pressure, right behind the fret.",
        "other": "Play the passage slowly against a drone or tuner and listen for the "
        "beats to disappear.",
    },
    ("pitch", False): {
        "bowed": "Your fingers are landing a little low (toward the scroll). Play the "
        "passage slowly against a drone, and check your hand frame doesn't shrink on "
        "shifts and extensions.",
        "guitar": "Flat notes on guitar usually mean a string has drifted out of tune or "
        "a bend stops short. Re-check your tuning, and make sure bends reach the note.",
        "other": "Play the passage slowly against a drone or tuner and listen for the "
        "beats to disappear.",
    },
    ("dynamics", True): {
        "bowed": "Use a lighter, slower bow, with the contact point nearer the fingerboard.",
        "guitar": "Pick or pluck more lightly here.",
        "other": "Hold back a little here.",
    },
    ("dynamics", False): {
        "bowed": "Use more bow (faster) and more arm weight, with the contact point "
        "nearer the bridge.",
        "guitar": "Pick or pluck more firmly here.",
        "other": "Play out more here.",
    },
    ("tone", True): {
        "bowed": "Move the bow slightly toward the fingerboard and ease off the pressure "
        "for a rounder sound.",
        "guitar": "Pick or pluck a little closer to the neck for a warmer sound.",
        "other": "Aim for a rounder, warmer sound here.",
    },
    ("tone", False): {
        "bowed": "Move the bow closer to the bridge, with a bit more weight, for a "
        "clearer, more ringing sound.",
        "guitar": "Pick or pluck a little closer to the bridge for a brighter sound.",
        "other": "Aim for a clearer, more ringing sound here.",
    },
}

# A different note entirely, not intonation (|error| >= config.WRONG_NOTE_CENTS).
WRONG_NOTE_TIPS: dict[Family, str] = {
    "bowed": "Check the note and fingering against the score. If it's a shift, practise "
    "it slowly with a guide finger, and hear the target note before you land on it.",
    "guitar": "Check the fret and string against the score or tab, then practise the "
    "change slowly until your hand lands there without looking.",
    "other": "Check the note against the score, then practise the passage slowly.",
}


def is_wrong_note(region: ProblemRegion) -> bool:
    return region.dimension == "pitch" and abs(region.mean_error) >= config.WRONG_NOTE_CENTS


ELECTRIC_TONE_NOTE = (
    " On electric guitar, the pickup, tone knob, and amp change the sound far more "
    "than your hands do, so match those to the reference first."
)


def spot_item(
    region: ProblemRegion, context: PlayerContext, piece_s: float, aside: str = ""
) -> FeedbackItem:
    """Advice for one problem region; ``aside`` is added to the message, e.g. ", even ..."."""
    where = time_span(region.start_s, region.end_s)
    if is_wrong_note(region):
        # "200 cents sharp" + "your fingers land a little high" would be wrong
        # advice: that's a different note, not an out-of-tune one.
        message = (
            f"At {where} you played {describe(region)} the reference: probably a wrong "
            "note or a missed shift."
        )
        tip = WRONG_NOTE_TIPS[family(context.instrument)]
    else:
        message = f"At {where} you were {describe(region)}{aside}."
        tip = SPOT_TIPS[(region.dimension, region.mean_error > 0)][family(context.instrument)]
    if region.dimension == "tone" and context.instrument == "electric_guitar":
        tip += ELECTRIC_TONE_NOTE
    return FeedbackItem(
        dimension=region.dimension,
        message=message,
        tip=tip,
        importance=importance(region.dimension, region.severity, region.duration_s / piece_s),
        confidence=confidence(region.dimension, context),
        source=SOURCE,
        region=region,
    )


# --- Habits ------------------------------------------------------------------
# Each habit rule looks at the whole take and returns one item, or None.

HabitRule = Callable[[FeatureSet, FeatureSet, ComparisonResult, PlayerContext], FeedbackItem | None]

# Informational notes ("you played an octave down") rank below any real advice.
NOTE_IMPORTANCE = 0.05


def explains_spots(count: int, kind: str) -> str:
    """Sentence saying how many listed problem spots a habit accounts for ("" for none).

    Those spots get no advice of their own: the habit's tip is the real fix.
    """
    if count == 0:
        return ""
    if count == 1:
        return f" This explains the {kind} spot in the problem-spot list."
    return f" This explains {count} {kind} spots in the problem-spot list."


def tuning_offset(result: ComparisonResult) -> tuple[float, float] | None:
    """(cents, share of notes) if the whole take is consistently sharp (+) or flat (-).

    None when the take isn't off overall, or only one passage is (that's a
    spot, not a tuning problem).
    """
    cents = result.differences.pitch_cents
    cents = cents[np.isfinite(cents)]
    if cents.size < config.MIN_HABIT_FRAMES:
        return None
    offset = float(np.median(cents))
    if abs(offset) <= config.PITCH_TOLERANCE_CENTS:
        return None
    same_side = float(np.mean(np.sign(offset) * cents > config.PITCH_TOLERANCE_CENTS / 2))
    return (offset, same_side) if same_side >= config.TUNING_HABIT_FRACTION else None


def spots_beyond_tuning(
    user: FeatureSet, result: ComparisonResult, offset: float
) -> list[ProblemRegion]:
    """Pitch spots that are still off once the overall tuning offset is allowed for.

    With the whole take 20 cents sharp, the comparison finds one long "sharp"
    region that would hide two notes played 60 cents sharp. Searching again
    with the offset removed finds those two notes on their own.

    mean_error is put back on the usual scale ("60 cents sharp"); severity
    measures how far past the tuning offset the spot is.
    """
    smoothed = median_smooth(result.differences.pitch_cents - offset, SMOOTHING_WIDTH)
    user_times = user.times[result.alignment.user_frames]
    found = find_regions(
        "pitch", result.differences.times, smoothed, config.PITCH_TOLERANCE_CENTS, user_times
    )
    spots = [replace(r, mean_error=r.mean_error + offset) for r in found]
    # e.g. sharp throughout but one note in tune: off from your tuning, but not wrong.
    return [r for r in spots if abs(r.mean_error) > config.PITCH_TOLERANCE_CENTS]


def tuning_habit(
    ref: FeatureSet, user: FeatureSet, result: ComparisonResult, context: PlayerContext
) -> FeedbackItem | None:
    found = tuning_offset(result)
    if found is None:
        return None
    offset, share = found
    word, lean = ("sharp", "high") if offset > 0 else ("flat", "low")
    same_way = [r for r in result.regions if r.dimension == "pitch" and r.mean_error * offset > 0]
    message = f"Your whole take is about {abs(offset):.0f} cents {word}."
    message += explains_spots(len(same_way), word)
    tips: dict[Family, str] = {
        "bowed": "Tune your open strings carefully before you play and recheck them in "
        f"rests. If they were in tune, your left hand is sitting {lean}: practise scales "
        "against a drone.",
        "guitar": "Retune before you play, and recheck between takes.",
        "other": "Tune carefully before you play.",
    }
    reference_a = reference_tuning_hz(ref)
    if reference_a is None:
        standard = (
            " If you were already in tune, the recording may use a different tuning "
            "standard (orchestras often tune to A = 442 Hz, Baroque groups to 415 Hz): "
            "tune to the recording itself instead."
        )
    else:
        standard = (
            f" Note the reference itself is tuned to about A = {reference_a:.0f} Hz, not "
            "440: tune to the recording (or set your tuner to that A) before you play."
        )
    tip = tips[family(context.instrument)] + standard
    return FeedbackItem(
        dimension="pitch",
        message=message,
        tip=tip,
        importance=importance("pitch", severity(offset, config.PITCH_TOLERANCE_CENTS), share),
        confidence=confidence("pitch", context),
        source=SOURCE,
    )


def reference_tuning_hz(ref: FeatureSet) -> float | None:
    """The reference's tuning as "A = ... Hz", or None if it's close to A = 440.

    Each pitched frame's distance from the nearest equal-tempered note (on the
    A = 440 grid) is measured; a consistent offset means the whole recording
    was tuned higher or lower. Vibrato and expressive pitch average out in the
    median.
    """
    f0 = ref.f0_hz[np.isfinite(ref.f0_hz)]
    if f0.size < config.MIN_HABIT_FRAMES:
        return None
    semitones = 12 * np.log2(f0 / 440.0)
    offset_cents = float(np.median(100 * (semitones - np.round(semitones))))
    if abs(offset_cents) < config.REFERENCE_TUNING_CENTS:
        return None
    return float(440.0 * 2 ** (offset_cents / 1200))


def octave_habit(
    ref: FeatureSet, user: FeatureSet, result: ComparisonResult, context: PlayerContext
) -> FeedbackItem | None:
    """Not a mistake, just worth knowing: we compared as if in the same octave."""
    shift = result.differences.octave_shift
    if shift == 0:
        return None
    s = "s" if abs(shift) > 1 else ""
    return FeedbackItem(
        dimension="pitch",
        message=f"You played {abs(shift)} octave{s} {'higher' if shift > 0 else 'lower'} "
        "than the reference.",
        tip="Fine if that's on purpose (e.g. a viola line on guitar): everything was "
        "compared as if you were in the same octave.",
        importance=NOTE_IMPORTANCE,
        confidence=confidence("pitch", context),
        source=SOURCE,
    )


def pitch_vs_loudness_habit(
    ref: FeatureSet, user: FeatureSet, result: ComparisonResult, context: PlayerContext
) -> FeedbackItem | None:
    """Does your pitch drift when you play loudly? Compares your loudest and softest thirds."""
    cents = result.differences.pitch_cents
    loudness = user.rms_db[result.alignment.user_frames]
    pitched = np.isfinite(cents)
    if pitched.sum() < config.MIN_HABIT_FRAMES:
        return None
    soft_cut, loud_cut = np.percentile(loudness[pitched], [100 / 3, 200 / 3])
    loud = pitched & (loudness >= loud_cut)
    soft = pitched & (loudness <= soft_cut)
    spread = float(np.median(loudness[loud]) - np.median(loudness[soft]))
    if spread < config.LOUD_SOFT_MIN_SPREAD_DB:
        return None  # not enough difference in loudness to tell anything
    gap = float(np.median(cents[loud]) - np.median(cents[soft]))
    if abs(gap) <= config.PITCH_TOLERANCE_CENTS:
        return None
    sharper = gap > 0
    tips: dict[tuple[bool, Family], str] = {
        (True, "bowed"): "Playing loudly seems to tense your left hand and push your "
        "fingers sharp. Get volume from bow speed and contact point, and keep the left "
        "hand relaxed.",
        (True, "guitar"): "Hard picking makes each note start sharp (the string stretches "
        "as it swings wide), and squeezing the neck bends notes sharp. Pick a little "
        "lighter and fret with just enough pressure.",
        (False, "bowed"): "Heavy bow pressure pulls the pitch down slightly. For loud "
        "passages use more bow speed and a contact point nearer the bridge, not more "
        "pressure.",
        (False, "guitar"): "Check you don't ease off the fretting hand when you pick hard, "
        "and that bends reach full pitch.",
    }
    generic = (
        "Practise the loud passages slowly with a tuner, keeping the pitch steady as you "
        "get louder."
    )
    return FeedbackItem(
        dimension="pitch",
        message=f"On your loudest notes you're {abs(gap):.0f} cents "
        f"{'sharper' if sharper else 'flatter'} than on your softest ones.",
        tip=tips.get((sharper, family(context.instrument)), generic),
        importance=importance("pitch", severity(gap, config.PITCH_TOLERANCE_CENTS), 1 / 3),
        confidence=confidence("pitch", context),
        source=SOURCE,
    )


def dynamic_range(features: FeatureSet) -> float | None:
    """dB from soft to loud (percentile 10 to 90) while playing, or None if too little audio."""
    playing = features.rms_db[features.rms_db > config.SOUNDING_DB]
    if playing.size < config.MIN_HABIT_FRAMES:
        return None
    low, high = np.percentile(playing, config.DYNAMIC_RANGE_PERCENTILES)
    return float(high - low)


def dynamic_range_habit(
    ref: FeatureSet, user: FeatureSet, result: ComparisonResult, context: PlayerContext
) -> FeedbackItem | None:
    ref_range, user_range = dynamic_range(ref), dynamic_range(user)
    if ref_range is None or user_range is None:
        return None
    gap = user_range - ref_range
    if abs(gap) <= config.DYNAMICS_TOLERANCE_DB:
        return None
    if gap < 0:
        message = (
            f"Your loud and soft playing are only {user_range:.0f} dB apart; "
            f"the reference spans {ref_range:.0f} dB."
        )
        tips: dict[Family, str] = {
            "bowed": "Exaggerate the contrast: a slow, light bow near the fingerboard for "
            "soft passages, a fast, heavy bow near the bridge for loud ones.",
            "guitar": "Exaggerate the contrast: pick or pluck very lightly in quiet "
            "passages and dig in for loud ones.",
            "other": "Exaggerate the contrast between your softest and loudest playing.",
        }
        tip = tips[family(context.instrument)]
        if context.instrument == "electric_guitar":
            tip += (
                " Amp compression and distortion squash dynamics; a cleaner setting lets "
                "the contrast through."
            )
        trust = confidence("dynamics", context)
    else:
        message = (
            f"Your dynamics are more extreme than the reference: {user_range:.0f} dB "
            f"from soft to loud, against {ref_range:.0f} dB."
        )
        tip = (
            "Ease off your loudest notes and don't drop quite as soft. Commercial "
            "recordings are often compressed, which narrows their range, so weigh this "
            "one lightly."
        )
        trust = config.DYNAMICS_WIDER_CONFIDENCE
    return FeedbackItem(
        dimension="dynamics",
        message=message,
        tip=tip,
        importance=importance("dynamics", severity(gap, config.DYNAMICS_TOLERANCE_DB), 1.0),
        confidence=trust,
        source=SOURCE,
    )


def flatness_db(flatness: FloatArray) -> FloatArray:
    """Spectral flatness in dB, floored at config.FLATNESS_FLOOR_DB."""
    return 10 * np.log10(np.maximum(flatness, 10 ** (config.FLATNESS_FLOOR_DB / 10)))


def noise_gap(
    ref: FeatureSet, user: FeatureSet, result: ComparisonResult
) -> tuple[float, float] | None:
    """(dB, share of the take) if your sound is noisier than the reference's, else None.

    Only "noisier" counts: a noisy reference isn't something you can fix.
    """
    u = result.alignment.user_frames
    both = (ref.rms_db > config.SOUNDING_DB) & (user.rms_db[u] > config.SOUNDING_DB)
    if both.sum() < config.MIN_HABIT_FRAMES:
        return None
    noisier = flatness_db(user.flatness[u][both]) - flatness_db(ref.flatness[both])
    gap = float(np.median(noisier))
    if gap <= config.NOISE_TOLERANCE_DB:
        return None
    return gap, float(np.mean(noisier > config.NOISE_TOLERANCE_DB))


def explained_by_noise(region: ProblemRegion) -> bool:
    """Noise sits mostly above the notes' harmonics, so it also measures as brighter."""
    return region.dimension == "tone" and region.mean_error > 0


def noisy_tone_habit(
    ref: FeatureSet, user: FeatureSet, result: ComparisonResult, context: PlayerContext
) -> FeedbackItem | None:
    """More noise (scratch, hiss, buzz) in your sound than in the reference's."""
    found = noise_gap(ref, user, result)
    if found is None:
        return None
    gap, share = found
    tips: dict[Family, str] = {
        "bowed": "A scratchy sound usually means too much bow pressure for the bow speed, "
        "or a bow drifting off its lane. Keep the bow parallel to the bridge and balance "
        "weight with speed.",
        "guitar": "For buzzing or string noise, press right behind the fret and mute the "
        "strings you're not playing.",
        "other": "Aim for a cleaner, more focused sound.",
    }
    tip = tips[family(context.instrument)]
    if context.instrument == "electric_guitar":
        tip += " Amp gain and distortion add noise too."
    tip += (
        " A noisy room or microphone looks the same to the app, so record somewhere "
        "quiet before trusting this one."
    )
    trust = (
        config.ELECTRIC_TONE_CONFIDENCE
        if context.instrument == "electric_guitar"
        else config.NOISE_CONFIDENCE
    )
    return FeedbackItem(
        dimension="tone",
        message="Your sound has more noise in it than the reference's (scratch, hiss, or buzz)."
        + explains_spots(sum(map(explained_by_noise, result.regions)), "brighter"),
        tip=tip,
        # Half the tolerance as the scale: flatness gaps are in dB, so a gap just
        # past the 10 dB tolerance (already 10x noisier) still matters.
        importance=importance("tone", severity(gap, config.NOISE_TOLERANCE_DB / 2), share),
        confidence=trust,
        source=SOURCE,
    )


# --- Coverage: notes missed, skipped, or added ---------------------------------
# These come first in importance: no tip about intonation matters if half the
# passage wasn't played, or a count-in threw the comparison off.

COVERAGE_SEVERITY = 0.8  # as bad as a clearly wrong passage
EXTRA_IMPORTANCE = 0.2  # worth knowing and easy to fix, but not a playing mistake
UNHEARD_CONFIDENCE = 0.7  # could also be a note too quiet or noisy to track


def spans_s(spans: IntArray) -> list[tuple[float, float]]:
    """(start, stop) frame spans -> (start, end) seconds."""
    return [(float(a * FRAME_S), float(b * FRAME_S)) for a, b in spans]


def skipped_habit(
    ref: FeatureSet, user: FeatureSet, result: ComparisonResult, context: PlayerContext
) -> FeedbackItem | None:
    """Stretches of the reference with nothing like them in the take."""
    skipped = result.alignment.skipped_ref
    if len(skipped) == 0:
        return None
    share = float((skipped[:, 1] - skipped[:, 0]).sum() / ref.n_frames)
    # Neutral wording on purpose: a wrong note that sounds like the *next*
    # reference note makes alignment jump ahead, which looks just like a skip.
    return FeedbackItem(
        dimension="pitch",
        message=f"At {span_list(spans_s(skipped))} of the reference, your take doesn't "
        "match: it sounds like you skipped these notes or played different ones.",
        tip="Check this passage against the score, practise it slowly, then play the "
        "section straight through. These moments count as missed in the scores.",
        importance=importance("pitch", COVERAGE_SEVERITY, share),
        confidence=confidence("pitch", context),
        source=SOURCE,
    )


def unheard_habit(
    ref: FeatureSet, user: FeatureSet, result: ComparisonResult, context: PlayerContext
) -> FeedbackItem | None:
    """The reference plays notes, but no clear note was heard from the player."""
    unheard = result.differences.missing.copy()
    for start, stop in result.alignment.skipped_ref:
        unheard[start:stop] = False  # reported by skipped_habit
    runs = np.array(find_runs(unheard), dtype=np.intp).reshape(-1, 2)
    if len(runs) == 0:
        return None
    return FeedbackItem(
        dimension="pitch",
        message=f"At {span_list(spans_s(runs))} the reference plays notes, but we couldn't "
        "hear a clear note from you.",
        tip="If you stopped or skipped here, play the passage through. If you did play "
        "it, the notes may have been too quiet, muted, or too noisy for the app to "
        "follow. Either way these moments count as missed in the scores.",
        importance=importance("pitch", COVERAGE_SEVERITY, float(unheard.mean())),
        confidence=UNHEARD_CONFIDENCE,
        source=SOURCE,
    )


def extra_material_habit(
    ref: FeatureSet, user: FeatureSet, result: ComparisonResult, context: PlayerContext
) -> FeedbackItem | None:
    """Parts of the take with nothing like them in the reference: never scored."""
    extra = result.alignment.extra_user
    if len(extra) == 0:
        return None
    seconds = float((extra[:, 1] - extra[:, 0]).sum() * FRAME_S)
    return FeedbackItem(
        dimension="pitch",
        message=f"Your take has about {seconds:.1f} s that isn't in the reference (at "
        f"{span_list(spans_s(extra))} of your recording): a count-in, a repeat, or "
        "extra notes. That part isn't scored.",
        tip="Trim your recording to the same passage as the reference, so every note "
        "you play gets compared.",
        importance=EXTRA_IMPORTANCE,
        confidence=confidence("pitch", context),
        source=SOURCE,
    )


HABIT_RULES: tuple[HabitRule, ...] = (
    skipped_habit,
    unheard_habit,
    extra_material_habit,
    tuning_habit,
    octave_habit,
    pitch_vs_loudness_habit,
    dynamic_range_habit,
    noisy_tone_habit,
)


class RuleBasedFeedback:
    """The v1 FeedbackProvider: every habit rule, then one item per problem spot."""

    source = SOURCE

    def generate(
        self,
        ref: FeatureSet,
        user: FeatureSet,
        result: ComparisonResult,
        context: PlayerContext,
    ) -> list[FeedbackItem]:
        items = [
            item for rule in HABIT_RULES if (item := rule(ref, user, result, context)) is not None
        ]
        # Spots a habit explains get no advice of their own (the habit says so).
        regions = result.regions
        aside = ""
        tuning = tuning_offset(result)
        if tuning is not None:
            pitch_spots = spots_beyond_tuning(user, result, tuning[0])
            regions = [r for r in regions if r.dimension != "pitch"] + pitch_spots
            aside = ", even allowing for your tuning"
        if noise_gap(ref, user, result) is not None:
            regions = [r for r in regions if not explained_by_noise(r)]

        # Spots barely past the tolerance stay in the problem-spot list, but
        # don't get tips: they would bury the advice that matters.
        regions = [r for r in regions if r.severity >= config.MIN_TIP_SEVERITY]

        piece_s = float(result.differences.times[-1]) + FRAME_S
        for region in regions:
            note = aside if region.dimension == "pitch" else ""
            items.append(spot_item(region, context, piece_s, note))
        if busy_reference(ref, user):
            # Accompaniment or reverb leaks into every measurement of the reference.
            items = [
                replace(item, confidence=item.confidence * config.BUSY_CONFIDENCE_FACTOR)
                for item in items
            ]
        return sorted(items, key=lambda item: item.rank, reverse=True)
