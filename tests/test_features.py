"""Known-answer tests for pitch, dynamics, and tone features.

Each test feeds in synthetic audio where physics tells us the right answer.
"""

import librosa
import numpy as np
import pytest

from musiccompare import config
from musiccompare.features.dynamics import extract_rms_db
from musiccompare.features.extract import EXTRACTOR_VERSION, extract_features
from musiccompare.features.pitch import extract_pitch, fold_octaves, hz_to_cents, refine_f0
from musiccompare.features.tone import extract_tone
from musiccompare.models import AudioClip

SR = config.SAMPLE_RATE


def clip_of(y: np.ndarray) -> AudioClip:
    return AudioClip(y=y, sr=SR, name="test")


def middle(x: np.ndarray) -> np.ndarray:
    """Drop the first/last few frames, where zero-padding at the edges skews values."""
    return x[5:-5]


# --- Pitch -----------------------------------------------------------------


def test_440hz_is_detected_as_a4(make_tone) -> None:
    f0, voiced_prob = extract_pitch(clip_of(make_tone(440.0)))

    assert librosa.hz_to_note(np.nanmedian(f0)) == "A4"
    assert np.nanmedian(f0) == pytest.approx(440.0, abs=1.0)
    assert np.median(voiced_prob) > 0.5


def test_450hz_reads_as_39_cents_sharp(make_tone) -> None:
    f0, _ = extract_pitch(clip_of(make_tone(450.0)))

    cents = hz_to_cents(np.nanmedian(f0), 440.0)

    assert cents == pytest.approx(38.9, abs=0.5)  # 1200 * log2(450/440)


def test_low_guitar_e_is_tracked(make_tone) -> None:
    # Lowest guitar string, E2 ≈ 82.4 Hz: near the bottom of our search range.
    f0, _ = extract_pitch(clip_of(make_tone(82.41)))

    assert librosa.hz_to_note(np.nanmedian(f0)) == "E2"
    # pYIN alone reads ~10 cents off here (its 10-cent grid); refinement fixes it.
    assert abs(hz_to_cents(np.nanmedian(f0), 82.41)) < 1.0


def test_pitch_refinement_is_precise_on_a_melody(make_tone) -> None:
    # Off-grid pitches (not exact multiples of 10 cents from anything) in sequence.
    true_hz = [196.0, 247.3, 293.1, 351.7, 441.9, 523.0]
    y = np.concatenate([make_tone(f, duration_s=0.5) for f in true_hz])

    f0, _ = extract_pitch(clip_of(y))

    frames_per_note = int(0.5 * SR) // config.HOP_LENGTH
    for i, f in enumerate(true_hz):
        # Use the stable middle of each note, away from the note changes.
        mid = f0[i * frames_per_note + 5 : (i + 1) * frames_per_note - 5]
        assert abs(np.nanmedian(hz_to_cents(mid, f))) < 1.0, f"note {i} ({f} Hz)"


def test_refinement_handles_weak_fundamental_and_noise(make_tone) -> None:
    # Guitar-like: the fundamental is quieter than the 2nd/3rd harmonics, plus noise.
    f = 110.7
    y = sum(make_tone(f * h, amplitude=a) for h, a in [(1, 0.1), (2, 0.4), (3, 0.3), (4, 0.2)])
    y = y + 0.01 * np.random.default_rng(seed=0).standard_normal(len(y))

    f0, _ = extract_pitch(clip_of(y.astype(np.float32)))

    assert abs(hz_to_cents(np.nanmedian(f0), f)) < 1.0


def test_refine_keeps_unvoiced_frames_nan(make_tone) -> None:
    coarse = np.array([np.nan, 440.0, np.nan])

    refined = refine_f0(make_tone(440.0, duration_s=0.1), coarse, SR)

    assert np.isnan(refined[0]) and np.isnan(refined[2])
    assert refined[1] == pytest.approx(440.0, abs=0.5)


def test_silence_is_unvoiced(make_silence) -> None:
    f0, _ = extract_pitch(clip_of(make_silence()))

    assert np.all(np.isnan(f0))


def test_hz_to_cents_known_values() -> None:
    np.testing.assert_allclose(hz_to_cents([880.0, 220.0, 440.0], 440.0), [1200, -1200, 0])
    assert np.isnan(hz_to_cents(np.nan, 440.0))


@pytest.mark.parametrize(
    ("cents", "expected"),
    [(30, 30), (1230, 30), (-1170, 30), (-30, -30), (2400, 0), (599, 599), (600, -600)],
)
def test_fold_octaves(cents: float, expected: float) -> None:
    assert fold_octaves(cents) == pytest.approx(expected)


# --- Dynamics --------------------------------------------------------------


def test_rms_db_of_sine_matches_theory(make_tone) -> None:
    db = extract_rms_db(clip_of(make_tone(440.0, amplitude=0.5)))

    expected = 20 * np.log10(0.5 / np.sqrt(2))  # ≈ -9.03 dB
    np.testing.assert_allclose(middle(db), expected, atol=0.1)


def test_half_amplitude_is_6db_quieter(make_tone) -> None:
    loud = extract_rms_db(clip_of(make_tone(amplitude=0.5)))
    soft = extract_rms_db(clip_of(make_tone(amplitude=0.25)))

    assert np.median(loud - soft) == pytest.approx(6.02, abs=0.05)


def test_silence_hits_the_floor(make_silence) -> None:
    db = extract_rms_db(clip_of(make_silence()))

    np.testing.assert_array_equal(db, config.SILENCE_DB)


def test_crescendo_gets_louder(make_tone) -> None:
    y = make_tone(duration_s=2.0, amplitude=1.0) * np.linspace(0.05, 1.0, 2 * SR)

    db = middle(extract_rms_db(clip_of(y.astype(np.float32))))

    assert np.all(np.diff(db) > 0)  # strictly increasing loudness


# --- Tone ------------------------------------------------------------------


def test_centroid_of_pure_tone_is_its_frequency(make_tone) -> None:
    centroid, _, _ = extract_tone(clip_of(make_tone(1000.0)))

    assert np.median(centroid) == pytest.approx(1000.0, rel=0.05)


def test_overtones_make_tone_brighter(make_tone) -> None:
    pure = make_tone(220.0, amplitude=0.5)
    rich = pure + make_tone(660.0, amplitude=0.3) + make_tone(1100.0, amplitude=0.2)

    pure_centroid, _, _ = extract_tone(clip_of(pure))
    rich_centroid, _, _ = extract_tone(clip_of(rich.astype(np.float32)))

    assert np.median(rich_centroid) > 1.5 * np.median(pure_centroid)


def test_noise_is_flatter_than_a_tone(make_tone) -> None:
    rng = np.random.default_rng(seed=0)  # seeded -> same "random" noise every run
    noise = (0.3 * rng.standard_normal(SR)).astype(np.float32)

    _, tone_flatness, _ = extract_tone(clip_of(make_tone()))
    _, noise_flatness, _ = extract_tone(clip_of(noise))

    assert np.median(tone_flatness) < 0.01
    assert np.median(noise_flatness) > 0.3


# --- Everything together ---------------------------------------------------


def test_extract_features_frames_line_up(make_tone) -> None:
    clip = clip_of(make_tone(duration_s=2.0))

    fs = extract_features(clip)

    expected_frames = 1 + len(clip.y) // config.HOP_LENGTH
    assert fs.n_frames == expected_frames
    assert fs.mfcc.shape == (config.N_MFCC, expected_frames)
    np.testing.assert_allclose(np.diff(fs.times), config.HOP_LENGTH / SR)
    assert fs.extractor_version == EXTRACTOR_VERSION
