"""Tests for musiccompare.viz: figures build without errors and show the right data."""

import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from musiccompare import config
from musiccompare.analysis.alignment import align
from musiccompare.analysis.compare import compare
from musiccompare.features.extract import extract_features
from musiccompare.models import AudioClip, ComparisonResult, FeatureSet
from musiccompare.viz.alignment_plot import plot_alignment_path
from musiccompare.viz.comparison_plot import (
    plot_brightness_comparison,
    plot_loudness_comparison,
    plot_pitch_comparison,
    plot_problem_timeline,
)
from musiccompare.viz.dynamics_plot import plot_loudness_contour
from musiccompare.viz.history_plot import plot_score_history
from musiccompare.viz.overview import plot_waveform, waveform_envelope
from musiccompare.viz.pitch_plot import plot_pitch_contour
from musiccompare.viz.spectrogram_plot import MAX_COLUMNS, mel_db, plot_spectrograms
from musiccompare.viz.style import ABOVE_COLOR, BELOW_COLOR, to_png

SR = config.SAMPLE_RATE


def test_envelope_keeps_peaks() -> None:
    y = np.zeros(100_000, dtype=np.float32)
    y[54_321] = 0.8  # one isolated spike: naive "every Nth sample" would miss it

    times, lower, upper = waveform_envelope(y, SR, n_points=1_000)

    assert len(times) == len(lower) == len(upper) == 1_000
    assert upper.max() == np.float32(0.8)
    assert np.all(lower <= upper)


def test_envelope_short_signal_is_unchanged() -> None:
    y = np.linspace(-1, 1, 50, dtype=np.float32)

    _, lower, upper = waveform_envelope(y, SR, n_points=1_000)

    np.testing.assert_array_equal(lower, y)
    np.testing.assert_array_equal(upper, y)


def test_plot_waveform_returns_labelled_figure(make_tone) -> None:
    clip = AudioClip(y=make_tone(duration_s=2.0), sr=SR, name="my take")

    fig = plot_waveform(clip)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_title() == "my take"
    assert ax.get_xlim() == (0, 2.0)


def test_pitch_contour_uses_note_names(make_tone) -> None:
    clip = AudioClip(y=make_tone(440.0), sr=SR, name="a4")
    features = extract_features(clip)

    fig = plot_pitch_contour(features, title="pitch")

    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert "A4" in labels


def test_loudness_contour_plots_db(make_tone) -> None:
    clip = AudioClip(y=make_tone(amplitude=0.5), sr=SR, name="tone")
    features = extract_features(clip)

    fig = plot_loudness_contour(features, title="loud")

    ax = fig.axes[0]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), features.rms_db)
    assert ax.get_ylim()[1] == 0  # top of the axis is 0 dBFS


def test_alignment_path_plot_spans_both_recordings(make_melody) -> None:
    ref = extract_features(AudioClip(y=make_melody([60, 64, 67]), sr=SR, name="ref"))
    user = extract_features(AudioClip(y=make_melody([60, 64, 67], durations=0.6), sr=SR, name="me"))
    alignment = align(ref, user)

    fig = plot_alignment_path(ref, user, alignment)

    ax = fig.axes[0]
    path_x, path_y = ax.lines[0].get_data()
    assert path_x[-1] == ref.times[-1]  # the path runs corner to corner...
    assert path_y[-1] == user.times[-1]
    assert ax.images[0].get_extent()[1:4:2] == [ref.times[-1], user.times[-1]]  # ...of the grid


# --- Comparison plots -------------------------------------------------------

TUNE = [60, 64, 67, 72, 71, 67, 65, 69]  # 8 notes x 0.4 s = 3.2 s


def compared(
    ref_y: np.ndarray, user_y: np.ndarray
) -> tuple[FeatureSet, FeatureSet, ComparisonResult]:
    ref = extract_features(AudioClip(y=ref_y, sr=SR, name="ref"))
    user = extract_features(AudioClip(y=user_y, sr=SR, name="me"))
    return ref, user, compare(ref, user, align(ref, user))


def test_pitch_comparison_puts_user_on_reference_timeline(make_melody) -> None:
    ref, user, result = compared(make_melody(TUNE), make_melody(TUNE, durations=0.6))

    fig = plot_pitch_comparison(ref, user, result)

    ref_line, user_line = fig.axes[0].lines
    # The user's (longer) take is squeezed onto the reference's time axis.
    np.testing.assert_array_equal(user_line.get_xdata(), ref.times)
    # Same notes in both, so once aligned the two lines mostly agree.
    both = np.isfinite(ref_line.get_ydata()) & np.isfinite(user_line.get_ydata())
    agree = np.abs(ref_line.get_ydata()[both] - user_line.get_ydata()[both]) < 0.5
    assert agree.mean() > 0.9


def test_pitch_comparison_shades_sharp_and_flat_spots(make_melody) -> None:
    cents = [0, 0, 40, 40, 0, -40, -40, 0]  # notes 3-4 sharp, notes 6-7 flat
    ref, user, result = compared(make_melody(TUNE), make_melody(TUNE, cents=cents))

    ax = plot_pitch_comparison(ref, user, result).axes[0]

    sharp, flat = ax.patches  # one shaded band per problem region, in time order
    assert sharp.get_facecolor()[:3] == to_rgb(ABOVE_COLOR)
    assert flat.get_facecolor()[:3] == to_rgb(BELOW_COLOR)
    assert sharp.get_x() == pytest.approx(2 * 0.4, abs=0.1)
    assert flat.get_x() == pytest.approx(5 * 0.4, abs=0.1)
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["Sharp", "Flat", "Reference", "You"]


@pytest.mark.parametrize(
    "plot", [plot_pitch_comparison, plot_loudness_comparison, plot_brightness_comparison]
)
def test_comparison_legends_never_cover_the_curves(plot, make_melody) -> None:
    ref, user, result = compared(make_melody(TUNE), make_melody(TUNE, cents=40))
    fig = plot(ref, user, result)
    ax = fig.axes[0]

    fig.draw_without_rendering()  # run the layout so positions are final
    # The legend sits entirely to the right of the plotting area.
    assert ax.get_legend().get_window_extent().x0 >= ax.get_window_extent().x1


def test_pitch_comparison_undoes_an_octave_shift(make_melody) -> None:
    ref, user, result = compared(make_melody(TUNE), make_melody([m + 12 for m in TUNE]))

    ax = plot_pitch_comparison(ref, user, result).axes[0]

    ref_line, user_line = ax.lines
    both = np.isfinite(ref_line.get_ydata()) & np.isfinite(user_line.get_ydata())
    diff = np.abs(ref_line.get_ydata()[both] - user_line.get_ydata()[both])
    assert np.median(diff) < 0.1  # drawn on top of each other, not an octave apart
    assert user_line.get_label() == "You, 1 octave down"


def test_loudness_comparison_is_level_matched(make_melody) -> None:
    ref, user, result = compared(make_melody(TUNE), make_melody(TUNE, amplitudes=0.5 * 10**-0.5))

    ax = plot_loudness_comparison(ref, user, result).axes[0]

    ref_line, user_line = ax.lines
    sounding = ref.rms_db > config.SOUNDING_DB
    gap = user_line.get_ydata()[sounding] - ref_line.get_ydata()[sounding]
    assert np.median(np.abs(gap)) < 0.5  # 10 dB quieter take drawn on top of the reference
    assert "10.0 dB quieter overall" in ax.texts[0].get_text()


def test_brightness_comparison_skips_silence(make_melody) -> None:
    y = np.concatenate(
        [make_melody(TUNE[:4]), np.zeros(SR // 2, np.float32), make_melody(TUNE[4:])]
    )
    ref, user, result = compared(y, y)

    ax = plot_brightness_comparison(ref, user, result).axes[0]

    assert ax.get_yscale() == "log"
    rest = (ref.times > 1.75) & (ref.times < 1.95)  # inside the half-second gap
    assert np.all(np.isnan(ax.lines[0].get_ydata()[rest]))


def test_problem_timeline_has_one_bar_per_region(make_melody) -> None:
    cents = [0, 0, 40, 40, 0, 0, 0, 0]
    amps = [0.5] * 5 + [0.1] * 3
    ref, user, result = compared(make_melody(TUNE), make_melody(TUNE, cents=cents, amplitudes=amps))

    ax = plot_problem_timeline(result, duration_s=3.2).axes[0]

    assert len(ax.collections) == len(result.regions) >= 2
    assert [t.get_text() for t in ax.get_yticklabels()] == ["Tone", "Dynamics", "Pitch"]
    assert ax.get_xlim() == (0, 3.2)


# --- Spectrograms -----------------------------------------------------------


def test_spectrograms_share_one_colour_scale(make_melody) -> None:
    ref = AudioClip(y=make_melody(TUNE), sr=SR, name="ref")
    user = AudioClip(y=make_melody(TUNE, amplitudes=0.1), sr=SR, name="me")

    fig = plot_spectrograms(ref, user)

    top, bottom, _colorbar = fig.axes
    assert top.get_title() == "Reference"
    assert top.collections[0].get_clim() == bottom.collections[0].get_clim()


def test_long_clips_are_drawn_with_limited_columns(make_tone) -> None:
    long_clip = AudioClip(y=make_tone(duration_s=120.0), sr=SR, name="long")

    s_db, hop = mel_db(long_clip)

    assert s_db.shape[1] <= MAX_COLUMNS + 1
    assert hop % config.HOP_LENGTH == 0


def test_to_png_renders_a_png(make_tone) -> None:
    features = extract_features(AudioClip(y=make_tone(), sr=SR, name="tone"))

    png = to_png(plot_loudness_contour(features, title="x"), dpi=50)

    assert png.startswith(b"\x89PNG")


# --- Score history (Phase 7) -------------------------------------------------


def test_score_history_draws_one_line_per_dimension() -> None:
    scores = [
        {"pitch": 60.0, "dynamics": 70.0, "tone": None},
        {"pitch": 75.0, "dynamics": 65.0, "tone": 90.0},
    ]
    fig = plot_score_history(scores, ["Nov 01", "Nov 02"])
    [ax] = fig.axes

    lines = {line.get_label(): line for line in ax.get_lines()}
    assert set(lines) == {"Pitch", "Dynamics", "Tone"}
    np.testing.assert_array_equal(lines["Pitch"].get_ydata(), [60, 75])
    assert np.isnan(lines["Tone"].get_ydata()[0])  # a gap, not a 0
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Nov 01", "Nov 02"]
    assert ax.get_ylim() == (0, 105)


def test_score_history_with_many_sessions_counts_them() -> None:
    scores = [{"pitch": 50.0 + i, "dynamics": 70.0, "tone": 80.0} for i in range(30)]
    fig = plot_score_history(scores, [f"day {i}" for i in range(30)])  # too many to label
    assert fig.axes[0].get_xlabel() == "Session"


def test_score_history_checks_labels() -> None:
    with pytest.raises(ValueError, match="2 labels for 1 sessions"):
        plot_score_history([{"pitch": 50.0, "dynamics": 70.0, "tone": 80.0}], ["a", "b"])


def test_score_history_with_no_sessions_is_empty() -> None:
    assert isinstance(plot_score_history([]), Figure)
