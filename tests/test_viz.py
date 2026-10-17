"""Tests for musiccompare.viz: figures build without errors and show the right data."""

import numpy as np
from matplotlib.figure import Figure

from musiccompare import config
from musiccompare.features.extract import extract_features
from musiccompare.models import AudioClip
from musiccompare.viz.dynamics_plot import plot_loudness_contour
from musiccompare.viz.overview import plot_waveform, waveform_envelope
from musiccompare.viz.pitch_plot import plot_pitch_contour

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
