"""Dynamics plots: loudness (dB) over time."""

from matplotlib.figure import Figure

from musiccompare import config
from musiccompare.models import FeatureSet
from musiccompare.viz.style import REF_COLOR, new_figure


def plot_loudness_contour(features: FeatureSet, title: str, color: str = REF_COLOR) -> Figure:
    """Plot one recording's loudness curve in dBFS (0 dB = digital maximum)."""
    fig, ax = new_figure()
    ax.plot(features.times, features.rms_db, color=color, linewidth=1.5)
    ax.set(
        title=title,
        xlabel="Time (s)",
        ylabel="Loudness (dBFS)",
        ylim=(max(config.SILENCE_DB, float(features.rms_db.min()) - 3), 0),
    )
    return fig
