"""Pitch plots: the melody line (f0 over time) on a note-name axis."""

import librosa
from matplotlib.figure import Figure

from musiccompare.models import FeatureSet
from musiccompare.viz.style import REF_COLOR, new_figure, use_note_axis


def plot_pitch_contour(features: FeatureSet, title: str, color: str = REF_COLOR) -> Figure:
    """Plot one recording's pitch over time. Gaps are frames with no detected pitch."""
    midi = librosa.hz_to_midi(features.f0_hz)  # NaN stays NaN -> matplotlib leaves a gap

    fig, ax = new_figure()
    ax.plot(features.times, midi, color=color, linewidth=1.5)
    use_note_axis(ax, midi)
    ax.set(title=title, xlabel="Time (s)", ylabel="Note")
    return fig
