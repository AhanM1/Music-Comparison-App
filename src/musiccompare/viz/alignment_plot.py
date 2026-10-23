"""Alignment plots: the DTW cost grid with its path, and both pitch lines in sync."""

import librosa
import numpy as np
from matplotlib.figure import Figure

from musiccompare.analysis.alignment import cost_grid, dtw_resolution
from musiccompare.models import Alignment, FeatureSet
from musiccompare.viz.style import REF_COLOR, USER_COLOR, new_figure, use_note_axis

# The cost grid can have millions of cells; a picture never needs more than this.
MAX_IMAGE_CELLS = 250_000


def plot_alignment_path(ref: FeatureSet, user: FeatureSet, alignment: Alignment) -> Figure:
    """Show the cost grid DTW searched (dark = similar) and the path it chose.

    A straight 45° path means equal tempo. Steeper than 45° = the user was
    slower there; flatter = the user was faster.
    """
    k = dtw_resolution(ref.n_frames, user.n_frames, MAX_IMAGE_CELLS)
    grid = cost_grid(ref.chroma, user.chroma, k)

    fig, ax = new_figure(height=5.0)
    ax.grid(False)
    image = ax.imshow(
        grid.T,  # rows = user frames, so the user's time runs up the y-axis
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=1,
        extent=(0, ref.times[-1], 0, user.times[-1]),
    )
    ax.plot(ref.times[alignment.ref_frames], user.times[alignment.user_frames], color="cyan")
    fig.colorbar(image, ax=ax, label="Chroma distance")
    ax.set(
        title="How your recording lines up with the reference",
        xlabel="Reference time (s)",
        ylabel="Your time (s)",
    )
    return fig


def plot_aligned_pitch(ref: FeatureSet, user: FeatureSet, alignment: Alignment) -> Figure:
    """Both pitch lines on the reference's timeline, so the same note sits at the same x."""
    ref_midi = librosa.hz_to_midi(ref.f0_hz)
    user_midi = librosa.hz_to_midi(user.f0_hz[alignment.user_frames])

    fig, ax = new_figure()
    ax.plot(ref.times, ref_midi, color=REF_COLOR, linewidth=1.5, label="Reference")
    ax.plot(ref.times, user_midi, color=USER_COLOR, linewidth=1.5, label="You (aligned)")
    use_note_axis(ax, np.concatenate([ref_midi, user_midi]))
    ax.legend(loc="upper right")
    ax.set(title="Pitch, aligned to the reference", xlabel="Reference time (s)", ylabel="Note")
    return fig
