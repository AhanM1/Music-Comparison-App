"""Shared plotting helpers so every chart looks consistent."""

import io

import librosa
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

REF_COLOR = "tab:blue"
USER_COLOR = "tab:orange"
PROBLEM_COLORS = {"pitch": "tab:red", "dynamics": "tab:purple", "tone": "tab:green"}
# Shading for problem regions: above the reference (sharp / louder / brighter)
# in a warm colour, below it (flat / softer / darker) in a cool one.
ABOVE_COLOR = "tab:red"
BELOW_COLOR = "tab:cyan"


def new_figure(height: float = 3.0) -> tuple[Figure, Axes]:
    """A wide figure with one set of axes and light grid lines."""
    fig = Figure(figsize=(10, height), layout="constrained")
    ax = fig.add_subplot()
    ax.grid(alpha=0.3)
    return fig, ax


def legend_outside(ax: Axes) -> None:
    """Put the legend just right of the plot, where it can never hide a curve.

    Inside the plot, any corner can collide with a melody that happens to go
    there. The figure's constrained layout makes room for it automatically.
    """
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)


def to_png(fig: Figure, dpi: int = 200) -> bytes:
    """Render a figure to PNG bytes (200 dpi stays sharp on high-resolution screens).

    Rendering is the slow part of a chart (often 10x slower than building it),
    so the app renders each chart once and caches the bytes.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    return buffer.getvalue()


def use_note_axis(ax: Axes, midi: npt.NDArray[np.floating]) -> None:
    """Label a y-axis that holds MIDI note numbers with note names (C4, D#4, ...).

    MIDI numbers are a semitone scale: 60 = C4 (middle C), 69 = A4 (440 Hz).
    """
    finite = midi[np.isfinite(midi)]
    if finite.size == 0:
        return
    low, high = int(np.floor(finite.min())) - 1, int(np.ceil(finite.max())) + 1
    span = high - low
    step = 1 if span <= 12 else 2 if span <= 24 else 12  # keep labels readable
    ticks = np.arange(low, high + 1, step)
    ax.set_ylim(low, high)
    ax.set_yticks(ticks, [librosa.midi_to_note(int(t)) for t in ticks])
