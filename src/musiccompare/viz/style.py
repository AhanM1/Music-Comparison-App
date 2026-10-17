"""Shared plotting helpers so every chart looks consistent."""

import librosa
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

REF_COLOR = "tab:blue"
USER_COLOR = "tab:orange"
PROBLEM_COLORS = {"pitch": "tab:red", "dynamics": "tab:purple", "tone": "tab:green"}


def new_figure(height: float = 3.0) -> tuple[Figure, Axes]:
    """A wide figure with one set of axes and light grid lines."""
    fig = Figure(figsize=(10, height), layout="constrained")
    ax = fig.add_subplot()
    ax.grid(alpha=0.3)
    return fig, ax


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
