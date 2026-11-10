"""Your scores over time: one line per dimension, one point per saved session."""

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from musiccompare.models import DIMENSIONS, Dimension
from musiccompare.viz.style import PROBLEM_COLORS, legend_outside, new_figure

# Up to this many sessions, each gets its own labelled tick (e.g. its date);
# beyond it, labels would overlap, so the axis just counts sessions.
MAX_LABELLED_SESSIONS = 10


def plot_score_history(
    scores: Sequence[Mapping[Dimension, float | None]],
    labels: Sequence[str] | None = None,
) -> Figure:
    """Scores (0-100) per session, oldest first.

    The x-axis is the session *number*, not the date: practice happens in
    bursts (five takes one evening, then nothing for a week), and on a date axis
    the burst would be squashed into one spot. A missing score (None: nothing
    to compare) leaves a gap in that line rather than pretending it was 0.
    """
    if labels is not None and len(labels) != len(scores):
        raise ValueError(f"{len(labels)} labels for {len(scores)} sessions")
    fig, ax = new_figure(height=3.5)
    x = np.arange(1, len(scores) + 1)
    for dimension in DIMENSIONS:
        y = [np.nan if (v := s[dimension]) is None else v for s in scores]
        ax.plot(x, y, marker="o", label=dimension.title(), color=PROBLEM_COLORS[dimension])
    ax.set_ylim(0, 105)  # a little headroom so points at 100 aren't cut in half
    ax.set_ylabel("Score")
    if labels is not None and len(labels) <= MAX_LABELLED_SESSIONS:
        ax.set_xticks(x, labels, rotation=30, ha="right")
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # no "session 2.5"
        ax.set_xlabel("Session")
    if len(scores):
        ax.set_xlim(0.5, len(scores) + 0.5)
    ax.set_title("Scores over time")
    legend_outside(ax)
    return fig
