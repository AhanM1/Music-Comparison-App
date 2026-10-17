"""Overview plots: the waveform of a clip.

All viz functions build and *return* a matplotlib Figure. They never call
plt.show() and never touch matplotlib.pyplot, whose hidden global "current
figure" state is unsafe in a web server that serves several users at once.
The caller decides what to do with the figure (Streamlit: st.pyplot(fig)).
"""

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from musiccompare.models import AudioClip

# A screen is ~2,000 pixels wide; plotting millions of samples is slow and
# looks identical. We draw a min/max envelope with about this many points.
MAX_PLOT_POINTS = 2_000


def waveform_envelope(
    y: npt.NDArray[np.float32], sr: int, n_points: int = MAX_PLOT_POINTS
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Downsample a waveform to (times, lower, upper) for fast, faithful plotting.

    The signal is split into ``n_points`` equal blocks; for each block we keep the
    minimum and maximum sample. Unlike simply taking every Nth sample, this never
    hides a loud peak, so the plot still shows the true dynamics.
    """
    if len(y) <= n_points:
        times = np.arange(len(y)) / sr
        return times, y, y

    block = len(y) // n_points
    usable = y[: block * n_points].reshape(n_points, block)  # one row per block
    times = (np.arange(n_points) * block + block / 2) / sr  # centre of each block
    return times, usable.min(axis=1), usable.max(axis=1)


def plot_waveform(clip: AudioClip, color: str = "tab:blue") -> Figure:
    """Plot a clip's waveform against time in seconds."""
    times, lower, upper = waveform_envelope(clip.y, clip.sr)

    fig = Figure(figsize=(10, 2.5), layout="constrained")
    ax = fig.add_subplot()
    ax.fill_between(times, lower, upper, color=color, linewidth=0.5)
    ax.set(
        title=clip.name,
        xlabel="Time (s)",
        ylabel="Amplitude",
        xlim=(0, clip.duration_s),
        ylim=(-1.05, 1.05),
    )
    ax.grid(alpha=0.3)
    return fig
