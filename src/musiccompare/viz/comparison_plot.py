"""Comparison plots: your take drawn over the reference, with problem spots shaded.

Everything is on the reference's timeline: the alignment tells us which of your
frames matches each reference frame, so "the same note" sits at the same x in
both curves. Your curve is also adjusted the same way the comparison was
(Phase 4), so the picture shows exactly what was scored:

- pitch: shifted by whole octaves if you played the piece in another octave;
- loudness: shifted by the overall level offset (mic/room differences);
- brightness: shifted by the same whole octaves as pitch.

Shaded bands are the problem regions. Warm = you were above the reference
(sharp / louder / brighter), cool = below. Stronger shading = more severe.
"""

import librosa
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

from musiccompare import config
from musiccompare.models import DIMENSIONS, ComparisonResult, Dimension, FeatureSet
from musiccompare.viz.style import (
    ABOVE_COLOR,
    BELOW_COLOR,
    PROBLEM_COLORS,
    REF_COLOR,
    USER_COLOR,
    legend_outside,
    new_figure,
    use_note_axis,
)


def shade_regions(
    ax: Axes, result: ComparisonResult, dimension: Dimension, labels: tuple[str, str]
) -> None:
    """Shade this dimension's problem regions; ``labels`` = (above, below) for the legend."""
    labelled = set()
    for region in result.regions:
        if region.dimension != dimension:
            continue
        above = region.mean_error > 0
        label = labels[0] if above else labels[1]
        ax.axvspan(
            region.start_s,
            region.end_s,
            color=ABOVE_COLOR if above else BELOW_COLOR,
            alpha=0.1 + 0.25 * region.severity,
            linewidth=0,
            # One legend entry per kind; matplotlib skips labels starting with "_".
            label=label if label not in labelled else "_" + label,
        )
        labelled.add(label)


def octave_note(octave_shift: int) -> str:
    """Legend suffix explaining an octave adjustment, e.g. ", 1 octave down"."""
    if octave_shift == 0:
        return ""
    plural = "s" if abs(octave_shift) > 1 else ""
    return f", {abs(octave_shift)} octave{plural} {'down' if octave_shift > 0 else 'up'}"


def plot_pitch_comparison(ref: FeatureSet, user: FeatureSet, result: ComparisonResult) -> Figure:
    """Both melody lines in sync, with sharp/flat stretches shaded."""
    shift = result.differences.octave_shift
    ref_midi = librosa.hz_to_midi(ref.f0_hz)
    user_midi = librosa.hz_to_midi(user.f0_hz[result.alignment.user_frames]) - 12 * shift

    fig, ax = new_figure()
    shade_regions(ax, result, "pitch", ("Sharp", "Flat"))
    ax.plot(ref.times, ref_midi, color=REF_COLOR, linewidth=1.5, label="Reference")
    ax.plot(ref.times, user_midi, color=USER_COLOR, linewidth=1.5, label=f"You{octave_note(shift)}")
    use_note_axis(ax, np.concatenate([ref_midi, user_midi]))
    legend_outside(ax)
    ax.set(title="Pitch", xlabel="Reference time (s)", ylabel="Note")
    return fig


def plot_loudness_comparison(ref: FeatureSet, user: FeatureSet, result: ComparisonResult) -> Figure:
    """Both loudness curves in sync and level-matched, with louder/softer stretches shaded."""
    offset = result.differences.level_offset_db
    user_db = user.rms_db[result.alignment.user_frames] - offset

    fig, ax = new_figure()
    shade_regions(ax, result, "dynamics", ("Louder", "Softer"))
    ax.plot(ref.times, ref.rms_db, color=REF_COLOR, linewidth=1.5, label="Reference")
    ax.plot(ref.times, user_db, color=USER_COLOR, linewidth=1.5, label="You (level-matched)")
    ax.set_ylim(bottom=config.SOUNDING_DB - 5)  # rests drop to -80 dB; don't waste space on them
    legend_outside(ax)
    ax.set(title="Loudness", xlabel="Reference time (s)", ylabel="Loudness (dB)")
    if abs(offset) >= 0.5:
        louder = "louder" if offset > 0 else "quieter"
        note = f"Your recording was {abs(offset):.1f} dB {louder} overall (removed here)"
        ax.text(0.01, 0.03, note, transform=ax.transAxes, fontsize=8, color="gray")
    return fig


def plot_brightness_comparison(
    ref: FeatureSet, user: FeatureSet, result: ComparisonResult
) -> Figure:
    """Both spectral-centroid curves in sync, with brighter/darker stretches shaded."""
    u = result.alignment.user_frames
    shift = result.differences.octave_shift
    # Only draw while each side is actually playing: a centroid of silence is noise.
    ref_hz = np.where(ref.rms_db > config.SOUNDING_DB, ref.centroid_hz, np.nan)
    user_hz = np.where(user.rms_db[u] > config.SOUNDING_DB, user.centroid_hz[u], np.nan) / 2**shift

    fig, ax = new_figure()
    shade_regions(ax, result, "tone", ("Brighter", "Darker"))
    ax.plot(ref.times, ref_hz, color=REF_COLOR, linewidth=1.5, label="Reference")
    ax.plot(ref.times, user_hz, color=USER_COLOR, linewidth=1.5, label=f"You{octave_note(shift)}")
    ax.set_yscale("log")  # brightness is heard on a ratio scale, like pitch
    # Plain-number ticks (500, 700, 1000, ...) instead of matplotlib's default 10^3,
    # spaced more tightly when the curve covers less than a factor of 10.
    low, high = ax.get_ylim()
    subs = (1.0, 2.0, 5.0) if high / low > 10 else (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)
    ax.yaxis.set_major_locator(LogLocator(subs=subs))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    legend_outside(ax)
    ax.set(title="Brightness", xlabel="Reference time (s)", ylabel="Spectral centroid (Hz)")
    return fig


def plot_problem_timeline(result: ComparisonResult, duration_s: float) -> Figure:
    """One lane per dimension, with a bar for every problem region (darker = more severe)."""
    fig, ax = new_figure(height=1.8)
    lanes = {dimension: lane for lane, dimension in enumerate(reversed(DIMENSIONS))}
    for region in result.regions:
        ax.broken_barh(
            [(region.start_s, region.duration_s)],
            (lanes[region.dimension] - 0.35, 0.7),
            color=PROBLEM_COLORS[region.dimension],
            alpha=0.3 + 0.7 * region.severity,
        )
    ax.set_yticks(list(lanes.values()), [d.title() for d in lanes])
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.set_xlim(0, duration_s)
    ax.set(title="Where the problem spots are", xlabel="Reference time (s)")
    return fig
