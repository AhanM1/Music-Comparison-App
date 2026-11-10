"""Turn frame-by-frame differences into a short list of "problem regions".

A raw difference curve is noisy: a 2-frame glitch where two notes change, or
two vibratos wobbling out of step, would flag dozens of tiny "problems". So:

1. Smooth with a *median* filter. A median ignores short spikes entirely
   (a mean would smear them out instead), while keeping real, sustained
   errors and their edges sharp. That's the "robust statistics" idea: one
   wild value can't drag a median around.
2. Threshold: frames whose smoothed difference is beyond the tolerance are bad.
   Sharp and flat (louder and quieter, ...) are handled separately, so a sharp
   note next to a flat one becomes two regions instead of averaging to "fine".
3. Merge bad stretches separated by tiny gaps, and drop very short ones.
"""

import warnings

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from musiccompare import config
from musiccompare.models import Dimension, FloatArray, ProblemRegion

FRAME_S = config.HOP_LENGTH / config.SAMPLE_RATE  # seconds between frames
SMOOTHING_WIDTH = round(config.COMPARISON_SMOOTHING_S / FRAME_S)  # in frames


def median_smooth(x: FloatArray, width: int) -> FloatArray:
    """Median of each value and its neighbours (``width`` values in total, rounded up to odd).

    NaN means "nothing to compare", so NaNs are skipped inside each window and
    stay NaN in the output: smoothing must never invent an error during a rest.
    """
    x = np.asarray(x, dtype=np.float64)
    half = width // 2
    if half < 1:
        return x.copy()
    windows = sliding_window_view(np.pad(x, half, constant_values=np.nan), 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows -> NaN, as wanted
        smoothed = np.nanmedian(windows, axis=1)
    smoothed[np.isnan(x)] = np.nan
    return smoothed


def find_regions(
    dimension: Dimension,
    times: FloatArray,
    smoothed: FloatArray,
    tolerance: float,
    user_times: FloatArray,
    min_duration_s: float = config.MIN_REGION_SECONDS,
    merge_gap_s: float = config.REGION_MERGE_GAP_S,
) -> list[ProblemRegion]:
    """Problem regions where ``smoothed`` stays beyond ±``tolerance``.

    ``times`` is the reference timeline; ``user_times[i]`` is where the user
    was at reference frame ``i`` (from the alignment).
    """
    regions = []
    with np.errstate(invalid="ignore"):  # NaN compares as False: not a problem
        directions = (smoothed > tolerance, smoothed < -tolerance)
    for bad in directions:
        for start, stop in merge_runs(find_runs(bad), times, merge_gap_s):
            end_s = times[stop - 1] + FRAME_S  # stop is exclusive; a frame lasts FRAME_S
            if end_s - times[start] < min_duration_s:
                continue
            mean_error = float(np.nanmean(smoothed[start:stop]))
            regions.append(
                ProblemRegion(
                    dimension=dimension,
                    start_s=float(times[start]),
                    end_s=float(end_s),
                    user_start_s=float(user_times[start]),
                    user_end_s=float(user_times[stop - 1] + FRAME_S),
                    mean_error=mean_error,
                    severity=severity(mean_error, tolerance),
                )
            )
    return sorted(regions, key=lambda r: r.start_s)


def find_runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """(start, stop) index pairs of every stretch of True values; stop is exclusive.

    e.g. [F, T, T, F, T] -> [(1, 3), (4, 5)]
    """
    # Pad with False on both sides; every False->True step is a start, True->False a stop.
    edges = np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def merge_runs(
    runs: list[tuple[int, int]], times: FloatArray, max_gap_s: float
) -> list[tuple[int, int]]:
    """Join runs whose gap (first good frame -> next bad frame) is shorter than ``max_gap_s``."""
    merged: list[tuple[int, int]] = []
    for start, stop in runs:
        if merged and times[start] - times[merged[-1][1]] < max_gap_s:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return merged


def severity(mean_error: float, tolerance: float) -> float:
    """0 at the tolerance, 0.5 at twice it, 0.75 at four times it, approaching 1."""
    return max(0.0, 1.0 - tolerance / abs(mean_error)) if mean_error else 0.0
