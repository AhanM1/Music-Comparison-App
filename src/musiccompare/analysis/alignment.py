"""Line two performances up in time with Dynamic Time Warping (DTW).

The problem: the student may start late, rush one bar, and drag the next. To
compare "the same note" in both recordings we need, for every moment of the
reference, the matching moment of the student's take.

DTW solves this with dynamic programming:
1. Build a cost grid C[i, j] = how different reference frame i is from user
   frame j (here: cosine distance between their chroma vectors).
2. Fill an accumulated-cost grid D, where D[i, j] is the cheapest way to get
   from (0, 0) to (i, j) moving only forward in time:
       D[i, j] = C[i, j] + min(D[i-1, j-1], D[i-1, j], D[i, j-1])
3. Backtrack from the end to the start, following the cheapest steps.
   That route is the *warping path*.

A diagonal step means both played at the same speed; a vertical/horizontal
step means one of them lingered.

Two refinements on top of plain DTW:
- A small penalty on vertical/horizontal steps (``config.DTW_STEP_PENALTY``),
  so DTW takes the diagonal unless the music says otherwise.
- Smoothing the final mapping (``config.ALIGNMENT_SMOOTHING_S``). Inside a held
  note every frame looks alike, so DTW only really knows where notes start and
  end; a moving average spreads the timing evenly across each note.

What DTW can't do is leave anything out: its path must run from both starts
to both ends. So a count-in, a repeat, or a skipped passage shows up as a long
horizontal or vertical run, where one recording moves on while the other stays
put. A held note makes such runs too, but there both sides sound alike (low
cost); unmatched material sounds *different* (high cost). ``unmatched_spans``
uses that to report those stretches instead of silently absorbing them.

Known limit: because the path must end at both ends, a take that stops early
gets its last notes matched to the reference's last notes, so the "skipped"
stretch can be reported a little before where you actually stopped. How
*much* was skipped is reliable; exactly where is approximate. (Open-end DTW,
which lets the path stop early, would fix this; it's a candidate for later.)
"""

import math

import librosa
import numpy as np
import numpy.typing as npt

from musiccompare import config
from musiccompare.models import Alignment, FeatureSet, IntArray


def align(ref: FeatureSet, user: FeatureSet, max_cells: int = config.MAX_DTW_CELLS) -> Alignment:
    """Find, for every reference frame, the matching frame of the user's recording."""
    resolution = dtw_resolution(ref.n_frames, user.n_frames, max_cells)
    grid = cost_grid(ref.chroma, user.chroma, resolution)

    # librosa's steps, in order: both move, only the user moves, only the reference moves.
    penalty = config.DTW_STEP_PENALTY
    _, path = librosa.sequence.dtw(C=grid, weights_add=np.array([0.0, penalty, penalty]))
    path = path[::-1]  # librosa returns the path end -> start; flip it

    mean_cost = float(np.mean(grid[path[:, 0], path[:, 1]]))
    smooth_frames = round(config.ALIGNMENT_SMOOTHING_S * config.SAMPLE_RATE / config.HOP_LENGTH)
    full_path = upsample_path(path, resolution, ref.n_frames, user.n_frames, smooth_frames)
    return Alignment(
        path=full_path,
        cost=mean_cost,
        resolution=resolution,
        # A horizontal run (the reference stays, the user moves) is user material
        # the reference doesn't have; a vertical run is reference material skipped.
        extra_user=unmatched_spans(path, grid, resolution, stays=0, n_frames=user.n_frames),
        skipped_ref=unmatched_spans(path, grid, resolution, stays=1, n_frames=ref.n_frames),
    )


def unmatched_spans(
    path: IntArray,
    grid: npt.NDArray[np.float64],
    k: int,
    stays: int,
    n_frames: int,
    min_s: float = config.UNMATCHED_MIN_S,
    min_cost: float = config.UNMATCHED_MIN_COST,
) -> IntArray:
    """Stretches where one recording moves on alone through material that doesn't match.

    ``path`` is the coarse DTW path (``k`` frames per step) and ``stays`` the
    column (0 = reference, 1 = user) that stands still during the run. Returns
    (start, stop) full-resolution frame spans of the *other* recording, the one
    that moved, for runs lasting at least ``min_s`` whose mean cost is at least
    ``min_cost`` (a held note costs ~0; unrelated notes ~0.5-0.9).
    """
    moves = 1 - stays
    min_steps = math.ceil(min_s * config.SAMPLE_RATE / config.HOP_LENGTH / k)
    still = np.diff(path[:, stays]) == 0  # step i -> i+1 kept this column fixed
    edges = np.diff(np.concatenate([[0], still.astype(np.int8), [0]]))
    spans = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        if stop - start < min_steps:
            continue
        cells = path[start : stop + 1]  # the run's steps join these path points
        if grid[cells[:, 0], cells[:, 1]].mean() < min_cost:
            continue  # lingering on a matching note, e.g. a longer held note
        first, last = cells[1, moves], cells[-1, moves]  # cells[0] is still matched
        spans.append((first * k, min((last + 1) * k, n_frames)))
    return np.array(spans, dtype=np.intp).reshape(-1, 2)


def cost_grid(
    ref_chroma: npt.NDArray[np.floating], user_chroma: npt.NDArray[np.floating], k: int = 1
) -> npt.NDArray[np.float64]:
    """Cosine distance between every reference frame and every user frame.

    Shape (n_ref / k, n_user / k): 0 = same notes sounding, 1 = nothing in common.
    """
    ref_chroma = downsample_chroma(ref_chroma, k)
    user_chroma = downsample_chroma(user_chroma, k)
    # Columns are unit vectors, so a dot product is the cosine similarity.
    grid: npt.NDArray[np.float64] = 1.0 - ref_chroma.T @ user_chroma
    return grid


def dtw_resolution(n_ref: int, n_user: int, max_cells: int) -> int:
    """Smallest group size k such that a (n_ref/k) x (n_user/k) grid fits the budget."""
    k = max(1, math.ceil(math.sqrt(n_ref * n_user / max_cells)))
    while math.ceil(n_ref / k) * math.ceil(n_user / k) > max_cells:
        k += 1
    return k


def downsample_chroma(chroma: npt.NDArray[np.floating], k: int) -> npt.NDArray[np.float64]:
    """Average every ``k`` consecutive frames, then re-normalize columns to unit length."""
    if k == 1:
        return chroma.astype(np.float64)
    n_groups = math.ceil(chroma.shape[1] / k)
    padded = np.pad(chroma, ((0, 0), (0, n_groups * k - chroma.shape[1])), mode="edge")
    grouped = padded.reshape(chroma.shape[0], n_groups, k).mean(axis=2)
    return grouped / np.linalg.norm(grouped, axis=0, keepdims=True)


def upsample_path(
    path: IntArray, k: int, n_ref: int, n_user: int, smooth_frames: int = 1
) -> IntArray:
    """Convert a DTW path into one (ref_frame, user_frame) row per full-resolution ref frame.

    The DTW path may visit a reference frame several times (the user lingered);
    we use the average user position for it. Positions between path points are
    filled in by linear interpolation, then smoothed over ``smooth_frames``.
    """
    ref_idx, user_idx = path[:, 0], path[:, 1]
    unique_ref = np.unique(ref_idx)
    mean_user = np.array([user_idx[ref_idx == r].mean() for r in unique_ref])

    # Coarse frame g covers full frames g*k ... g*k + k-1; its centre is g*k + (k-1)/2.
    centre = (k - 1) / 2
    all_ref = np.arange(n_ref)
    user_pos = np.interp(all_ref, unique_ref * k + centre, mean_user * k + centre)
    user_pos = moving_average(user_pos, smooth_frames)

    user_frames = np.clip(np.round(user_pos), 0, n_user - 1).astype(np.intp)
    user_frames = np.maximum.accumulate(user_frames)  # rounding must never go backwards
    return np.column_stack([all_ref, user_frames])


def moving_average(x: npt.NDArray[np.float64], width: int) -> npt.NDArray[np.float64]:
    """Average each value with its neighbours, ``width`` values in total (rounded up to odd).

    The ends are padded by point reflection (mirroring through the end value),
    which leaves a straight line exactly unchanged, so a steady tempo is never
    bent near the start or end. Averaging a rising sequence keeps it rising.
    """
    half = min(width // 2, len(x) - 1)
    if half < 1:
        return x
    before = 2 * x[0] - x[half:0:-1]
    after = 2 * x[-1] - x[-2 : -half - 2 : -1]
    padded = np.concatenate([before, x, after])
    return np.convolve(padded, np.ones(2 * half + 1) / (2 * half + 1), mode="valid")


def is_probably_same_piece(alignment: Alignment) -> bool:
    """False when the recordings are so different they're likely not the same music."""
    return alignment.cost <= config.ALIGNMENT_COST_WARNING
