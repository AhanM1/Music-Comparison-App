"""Map a difference curve to a single 0-100 score.

Every moment gets its own score:

    score
     100 |==========.
         |            ' .
         |                ' .
       0 |                    '==========
         +-----------|-----------|---------->  |difference|
                 tolerance    zero_at

Inside the tolerance nobody could hear the difference, so it's perfect; past
``zero_at`` it's clearly wrong, so being *even more* wrong costs nothing extra.
That cap matters: one wrong note shouldn't wipe out an otherwise good take.
The final score is the average over every moment that could be compared, so it
reads as "how much of the piece matched".
"""

import numpy as np

from musiccompare.models import FloatArray


def score(smoothed: FloatArray, tolerance: float, zero_at: float) -> float | None:
    """0-100 for one dimension, or None when no frame could be compared (all NaN)."""
    values = np.abs(smoothed[~np.isnan(smoothed)])
    if values.size == 0:
        return None
    per_frame = np.clip((zero_at - values) / (zero_at - tolerance), 0.0, 1.0)
    return float(100 * per_frame.mean())
