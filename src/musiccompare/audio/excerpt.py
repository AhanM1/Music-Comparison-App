"""Cut a short excerpt out of a clip, e.g. to listen to one problem spot."""

import math
from dataclasses import replace

from musiccompare.models import AudioClip


def excerpt(clip: AudioClip, start_s: float, end_s: float, pad_s: float = 0.0) -> AudioClip:
    """The part of ``clip`` from ``start_s`` to ``end_s``, plus ``pad_s`` of context each side.

    Padding is cut short at the start and end of the clip.
    """
    start = max(0, math.floor((start_s - pad_s) * clip.sr))
    stop = min(len(clip.y), math.ceil((end_s + pad_s) * clip.sr))
    if start >= stop:
        raise ValueError(f"No audio between {start_s:.2f} s and {end_s:.2f} s")
    return replace(clip, y=clip.y[start:stop])
