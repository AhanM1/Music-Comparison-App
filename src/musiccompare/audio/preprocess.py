"""Clean-up steps applied to every clip before analysis.

Each function takes an AudioClip and returns a *new* AudioClip (clips are
immutable), so steps can be chained:  normalize_peak(trim_silence(clip)).
"""

from dataclasses import replace

import librosa
import numpy as np

from musiccompare.models import AudioClip

DEFAULT_TRIM_TOP_DB = 40.0
DEFAULT_PEAK = 0.9


def trim_silence(clip: AudioClip, top_db: float = DEFAULT_TRIM_TOP_DB) -> AudioClip:
    """Remove leading and trailing silence.

    A sample counts as "silence" if it is more than ``top_db`` decibels quieter than
    the loudest part of the clip. Silence in the *middle* (rests) is kept, since
    it's part of the music.
    """
    trimmed, _ = librosa.effects.trim(clip.y, top_db=top_db)
    return replace(clip, y=trimmed)


def normalize_peak(clip: AudioClip, peak: float = DEFAULT_PEAK) -> AudioClip:
    """Scale the clip so its loudest sample reaches ``peak`` (0 < peak <= 1).

    This makes recordings made at different volumes comparable. It is a single
    multiplication, so the *shape* of the dynamics (crescendos, accents) is
    preserved exactly. Peak 0.9 rather than 1.0 leaves a little headroom.
    """
    if not 0 < peak <= 1:
        raise ValueError(f"peak must be in (0, 1], got {peak}")

    current_peak = float(np.max(np.abs(clip.y))) if clip.y.size else 0.0
    if current_peak == 0.0:
        raise ValueError(f"{clip.name!r} is completely silent; nothing to normalize.")

    gain = peak / current_peak
    return replace(clip, y=(clip.y * gain).astype(np.float32))
