"""Chroma: how strongly each of the 12 pitch classes (C, C#, D, ... B) sounds.

Chroma folds all octaves together: a C2 and a C5 both count as "C". That makes
it robust to the things that differ between two performances of the same
music (instrument tone, loudness, octave errors) while still capturing *which
notes* are playing. That's exactly what we need to line two recordings up.
"""

import librosa
import numpy as np
import numpy.typing as npt

from musiccompare import config
from musiccompare.models import AudioClip

N_CHROMA = 12


def extract_chroma(clip: AudioClip, rms_db: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Unit-length chroma vectors per frame, shape (12, n_frames).

    ``rms_db`` (from dynamics) marks quiet frames: librosa scales every frame so
    its strongest pitch class is 1, which would blow faint background noise up
    into a fake "note". Quiet frames get a neutral, uniform vector instead.
    """
    chroma = librosa.feature.chroma_cqt(y=clip.y, sr=clip.sr, hop_length=config.HOP_LENGTH)

    quiet = rms_db < config.CHROMA_SILENCE_DB
    chroma[:, quiet] = 1.0

    # Scale every column to length 1, so comparing two frames (cosine distance)
    # depends only on *which* notes sound, not on how loud they are.
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True)
    return chroma.astype(np.float32)
