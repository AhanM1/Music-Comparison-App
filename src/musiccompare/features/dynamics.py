"""Dynamics: how loud the playing is, frame by frame.

RMS (root-mean-square) energy is the standard measure of a signal's strength:
square every sample in the frame, take the mean, take the square root.
We convert it to decibels because loudness perception is logarithmic too:
every -6 dB is roughly "half the amplitude", every -20 dB is a tenth.

We report dBFS (decibels relative to Full Scale): 0 dB is the loudest possible
digital signal, so all values are negative. A sine wave of amplitude A has
RMS A/√2, e.g. amplitude 0.5 -> 20·log10(0.354) ≈ -9 dB.
"""

import librosa
import numpy as np
import numpy.typing as npt

from musiccompare import config
from musiccompare.models import AudioClip


def extract_rms_db(clip: AudioClip) -> npt.NDArray[np.float32]:
    """Loudness per frame in dBFS, floored at config.SILENCE_DB."""
    rms = librosa.feature.rms(
        y=clip.y, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH
    )[0]  # rms() returns shape (1, n_frames); take the single row
    amin = 10 ** (config.SILENCE_DB / 20)  # the amplitude that equals SILENCE_DB
    db = librosa.amplitude_to_db(rms, ref=1.0, amin=amin, top_db=None)
    return np.asarray(db, dtype=np.float32)
