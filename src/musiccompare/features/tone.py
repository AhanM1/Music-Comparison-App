"""Tone (timbre): *how* a note sounds, beyond its pitch and loudness.

All three features come from the spectrogram: the Short-Time Fourier Transform
(STFT) splits each frame into its frequency components, telling us how much
energy is at each frequency at each moment.

- Spectral centroid: the "centre of mass" of the spectrum, in Hz. Higher means
  brighter / edgier; lower means darker / warmer.
- Spectral flatness: 0 = pure tone (energy in a few peaks), 1 = white noise
  (energy spread evenly). Breathy, scratchy, or airy tone raises it.
- MFCCs (mel-frequency cepstral coefficients): a compact summary of the
  spectrum's overall shape on a perceptual (mel) frequency scale. Two sounds
  with similar MFCCs sound similar in timbre. Often called a timbre fingerprint.
"""

import librosa
import numpy as np
import numpy.typing as npt

from musiccompare import config
from musiccompare.models import AudioClip

FloatArray = npt.NDArray[np.float32]


def extract_tone(clip: AudioClip) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute (centroid_hz, flatness, mfcc) per frame.

    ``mfcc`` has shape (config.N_MFCC, n_frames); the other two are 1-D.
    """
    # Compute the magnitude spectrogram once and reuse it for centroid + flatness.
    spectrum = np.abs(librosa.stft(clip.y, n_fft=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH))
    centroid = librosa.feature.spectral_centroid(S=spectrum, sr=clip.sr)[0]
    flatness = librosa.feature.spectral_flatness(S=spectrum)[0]
    mfcc = librosa.feature.mfcc(
        y=clip.y,
        sr=clip.sr,
        n_mfcc=config.N_MFCC,
        n_fft=config.FRAME_LENGTH,
        hop_length=config.HOP_LENGTH,
    )
    return (
        centroid.astype(np.float32),
        flatness.astype(np.float32),
        mfcc.astype(np.float32),
    )
