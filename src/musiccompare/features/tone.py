"""Tone (timbre): *how* a note sounds, beyond its pitch and loudness.

All three features come from the spectrogram: the Short-Time Fourier Transform
(STFT) splits each frame into its frequency components, telling us how much
energy is at each frequency at each moment.

- Spectral centroid: the "centre of mass" of the spectrum, in Hz. Higher means
  brighter / edgier; lower means darker / warmer. Measured with the
  recording's steady background noise removed (see ``remove_noise_floor``).
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
    clean = remove_noise_floor(spectrum)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = librosa.feature.spectral_centroid(S=clean, sr=clip.sr)[0]
    # Nothing left above the noise floor = no tone to measure (NaN, not 0 Hz).
    centroid[clean.sum(axis=0) == 0] = np.nan
    # Flatness *is* the noise detector, so it sees the full spectrum, noise included.
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


def remove_noise_floor(spectrum: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """The spectrum minus the recording's steady background noise (hiss, hum, room).

    Why: hiss spreads energy evenly up to the top of the spectrum, which drags
    the centroid far upward, so a cheap microphone used to read as "184%
    brighter" even with a perfect tone.

    How ("minimum statistics" noise estimation): at each frequency, a note is
    only sounding some of the time, but the background noise is always there.
    So a low percentile of that frequency's level *over time* is the noise
    level. Anything less than a few times that is treated as noise and removed;
    the notes' harmonics are far above it and barely change.

    One catch: a note held through the *whole* recording (a long-tone exercise)
    is "always there" too, and would be removed as if it were noise. So each
    estimate is capped by the frame's median level across all frequencies: a
    note's harmonics fill only a handful of the 1025 frequency bins, so that
    median is the frame's noise level, never the note.
    """
    over_time = np.percentile(spectrum, config.NOISE_FLOOR_PERCENTILE, axis=1, keepdims=True)
    across_frequency = np.median(spectrum, axis=0, keepdims=True)
    floor = np.minimum(over_time, across_frequency)
    clean: npt.NDArray[np.floating] = np.maximum(spectrum - config.NOISE_FLOOR_MARGIN * floor, 0.0)
    return clean
