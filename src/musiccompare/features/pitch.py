"""Pitch: what note is being played, frame by frame, and how far off it is.

We track the *fundamental frequency* (f0) in two passes:

1. pYIN (probabilistic YIN) decides *whether* each frame has a pitch and roughly
   *which*: it scores candidate periods at which the waveform repeats, then a
   hidden Markov model picks a smooth, plausible path through time. It's robust,
   but only considers pitches on a grid of 0.1 semitone (10 cents), so readings
   can be up to ~10 cents off. That's too coarse for intonation feedback.
2. Refinement: near pYIN's estimate we evaluate the *difference function*
       d(lag) = sum((x[j] - x[j + lag]) ** 2)
   which is smallest when ``lag`` equals the true period (the waveform lines up
   with itself). Fitting a parabola through the three points around the minimum
   finds the period to a fraction of a sample. Benchmarks on synthetic tones,
   including a noisy guitar-like tone with a weak fundamental, gave < 0.5 cent error.

(Things we measured and rejected: a finer pYIN grid was much slower *and* less
accurate in librosa 1.0; librosa.yin has a sharp bias of up to +6 cents on low
notes. Always measure!)

Pitch differences are measured in **cents** rather than Hz, because hearing is
logarithmic: 100 cents = 1 semitone at any register, while 10 Hz is a huge
error at A2 (110 Hz) but tiny at A6 (1760 Hz).
"""

import librosa
import numpy as np
import numpy.typing as npt

from musiccompare import config
from musiccompare.models import AudioClip

CENTS_PER_OCTAVE = 1200.0

# How far around pYIN's period estimate to search, as a fraction of the period.
# 1.5% is about 25 cents: comfortably more than pYIN's worst-case grid error.
REFINE_SEARCH_FRACTION = 0.015


def extract_pitch(
    clip: AudioClip,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Track pitch: pYIN for voicing and the coarse path, then refine each frame.

    Returns:
        (f0_hz, voiced_prob): f0 per frame (NaN where unvoiced) and the
        probability (0..1) that each frame contains a pitched sound.
    """
    coarse_hz, _voiced_flag, voiced_prob = librosa.pyin(
        clip.y,
        fmin=librosa.note_to_hz(config.FMIN),
        fmax=librosa.note_to_hz(config.FMAX),
        sr=clip.sr,
        frame_length=config.FRAME_LENGTH,
        hop_length=config.HOP_LENGTH,
    )
    return refine_f0(clip.y, coarse_hz, clip.sr), voiced_prob


def refine_f0(
    y: npt.NDArray[np.float32],
    coarse_hz: npt.NDArray[np.float64],
    sr: int,
    frame_length: int = config.FRAME_LENGTH,
    hop_length: int = config.HOP_LENGTH,
) -> npt.NDArray[np.float64]:
    """Sharpen coarse f0 estimates with parabolic interpolation of the difference function.

    Frames where the refinement isn't trustworthy (no clear minimum near the
    coarse estimate) keep their coarse value. NaN (unvoiced) frames stay NaN.
    """
    # Same framing as pYIN (center=True): frame i is centred on sample i * hop_length.
    padded = np.pad(y, frame_length // 2)
    frames = librosa.util.frame(padded, frame_length=frame_length, hop_length=hop_length)

    refined = coarse_hz.copy()
    for i in np.flatnonzero(np.isfinite(coarse_hz)):
        x = frames[:, i]
        period = round(sr / coarse_hz[i])  # in samples
        search = int(np.ceil(REFINE_SEARCH_FRACTION * period)) + 1
        lags = np.arange(period - search, period + search + 1)

        window = frame_length - lags[-1]  # compare the same number of samples at every lag
        d = np.array([np.sum((x[:window] - x[lag : lag + window]) ** 2) for lag in lags])

        j = int(np.argmin(d))
        if j == 0 or j == len(d) - 1:
            continue  # minimum at the edge of the search: not a clear dip, keep coarse
        left, mid, right = d[j - 1], d[j], d[j + 1]
        curvature = left - 2 * mid + right
        if curvature <= 0:
            continue  # flat or upside-down parabola: no reliable minimum

        # Vertex of the parabola through the three points, as an offset from lags[j].
        offset = 0.5 * (left - right) / curvature
        refined[i] = sr / (lags[j] + offset)
    return refined


def hz_to_cents(f_hz: npt.ArrayLike, ref_hz: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """How many cents ``f_hz`` is above ``ref_hz`` (negative = below).

    cents = 1200 * log2(f / ref).   e.g. 880 Hz vs 440 Hz -> +1200 (one octave).
    NaN in either input gives NaN out (no pitch -> no comparison).
    """
    f = np.asarray(f_hz, dtype=np.float64)
    ref = np.asarray(ref_hz, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return CENTS_PER_OCTAVE * np.log2(f / ref)


def fold_octaves(cents: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Wrap a cents difference into [-600, +600) by removing whole octaves.

    Pitch trackers sometimes lock onto the wrong octave (reporting 880 Hz for a
    440 Hz note). Musically that's the same note, so +1230 cents becomes +30.
    """
    c = np.asarray(cents, dtype=np.float64)
    half = CENTS_PER_OCTAVE / 2
    return np.mod(c + half, CENTS_PER_OCTAVE) - half
