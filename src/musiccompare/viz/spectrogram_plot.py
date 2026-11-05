"""Spectrograms: which frequencies sound at each moment, as a picture.

Time runs left to right, frequency bottom to top, colour = loudness. A played
note shows up as a stack of horizontal lines: the fundamental at the bottom,
its overtones (harmonics) above. That overtone pattern *is* the tone colour,
so comparing the two pictures shows why two tones sound different, e.g. a
brighter tone has stronger high overtones.

We draw a *mel* spectrogram: frequency bands spaced the way hearing works
(fine detail low down, coarser up high). 128 mel bands instead of 1025 FFT
bins also means ~8x less to draw.
"""

import math

import librosa
import librosa.display
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from musiccompare import config
from musiccompare.models import AudioClip

N_MELS = 128
MAX_COLUMNS = 2_000  # a screen is ~2,000 px wide; more time columns can't be seen
DYNAMIC_RANGE_DB = 80.0  # colours span this many dB below the loudest point


def mel_db(clip: AudioClip) -> tuple[npt.NDArray[np.float32], int]:
    """Mel spectrogram in dB, plus the hop length used.

    Long clips use a longer hop, so the picture never has more than MAX_COLUMNS
    time columns: same look, far less drawing.
    """
    hop = config.HOP_LENGTH * max(1, math.ceil(len(clip.y) / config.HOP_LENGTH / MAX_COLUMNS))
    power = librosa.feature.melspectrogram(
        y=clip.y, sr=clip.sr, n_fft=config.FRAME_LENGTH, hop_length=hop, n_mels=N_MELS
    )
    # ref=1.0 (not each clip's own maximum) keeps both clips on one shared scale.
    return librosa.power_to_db(power, ref=1.0, top_db=None).astype(np.float32), hop


def plot_spectrograms(ref_clip: AudioClip, user_clip: AudioClip) -> Figure:
    """The reference's and your spectrogram stacked, with one shared colour scale.

    The scale is shifted so the loudest point of either clip is 0 dB, matching
    the "0 dB = loudest" convention of the loudness plots.
    """
    spectra = [mel_db(ref_clip), mel_db(user_clip)]
    loudest = max(float(s.max()) for s, _ in spectra)

    fig = Figure(figsize=(10, 6), layout="constrained")
    axes = fig.subplots(2, 1)
    titles = ("Reference", "You (on your own timeline)")
    for ax, clip, (s_db, hop), title in zip(
        axes, (ref_clip, user_clip), spectra, titles, strict=True
    ):
        mesh = librosa.display.specshow(
            s_db - loudest,
            sr=clip.sr,
            hop_length=hop,
            x_axis="time",
            y_axis="mel",
            ax=ax,
            vmin=-DYNAMIC_RANGE_DB,
            vmax=0,
            cmap="magma",
        )
        ax.set(title=title)
    fig.colorbar(mesh, ax=axes, label="dB (0 = loudest)", format="%.0f")
    return fig
