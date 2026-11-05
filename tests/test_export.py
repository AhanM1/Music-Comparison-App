"""Tests for writing clips back out as sound files."""

import io

import numpy as np
import soundfile as sf

from musiccompare.audio.export import to_wav_bytes
from musiccompare.models import AudioClip


def test_wav_bytes_round_trip(make_tone) -> None:
    clip = AudioClip(y=make_tone(amplitude=0.5), sr=22_050, name="tone")

    y, sr = sf.read(io.BytesIO(to_wav_bytes(clip)), dtype="float32")

    assert sr == clip.sr
    # 16-bit samples are accurate to 1/32768, far below anything audible.
    np.testing.assert_allclose(y, clip.y, atol=1 / 32768)
