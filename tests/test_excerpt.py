"""Tests for cutting excerpts out of a clip."""

import numpy as np
import pytest

from musiccompare.audio.excerpt import excerpt
from musiccompare.models import AudioClip

SR = 1000  # 1 sample per millisecond keeps the arithmetic easy to check


def ramp_clip(duration_s: float = 3.0) -> AudioClip:
    """Each sample's value is its own index, so we can see exactly what was cut."""
    return AudioClip(y=np.arange(int(duration_s * SR), dtype=np.float32), sr=SR, name="ramp")


def test_excerpt_cuts_the_requested_span() -> None:
    piece = excerpt(ramp_clip(), 1.0, 1.5)

    assert piece.duration_s == pytest.approx(0.5)
    assert piece.y[0] == 1000 and piece.y[-1] == 1499
    assert piece.sr == SR and piece.name == "ramp"


def test_excerpt_adds_padding_but_stays_inside_the_clip() -> None:
    clip = ramp_clip(3.0)

    assert excerpt(clip, 1.0, 1.5, pad_s=0.25).duration_s == pytest.approx(1.0)
    start = excerpt(clip, 0.1, 0.5, pad_s=0.5)
    assert start.y[0] == 0  # padding cut short at the beginning...
    end = excerpt(clip, 2.8, 3.0, pad_s=0.5)
    assert end.y[-1] == len(clip.y) - 1  # ...and at the end


def test_excerpt_outside_the_clip_is_an_error() -> None:
    with pytest.raises(ValueError, match="No audio"):
        excerpt(ramp_clip(3.0), 5.0, 6.0)
