"""Tests for the validation built into the data models."""

import numpy as np
import pytest

from musiccompare.models import FeatureSet


def make_feature_set(n: int = 10, **overrides: object) -> FeatureSet:
    """A valid FeatureSet with ``n`` frames; pass keyword overrides to break it."""
    fields: dict[str, object] = {
        "times": np.arange(n, dtype=np.float64),
        "f0_hz": np.full(n, 440.0),
        "voiced_prob": np.ones(n),
        "rms_db": np.zeros(n),
        "centroid_hz": np.full(n, 1000.0),
        "flatness": np.zeros(n),
        "mfcc": np.zeros((13, n)),
        "extractor_version": "test",
    }
    fields.update(overrides)
    return FeatureSet(**fields)  # type: ignore[arg-type]


def test_valid_feature_set() -> None:
    assert make_feature_set(n=10).n_frames == 10


def test_mismatched_frame_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="rms_db has 9 frames, expected 10"):
        make_feature_set(n=10, rms_db=np.zeros(9))


def test_mfcc_shape_is_checked() -> None:
    with pytest.raises(ValueError, match="mfcc"):
        make_feature_set(n=10, mfcc=np.zeros((13, 11)))
