"""Run every feature extractor on a clip and bundle the results into a FeatureSet."""

import librosa

from musiccompare import config
from musiccompare.features.chroma import extract_chroma
from musiccompare.features.dynamics import extract_rms_db
from musiccompare.features.pitch import extract_pitch
from musiccompare.features.tone import extract_tone
from musiccompare.models import AudioClip, FeatureSet

# Bump this whenever a change to feature code would change the numbers produced.
# Stored sessions record it, so ML training never mixes incompatible features.
# v2: brightness (spectral centroid) ignores the steady background noise.
EXTRACTOR_VERSION = "2"


def extract_features(clip: AudioClip) -> FeatureSet:
    """Compute pitch, dynamics, tone, and chroma features for every frame of ``clip``."""
    f0_hz, voiced_prob = extract_pitch(clip)
    rms_db = extract_rms_db(clip)
    centroid_hz, flatness, mfcc = extract_tone(clip)
    chroma = extract_chroma(clip, rms_db)

    times = librosa.times_like(f0_hz, sr=clip.sr, hop_length=config.HOP_LENGTH)
    return FeatureSet(
        times=times,
        f0_hz=f0_hz,
        voiced_prob=voiced_prob,
        rms_db=rms_db,
        centroid_hz=centroid_hz,
        flatness=flatness,
        mfcc=mfcc,
        chroma=chroma,
        extractor_version=EXTRACTOR_VERSION,
    )
