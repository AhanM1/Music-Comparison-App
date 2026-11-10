"""Is a recording clean enough to compare note by note?

The app assumes one instrument playing one note at a time. Professional
recordings often break that: a piano or orchestra underneath, a drone, or a
big church acoustic where each note rings on under the next. The pitch
tracker then follows the wrong line (a drone under the melody reads as "400
cents flat"), and loudness and brightness pick up the accompaniment.

We can't separate the instruments (yet: that's "source separation", a Phase 8
idea), but we can notice the problem and say so.

The measurement: **pitch confidence**, the pitch tracker's median certainty
(``voiced_prob``) that one clear pitch is present, over the frames where
something is sounding. A solo line scores ~0.95 whatever the instrument's
timbre; chords or a drone underneath drop it to ~0.02, and heavy reverb (each
note still ringing under the next) to ~0.75.

It's judged *relative to the other recording*: if the reference is much less
clearly pitched than your take, the reference is "busy".

(Why not look at chroma, i.e. "how many pitch classes sound at once"? We tried:
an overtone-rich tone spreads chroma almost as much as a chord does, so a
bright solo violin would have looked "busy" next to a dull one.)
"""

from dataclasses import dataclass

import numpy as np

from musiccompare import config
from musiccompare.models import FeatureSet


@dataclass(frozen=True)
class Clarity:
    """How much one recording sounds like a single clear melodic line."""

    pitch_confidence: float  # 0..1, the pitch tracker's median certainty


def clarity(features: FeatureSet) -> Clarity:
    sounding = features.rms_db > config.SOUNDING_DB
    if not sounding.any():
        return Clarity(pitch_confidence=0.0)
    return Clarity(pitch_confidence=float(np.median(features.voiced_prob[sounding])))


def is_busier(this: Clarity, other: Clarity) -> bool:
    """True if ``this`` recording sounds much less like one clear line than ``other``."""
    return other.pitch_confidence - this.pitch_confidence > config.BUSY_PITCH_CONFIDENCE_DROP


def busy_reference(ref: FeatureSet, user: FeatureSet) -> bool:
    """The reference likely has accompaniment, a drone, or heavy reverb."""
    return is_busier(clarity(ref), clarity(user))


def busy_take(ref: FeatureSet, user: FeatureSet) -> bool:
    """Your take likely has a backing track, accompaniment, or heavy reverb."""
    return is_busier(clarity(user), clarity(ref))
