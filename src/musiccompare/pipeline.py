"""The whole analysis, end to end, in two calls. Any front end uses these.

    clip, features = prepare(file_bytes, "take.wav")     # once per recording
    result = compare_takes(ref_features, user_features)  # once per pair

The Streamlit app calls exactly these, and so can a batch script, a test, or
a future ML training job, so they all analyse audio identically. That's why
the steps live here and not in the app: "the UI is a thin layer" (PLAN.md 2.1).

``prepare`` is also where unusable input is turned away with a clear message,
before it can produce meaningless scores and tips.
"""

from musiccompare import config
from musiccompare.analysis.alignment import align
from musiccompare.analysis.compare import COMPARISON_VERSION, compare
from musiccompare.audio.loader import AudioLoadError, AudioSource, load_audio
from musiccompare.audio.preprocess import normalize_peak, trim_silence
from musiccompare.features.extract import EXTRACTOR_VERSION, extract_features
from musiccompare.models import AudioClip, ComparisonResult, FeatureSet

# Everything that changes the numbers an analysis produces. Saved sessions are
# identified by their recordings *and* this, so new analysis code never shows
# stale scores for a take you saved before an upgrade.
ANALYSIS_VERSION = f"features-{EXTRACTOR_VERSION}/comparison-{COMPARISON_VERSION}"


def prepare(source: AudioSource, name: str | None = None) -> tuple[AudioClip, FeatureSet]:
    """Load, check, clean up, and measure one recording.

    Raises:
        AudioLoadError: if the file can't be used (unreadable, damaged, silent,
            too short, or too long), with a message meant for the player.
    """
    clip = load_audio(source, name=name)
    if not clip.y.any():
        raise AudioLoadError(f"{clip.name!r} is completely silent.")
    clip = trim_silence(clip)
    if clip.duration_s < config.MIN_DURATION_S:
        raise AudioLoadError(
            f"{clip.name!r} has only {clip.duration_s:.1f} s of sound once silence is "
            f"trimmed; at least {config.MIN_DURATION_S:g} s is needed to compare it."
        )
    clip = normalize_peak(clip)
    return clip, extract_features(clip)


def compare_takes(ref: FeatureSet, user: FeatureSet) -> ComparisonResult:
    """Line the two recordings up in time, then compare them."""
    return compare(ref, user, align(ref, user))
