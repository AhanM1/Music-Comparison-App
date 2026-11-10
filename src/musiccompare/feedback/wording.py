"""Turn numbers into words a musician reads at a glance."""

from musiccompare.models import ProblemRegion


def time_span(start_s: float, end_s: float) -> str:
    """e.g. 0:03.2 – 0:05.0"""
    return " – ".join(f"{int(s // 60)}:{s % 60:04.1f}" for s in (start_s, end_s))


def describe(region: ProblemRegion) -> str:
    """How far off, in words, e.g. "32 cents sharp" or "25% darker"."""
    e = region.mean_error
    if region.dimension == "pitch":
        return f"{abs(e):.0f} cents {'sharp' if e > 0 else 'flat'}"
    if region.dimension == "dynamics":
        return f"{abs(e):.1f} dB {'louder' if e > 0 else 'softer'}"
    ratio = 2**e  # tone: octaves -> brightness ratio
    return f"{(ratio - 1) * 100:.0f}% brighter" if e > 0 else f"{(1 - ratio) * 100:.0f}% darker"


def length_change(user_s: float, ref_s: float) -> str:
    """How your take's length compares, e.g. "+25% length" or "2.6x as long".

    Percentages read well for tempo differences but turn absurd for very
    different lengths ("+1500% length"), so past half / double we switch to a ratio.
    """
    ratio = user_s / ref_s
    if 0.5 <= ratio <= 2:
        return f"{(ratio - 1) * 100:+.0f}% length"
    return f"{ratio:.1f}x as long" if ratio > 2 else f"{1 / ratio:.1f}x shorter"
