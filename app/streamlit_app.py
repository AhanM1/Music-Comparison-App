"""Streamlit entry point.  Run with:  uv run streamlit run app/streamlit_app.py

How Streamlit works: this whole script re-runs from top to bottom every time the
user interacts with a widget (uploads a file, clicks a button, ...). Widgets
return their current value on each run. So:

- Anything slow is wrapped in @st.cache_data: a rerun with the same inputs
  reuses the stored result. That includes the *rendered* chart images, since
  turning a figure into pixels is often slower than building it.
- @st.fragment marks a part of the page that reruns on its own when one of its
  widgets changes, without rerunning (or redrawing) the rest of the page.

This file is deliberately thin: all real work happens in the musiccompare package.
"""

import streamlit as st

from musiccompare import config
from musiccompare.analysis.alignment import align, is_probably_same_piece
from musiccompare.analysis.compare import compare
from musiccompare.audio.excerpt import excerpt
from musiccompare.audio.export import to_wav_bytes
from musiccompare.audio.loader import AudioLoadError, load_audio
from musiccompare.audio.preprocess import normalize_peak, trim_silence
from musiccompare.features.extract import extract_features
from musiccompare.models import DIMENSIONS, AudioClip, ComparisonResult, FeatureSet, ProblemRegion
from musiccompare.viz.alignment_plot import plot_alignment_path
from musiccompare.viz.comparison_plot import (
    plot_brightness_comparison,
    plot_loudness_comparison,
    plot_pitch_comparison,
    plot_problem_timeline,
)
from musiccompare.viz.dynamics_plot import plot_loudness_contour
from musiccompare.viz.overview import plot_waveform
from musiccompare.viz.pitch_plot import plot_pitch_contour
from musiccompare.viz.spectrogram_plot import plot_spectrograms
from musiccompare.viz.style import REF_COLOR, USER_COLOR, to_png

AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]
COMPARISON_TABS = ["Pitch", "Loudness", "Brightness", "Spectrograms", "Alignment"]


@st.cache_data(show_spinner="Analyzing recording...")
def analyze_recording(data: bytes, name: str) -> tuple[AudioClip, FeatureSet]:
    """Load, clean, and extract features. Cached: same bytes in -> instant result."""
    clip = normalize_peak(trim_silence(load_audio(data, name=name)))
    return clip, extract_features(clip)


@st.cache_data(show_spinner=False)
def recording_audio(data: bytes, name: str) -> bytes:
    """The cleaned-up clip as WAV bytes for the audio player."""
    clip, _ = analyze_recording(data, name)
    return to_wav_bytes(clip)


@st.cache_data(show_spinner=False)
def recording_charts(data: bytes, name: str, label: str, color: str) -> list[bytes]:
    """Waveform, pitch, and loudness of one recording, rendered to PNG."""
    clip, features = analyze_recording(data, name)
    return [
        to_png(plot_waveform(clip, color=color)),
        to_png(plot_pitch_contour(features, title=f"{label}: pitch", color=color)),
        to_png(plot_loudness_contour(features, title=f"{label}: loudness", color=color)),
    ]


@st.cache_data(show_spinner="Comparing the two recordings...")
def compare_recordings(
    ref_data: bytes, ref_name: str, user_data: bytes, user_name: str
) -> ComparisonResult:
    """Align and compare two uploads. Keyed on the raw bytes, like analyze_recording."""
    _, ref_features = analyze_recording(ref_data, ref_name)
    _, user_features = analyze_recording(user_data, user_name)
    return compare(ref_features, user_features, align(ref_features, user_features))


@st.cache_data(show_spinner="Drawing the comparison charts...")
def comparison_charts(
    ref_data: bytes, ref_name: str, user_data: bytes, user_name: str
) -> tuple[bytes, list[bytes]]:
    """The problem timeline, plus one PNG per entry of COMPARISON_TABS."""
    ref_clip, ref = analyze_recording(ref_data, ref_name)
    user_clip, user = analyze_recording(user_data, user_name)
    result = compare_recordings(ref_data, ref_name, user_data, user_name)
    timeline = to_png(plot_problem_timeline(result, ref_clip.duration_s))
    tabs = [
        plot_pitch_comparison(ref, user, result),
        plot_loudness_comparison(ref, user, result),
        plot_brightness_comparison(ref, user, result),
        plot_spectrograms(ref_clip, user_clip),
        plot_alignment_path(ref, user, result.alignment),
    ]
    return timeline, [to_png(fig) for fig in tabs]


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


@st.fragment
def listen_to_problems(
    ref_clip: AudioClip, user_clip: AudioClip, regions: list[ProblemRegion]
) -> None:
    """Pick a problem spot and hear that stretch of both recordings.

    As a fragment, changing the selection reruns only this function, so the
    rest of the page isn't rebuilt every time you pick another spot.
    """
    region = st.selectbox(
        "Problem spot",
        regions,
        format_func=lambda r: (
            f"{time_span(r.start_s, r.end_s)}  {r.dimension.title()}: {describe(r)}"
        ),
        key="snippet",
    )
    if region is None:
        return
    st.caption(f"Includes {config.SNIPPET_PAD_S:g} s before and after, for context.")
    spans = [(region.start_s, region.end_s), (region.user_start_s, region.user_end_s)]
    for column, label, clip, (start, end) in zip(
        st.columns(2), ("Reference", "You"), (ref_clip, user_clip), spans, strict=True
    ):
        with column:
            st.markdown(f"**{label}** ({time_span(start, end)})")
            st.audio(to_wav_bytes(excerpt(clip, start, end, pad_s=config.SNIPPET_PAD_S)))


st.set_page_config(page_title="Music Comparison", page_icon="🎻", layout="wide")
st.title("🎻 Music Comparison")
st.caption("Upload a reference recording and your own take of the same passage.")

col_ref, col_user = st.columns(2)
with col_ref:
    ref_file = st.file_uploader("Reference recording", type=AUDIO_TYPES, key="ref")
with col_user:
    user_file = st.file_uploader("Your recording", type=AUDIO_TYPES, key="user")

# Recordings that loaded successfully, by label.
analyzed: dict[str, tuple[AudioClip, FeatureSet]] = {}
for label, upload, column, color in [
    ("Reference", ref_file, col_ref, REF_COLOR),
    ("You", user_file, col_user, USER_COLOR),
]:
    if upload is None:
        continue
    with column:
        try:
            clip, features = analyze_recording(upload.getvalue(), upload.name)
        except (AudioLoadError, ValueError) as e:
            st.error(f"Couldn't use {upload.name}: {e}")
            continue
        analyzed[label] = clip, features
        # Play the trimmed clip, not the raw upload: every time shown in the app
        # is measured from the first note, so the player must start there too.
        st.audio(recording_audio(upload.getvalue(), upload.name))
        st.caption(f"{label}: {clip.duration_s:.1f} s after trimming silence")
        charts = recording_charts(upload.getvalue(), upload.name, label, color)
        for tab, png in zip(st.tabs(["Waveform", "Pitch", "Loudness"]), charts, strict=True):
            tab.image(png, width="stretch")

if ref_file is None or user_file is None:
    st.info("⬆️ Upload both recordings to continue.")
elif len(analyzed) == 2:
    st.header("Comparison")
    (ref_clip, _), (user_clip, _) = analyzed["Reference"], analyzed["You"]
    uploads = (ref_file.getvalue(), ref_file.name, user_file.getvalue(), user_file.name)
    result = compare_recordings(*uploads)
    if not is_probably_same_piece(result.alignment):
        st.warning(
            "These recordings don't look like the same music "
            f"(match distance {result.alignment.cost:.2f}), so the comparison may not mean much."
        )
    ref_s, user_s = ref_clip.duration_s, user_clip.duration_s
    st.caption(
        f"You took {user_s:.1f} s, the reference took {ref_s:.1f} s "
        f"({(user_s / ref_s - 1) * 100:+.0f}% length)."
    )

    st.subheader("Scores")
    for column, dimension in zip(st.columns(len(DIMENSIONS)), DIMENSIONS, strict=True):
        value = result.scores[dimension]
        column.metric(dimension.title(), "n/a" if value is None else f"{value:.0f}/100")
    timeline, tab_charts = comparison_charts(*uploads)
    st.image(timeline, width="stretch")
    for tab, png in zip(st.tabs(COMPARISON_TABS), tab_charts, strict=True):
        tab.image(png, width="stretch")

    st.subheader("Problem spots")
    if not result.regions:
        st.success("No problem spots: everything stayed within tolerance.")
    else:
        st.dataframe(
            [
                {
                    "Reference time": time_span(r.start_s, r.end_s),
                    "Your time": time_span(r.user_start_s, r.user_end_s),
                    "What": r.dimension.title(),
                    "How far off": describe(r),
                    "Severity": r.severity,
                }
                for r in result.regions
            ],
            hide_index=True,
            column_config={"Severity": st.column_config.ProgressColumn(min_value=0, max_value=1)},
        )
        st.subheader("Listen")
        listen_to_problems(ref_clip, user_clip, result.regions)
