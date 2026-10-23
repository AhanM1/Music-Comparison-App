"""Streamlit entry point.  Run with:  uv run streamlit run app/streamlit_app.py

How Streamlit works: this whole script re-runs from top to bottom every time the
user interacts with a widget (uploads a file, clicks a button, ...). Widgets
return their current value on each run. Anything expensive is wrapped in
@st.cache_data so a rerun reuses the earlier result instead of recomputing.

This file is deliberately thin: all real work happens in the musiccompare package.
"""

import streamlit as st

from musiccompare.analysis.alignment import align, is_probably_same_piece
from musiccompare.analysis.compare import compare
from musiccompare.audio.loader import AudioLoadError, load_audio
from musiccompare.audio.preprocess import normalize_peak, trim_silence
from musiccompare.features.extract import extract_features
from musiccompare.models import DIMENSIONS, Alignment, AudioClip, FeatureSet, ProblemRegion
from musiccompare.viz.alignment_plot import plot_aligned_pitch, plot_alignment_path
from musiccompare.viz.dynamics_plot import plot_loudness_contour
from musiccompare.viz.overview import plot_waveform
from musiccompare.viz.pitch_plot import plot_pitch_contour
from musiccompare.viz.style import REF_COLOR, USER_COLOR

AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]


@st.cache_data(show_spinner="Analyzing recording...")
def analyze_recording(data: bytes, name: str) -> tuple[AudioClip, FeatureSet]:
    """Load, clean, and extract features. Cached: same bytes in -> instant result."""
    clip = normalize_peak(trim_silence(load_audio(data, name=name)))
    return clip, extract_features(clip)


@st.cache_data(show_spinner="Lining up the two recordings...")
def align_recordings(ref_data: bytes, ref_name: str, user_data: bytes, user_name: str) -> Alignment:
    """Align two uploads. Keyed on the raw bytes, like analyze_recording."""
    _, ref_features = analyze_recording(ref_data, ref_name)
    _, user_features = analyze_recording(user_data, user_name)
    return align(ref_features, user_features)


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
        st.audio(upload)
        st.caption(f"{label}: {clip.duration_s:.1f} s after trimming silence")
        tab_wave, tab_pitch, tab_loud = st.tabs(["Waveform", "Pitch", "Loudness"])
        with tab_wave:
            st.pyplot(plot_waveform(clip, color=color))
        with tab_pitch:
            st.pyplot(plot_pitch_contour(features, title=f"{label}: pitch", color=color))
        with tab_loud:
            st.pyplot(plot_loudness_contour(features, title=f"{label}: loudness", color=color))

if ref_file is None or user_file is None:
    st.info("⬆️ Upload both recordings to continue.")
elif len(analyzed) == 2:
    st.header("Comparison")
    (ref_clip, ref_features), (user_clip, user_features) = analyzed["Reference"], analyzed["You"]
    alignment = align_recordings(
        ref_file.getvalue(), ref_file.name, user_file.getvalue(), user_file.name
    )
    if not is_probably_same_piece(alignment):
        st.warning(
            "These recordings don't look like the same music "
            f"(match distance {alignment.cost:.2f}), so the comparison may not mean much."
        )
    ref_s, user_s = ref_clip.duration_s, user_clip.duration_s
    st.caption(
        f"You took {user_s:.1f} s, the reference took {ref_s:.1f} s "
        f"({(user_s / ref_s - 1) * 100:+.0f}% length)."
    )
    tab_sync, tab_path = st.tabs(["Pitch in sync", "Alignment path"])
    with tab_sync:
        st.pyplot(plot_aligned_pitch(ref_features, user_features, alignment))
    with tab_path:
        st.pyplot(plot_alignment_path(ref_features, user_features, alignment))

    result = compare(ref_features, user_features, alignment)
    st.subheader("Scores")
    for column, dimension in zip(st.columns(len(DIMENSIONS)), DIMENSIONS, strict=True):
        value = result.scores[dimension]
        column.metric(dimension.title(), "n/a" if value is None else f"{value:.0f}/100")

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
