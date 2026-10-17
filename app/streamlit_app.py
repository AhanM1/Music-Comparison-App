"""Streamlit entry point.  Run with:  uv run streamlit run app/streamlit_app.py

How Streamlit works: this whole script re-runs from top to bottom every time the
user interacts with a widget (uploads a file, clicks a button, ...). Widgets
return their current value on each run. Anything expensive is wrapped in
@st.cache_data so a rerun reuses the earlier result instead of recomputing.

This file is deliberately thin: all real work happens in the musiccompare package.
"""

import streamlit as st

from musiccompare.audio.loader import AudioLoadError, load_audio
from musiccompare.audio.preprocess import normalize_peak, trim_silence
from musiccompare.models import AudioClip
from musiccompare.viz.overview import plot_waveform

AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]


@st.cache_data(show_spinner="Loading audio...")
def prepare_clip(data: bytes, name: str) -> AudioClip:
    """Load + clean one recording. Cached: same bytes in -> instant result."""
    return normalize_peak(trim_silence(load_audio(data, name=name)))


st.set_page_config(page_title="Music Comparison", page_icon="🎻", layout="wide")
st.title("🎻 Music Comparison")
st.caption("Upload a reference recording and your own take of the same passage.")

col_ref, col_user = st.columns(2)
with col_ref:
    ref_file = st.file_uploader("Reference recording", type=AUDIO_TYPES, key="ref")
with col_user:
    user_file = st.file_uploader("Your recording", type=AUDIO_TYPES, key="user")

for label, upload, column, color in [
    ("Reference", ref_file, col_ref, "tab:blue"),
    ("You", user_file, col_user, "tab:orange"),
]:
    if upload is None:
        continue
    with column:
        try:
            clip = prepare_clip(upload.getvalue(), upload.name)
        except (AudioLoadError, ValueError) as e:
            st.error(f"Couldn't use {upload.name}: {e}")
            continue
        st.audio(upload)
        st.pyplot(plot_waveform(clip, color=color))
        st.caption(f"{label}: {clip.duration_s:.1f} s after trimming silence")

if ref_file is None or user_file is None:
    st.info("⬆️ Upload both recordings to continue.")
