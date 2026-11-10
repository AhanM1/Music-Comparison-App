"""Streamlit entry point.  Run with:  uv run streamlit run app/streamlit_app.py

How Streamlit works: this whole script re-runs from top to bottom every time the
user interacts with a widget (uploads a file, clicks a button, ...). Widgets
return their current value on each run. So:

- Anything slow is wrapped in @st.cache_data: a rerun with the same inputs
  reuses the stored result. That includes the *rendered* chart images, since
  turning a figure into pixels is often slower than building it.
- @st.fragment marks a part of the page that reruns on its own when one of its
  widgets changes, without rerunning (or redrawing) the rest of the page.

Saving: every comparison is stored on this computer (scores, measurements, the
tips and your 👍/👎 on them) unless you switch it off in the sidebar. The
recordings themselves are only kept if you tick the box asking for that.

This file is deliberately thin: all real work happens in the musiccompare package.
"""

import sqlite3
from dataclasses import dataclass

import streamlit as st

from musiccompare import config
from musiccompare.analysis.alignment import align, is_probably_same_piece
from musiccompare.analysis.compare import compare
from musiccompare.audio.excerpt import excerpt
from musiccompare.audio.export import to_wav_bytes
from musiccompare.audio.loader import AudioLoadError, load_audio
from musiccompare.audio.preprocess import normalize_peak, trim_silence
from musiccompare.data.store import Rating, SessionStore, default_data_dir, fingerprint
from musiccompare.features.extract import extract_features
from musiccompare.feedback.base import FeedbackProvider
from musiccompare.feedback.rules import RuleBasedFeedback
from musiccompare.feedback.wording import describe, time_span
from musiccompare.models import (
    DIMENSIONS,
    INSTRUMENTS,
    SKILL_LEVELS,
    AudioClip,
    ComparisonResult,
    FeatureSet,
    FeedbackItem,
    PlayerContext,
    ProblemRegion,
)
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
INSTRUMENT_LABELS = {
    "viola": "Viola",
    "acoustic_guitar": "Acoustic guitar",
    "electric_guitar": "Electric guitar",
    "other": "Other",
}
DIMENSION_ICONS = {"pitch": "🎯", "dynamics": "🔊", "tone": "🎨"}
MAX_TIPS_SHOWN = 5

# The feedback "socket": swap in any other FeedbackProvider (e.g. an ML model
# in Phase 8) here, and nothing else in the app changes.
PROVIDER: FeedbackProvider = RuleBasedFeedback()


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


def certainty(confidence: float) -> str:
    """Confidence as a word; a percentage would suggest more precision than we have."""
    return "high" if confidence >= 0.75 else "medium" if confidence >= 0.5 else "low"


def open_store() -> SessionStore | None:
    """The history store, or None (with a warning) if it can't be opened."""
    try:
        return SessionStore(default_data_dir())
    except (OSError, sqlite3.Error) as e:
        st.warning(f"Your practice history can't be opened, so nothing will be saved: {e}")
        return None


# st.feedback("thumbs") reports 0 for 👎 and 1 for 👍; the store uses -1 / +1.
THUMB_TO_RATING: dict[int, Rating] = {0: -1, 1: 1}
RATING_TO_THUMB = {rating: thumb for thumb, rating in THUMB_TO_RATING.items()}


def save_rating(store: SessionStore, session_id: str, index: int, key: str) -> None:
    """Runs when a 👍/👎 is clicked, before the page reruns."""
    thumb = st.session_state[key]
    store.rate(session_id, index, None if thumb is None else THUMB_TO_RATING[thumb])


@dataclass(frozen=True)
class Rater:
    """Draws the 👍/👎 widgets for the tips of one saved session."""

    store: SessionStore
    session_id: str
    saved: dict[int, Rating]  # ratings given earlier, by tip position

    def widget(self, index: int) -> None:
        # The key combines session and position, so a rating can never land on
        # the wrong tip, and coming back to the same take shows earlier ratings.
        key = f"rating-{self.session_id}-{index}"
        saved = self.saved.get(index)
        st.feedback(
            "thumbs",
            key=key,
            default=None if saved is None else RATING_TO_THUMB[saved],
            on_change=save_rating,
            args=(self.store, self.session_id, index, key),
        )


def show_feedback_item(item: FeedbackItem, index: int, rater: Rater | None) -> None:
    with st.container(border=True):
        st.markdown(f"{DIMENSION_ICONS[item.dimension]} **{item.message}**")
        st.markdown(item.tip)
        st.caption(f"{item.dimension.title()} · certainty: {certainty(item.confidence)}")
        if rater is not None:
            rater.widget(index)


def show_feedback(items: list[FeedbackItem], rater: Rater | None) -> None:
    """The most useful advice first; the rest folded away so it doesn't overwhelm.

    For a saved session, each tip gets a 👍/👎: these ratings are the labels a
    future ML model learns from ("which tips do players find helpful?").
    """
    if not items:
        st.success("Nothing to work on: your take matches the reference closely.")
        return
    if rater is not None:
        st.caption("Was a tip helpful? Your 👍/👎 helps improve future advice.")
    for index, item in enumerate(items[:MAX_TIPS_SHOWN]):
        show_feedback_item(item, index, rater)
    if len(items) > MAX_TIPS_SHOWN:
        with st.expander(f"{len(items) - MAX_TIPS_SHOWN} more"):
            for index, item in enumerate(items[MAX_TIPS_SHOWN:], start=MAX_TIPS_SHOWN):
                show_feedback_item(item, index, rater)


def save_to_history(
    store: SessionStore,
    uploads: tuple[bytes, str, bytes, str],
    recordings: tuple[tuple[AudioClip, FeatureSet], tuple[AudioClip, FeatureSet]],
    result: ComparisonResult,
    context: PlayerContext,
    feedback: list[FeedbackItem],
    keep_audio: bool,
) -> str | None:
    """Save this comparison (once: reruns find the same session). None if it failed."""
    (ref_clip, ref_features), (user_clip, user_features) = recordings
    try:
        return store.save_session(
            fingerprint=fingerprint(uploads[0], uploads[2], context, PROVIDER.source),
            ref=ref_features,
            user=user_features,
            result=result,
            context=context,
            feedback=feedback,
            feedback_source=PROVIDER.source,
            ref_clip=ref_clip,
            user_clip=user_clip,
            keep_audio=keep_audio,
        )
    except (OSError, sqlite3.Error) as e:
        st.warning(f"Couldn't save this session to your history: {e}")
        return None


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

with st.sidebar:
    st.header("About you")
    context = PlayerContext(
        instrument=st.selectbox(
            "Instrument", INSTRUMENTS, format_func=INSTRUMENT_LABELS.__getitem__, key="instrument"
        ),
        skill_level=st.selectbox("Level", SKILL_LEVELS, index=1, format_func=str.title),
        piece=st.text_input("Piece (optional)"),
    )
    st.caption("Tips are written for your instrument.")

    st.header("Your history")
    save_history = st.toggle(
        "Save to my practice history",
        value=True,
        key="save_history",
        help="Saves the scores, measurements, and tips on this computer, so you can "
        "track your progress. Your recordings aren't saved unless you tick the box below.",
    )
    keep_audio = st.checkbox(
        "Also keep copies of the recordings",
        value=False,
        key="keep_audio",
        disabled=not save_history,
        help="Lets you listen back to old takes later. Off by default: recordings are "
        "personal and large.",
    )

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
    (ref_clip, ref_features), (user_clip, user_features) = analyzed["Reference"], analyzed["You"]
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

    feedback = PROVIDER.generate(ref_features, user_features, result, context)
    rater = None
    if save_history and (store := open_store()) is not None:
        recordings = (analyzed["Reference"], analyzed["You"])
        session_id = save_to_history(
            store, uploads, recordings, result, context, feedback, keep_audio
        )
        if session_id is not None:
            rater = Rater(store, session_id, store.ratings(session_id))

    st.subheader("What to work on")
    show_feedback(feedback, rater)

    st.subheader("Where it differs")
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
