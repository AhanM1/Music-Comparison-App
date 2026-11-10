"""The History page: your saved sessions, how your scores changed, and your ratings.

Everything here is read from the session store that the Compare page writes
to (see musiccompare/data/store.py). Times are stored in UTC, the same
everywhere in the world, and shown here in this computer's local time.
"""

from collections.abc import Mapping, Sequence

import streamlit as st

from components import INSTRUMENT_LABELS, open_store
from musiccompare.data.store import SessionSummary
from musiccompare.models import DIMENSIONS, Dimension
from musiccompare.viz.history_plot import plot_score_history
from musiccompare.viz.style import to_png

ALL = "All"
UNTITLED = "(no piece name)"


def piece_label(session: SessionSummary) -> str:
    return session.piece or UNTITLED


def when(session: SessionSummary) -> str:
    """e.g. "Nov 10, 18:30" in local time."""
    return session.created_at.astimezone().strftime("%b %d, %H:%M")


def score_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}/100"


@st.cache_data(show_spinner=False)
def score_chart(scores: Sequence[Mapping[Dimension, float | None]], labels: list[str]) -> bytes:
    return to_png(plot_score_history(scores, labels))


st.title("📈 Practice history")

store = open_store()
if store is None:
    st.stop()  # open_store already showed why
sessions = store.history()
if not sessions:
    st.info("No saved sessions yet. Compare a take on the Compare page and it will show up here.")
    st.stop()

# --- Filters -----------------------------------------------------------------
col_piece, col_instrument = st.columns(2)
piece = col_piece.selectbox(
    "Piece", [ALL, *sorted({piece_label(s) for s in sessions})], key="piece_filter"
)
instrument = col_instrument.selectbox(
    "Instrument",
    [ALL, *sorted({s.instrument for s in sessions})],
    format_func=lambda i: INSTRUMENT_LABELS.get(i, i),
    key="instrument_filter",
)
shown = [
    s for s in sessions if piece in (ALL, piece_label(s)) and instrument in (ALL, s.instrument)
]
if not shown:
    st.info("No saved sessions match these filters.")
    st.stop()

# --- Latest scores, with the change since the session before -----------------
st.subheader("Latest session")
latest = shown[-1]
previous = shown[-2] if len(shown) > 1 else None
for column, dimension in zip(st.columns(len(DIMENSIONS)), DIMENSIONS, strict=True):
    now, before = latest.scores[dimension], previous and previous.scores[dimension]
    change = None if now is None or before is None else f"{now - before:+.0f}"
    column.metric(dimension.title(), score_text(now), change)
st.caption(
    f"{when(latest)} · {piece_label(latest)}"
    + ("" if previous is None else " · change since the session before")
)

st.image(score_chart([s.scores for s in shown], [when(s) for s in shown]), width="stretch")

# --- Every session, newest first ---------------------------------------------
st.subheader(f"Sessions ({len(shown)})")
st.dataframe(
    [
        {
            "When": when(s),
            "Piece": piece_label(s),
            "Instrument": INSTRUMENT_LABELS.get(s.instrument, s.instrument),
            **{d.title(): s.scores[d] for d in DIMENSIONS},
            "Tips": s.n_tips,
            "Recordings kept": s.audio_saved,
        }
        for s in reversed(shown)
    ],
    hide_index=True,
    column_config={d.title(): st.column_config.NumberColumn(format="%.0f") for d in DIMENSIONS},
)

# --- Ratings -----------------------------------------------------------------
st.subheader("Your ratings")
shown_ids = {s.id for s in shown}
rated = [t for t in store.rated_tips() if t.session_id in shown_ids]
if not rated:
    st.caption("You haven't rated any tips yet. Use 👍/👎 under a tip on the Compare page.")
else:
    helpful = sum(t.rating == 1 for t in rated)
    st.markdown(
        f"You rated **{len(rated)}** tips: {helpful} 👍 and {len(rated) - helpful} 👎. "
        "These ratings are what a future model will learn from."
    )

# --- Deleting ----------------------------------------------------------------
with st.expander("Delete history"):
    st.caption(
        "Removes every saved session (scores, measurements, tips, ratings, and any "
        "kept recordings) from this computer. This can't be undone."
    )
    confirm = st.checkbox("Yes, delete all of my history", key="confirm_delete")
    if st.button("Delete history", type="primary", disabled=not confirm, key="delete_all"):
        store.delete_all()
        st.rerun()
