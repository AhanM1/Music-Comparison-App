"""Small pieces shared by more than one page of the app."""

import sqlite3

import streamlit as st

from musiccompare.data.store import SessionStore, default_data_dir

INSTRUMENT_LABELS = {
    "viola": "Viola",
    "acoustic_guitar": "Acoustic guitar",
    "electric_guitar": "Electric guitar",
    "other": "Other",
}
DIMENSION_ICONS = {"pitch": "🎯", "dynamics": "🔊", "tone": "🎨"}

# What can go wrong when reading or writing the history: the disk (full, no
# permission, a file where the folder should be) or the database itself.
STORAGE_ERRORS = (OSError, sqlite3.Error)


def open_store() -> SessionStore | None:
    """The history store, or None (with a warning) if it can't be opened.

    Never lets a storage problem stop the page: the analysis still works, it
    just isn't saved.
    """
    try:
        return SessionStore(default_data_dir())
    except STORAGE_ERRORS as e:
        st.warning(f"Your practice history can't be opened, so nothing will be saved: {e}")
        return None
