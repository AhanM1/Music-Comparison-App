"""Streamlit entry point.  Run with:  uv run streamlit run app/streamlit_app.py

This file is only the "router": it sets up the browser tab and lists the pages.
``st.navigation`` draws the page menu in the sidebar and returns the page the
user picked; ``.run()`` runs that page's script. Each page is its own file:

- compare_page.py: upload two recordings and get the comparison and tips
- history_page.py: your saved sessions and how your scores changed

Streamlit puts this folder on Python's import path, so pages can share code
with ``from components import ...``.
"""

import streamlit as st

st.set_page_config(page_title="Music Comparison", page_icon="🎻", layout="wide")
st.navigation(
    [
        st.Page("compare_page.py", title="Compare", icon="🎻", default=True),
        st.Page("history_page.py", title="History", icon="📈"),
    ]
).run()
