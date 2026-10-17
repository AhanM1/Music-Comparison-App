"""musiccompare: compare a recording to a reference on pitch, dynamics, and tone.

This package is the app's "brain". It must never import Streamlit; the UI in app/
calls into it, not the other way around (see PLAN.md section 2.1).
"""

__version__ = "0.1.0"
