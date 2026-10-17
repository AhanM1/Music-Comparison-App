"""Smoke tests for the Streamlit app, run headlessly with Streamlit's AppTest.

These don't check layout details (the UI is meant to change freely). They check
that the script runs without crashing in the main situations a user hits.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# Build the path from this file's location so tests work from any directory.
APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")
TIMEOUT_S = 120  # first run imports librosa/numba, which is slow


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=TIMEOUT_S)


def test_app_starts_and_asks_for_uploads(app: AppTest) -> None:
    app.run()

    assert not app.exception
    assert any("Upload both" in info.value for info in app.info)


def test_app_shows_both_recordings(app: AppTest, make_tone, to_wav_bytes) -> None:
    app.run()
    app.file_uploader(key="ref").upload("ref.wav", to_wav_bytes(make_tone(440)), "audio/wav")
    app.file_uploader(key="user").upload("me.wav", to_wav_bytes(make_tone(450)), "audio/wav")
    app.run()

    assert not app.exception
    assert not app.error
    captions = " ".join(c.value for c in app.caption)
    assert "Reference: 1.0 s" in captions
    assert "You: 1.0 s" in captions


def test_app_reports_unreadable_upload(app: AppTest) -> None:
    app.run()
    app.file_uploader(key="ref").upload("broken.wav", b"not audio at all", "audio/wav")
    app.run()

    assert not app.exception  # handled gracefully...
    assert "broken.wav" in app.error[0].value  # ...with a message naming the file
