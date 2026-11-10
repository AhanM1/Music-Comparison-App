"""Tests for the History page, run headlessly with AppTest.

Sessions are written straight into a throwaway store with tiny made-up
measurements, so these tests don't need to analyse any audio and run fast.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from musiccompare import config
from musiccompare.data.store import SessionStore
from musiccompare.feedback.rules import SOURCE
from musiccompare.models import (
    Alignment,
    AudioClip,
    ComparisonResult,
    Dimension,
    FeatureSet,
    FeedbackItem,
    FrameDifferences,
    Instrument,
    PlayerContext,
)

APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")
START = datetime(2026, 11, 1, 18, 0, tzinfo=UTC)
N = 4  # frames in the made-up measurements


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SessionStore:
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    return SessionStore(tmp_path)


def add_session(
    store: SessionStore,
    key: str,
    scores: dict[Dimension, float | None],
    day: int = 0,
    piece: str = "Scale",
    instrument: Instrument = "viola",
) -> str:
    """Save a session with the given scores and one tip, ``day`` days after START."""
    zeros = np.zeros(N)
    features = FeatureSet(
        times=np.arange(N) * 0.023,
        f0_hz=np.full(N, 440.0),
        voiced_prob=np.ones(N),
        rms_db=np.full(N, -20.0),
        centroid_hz=np.full(N, 1000.0),
        flatness=np.full(N, 0.01),
        mfcc=np.zeros((config.N_MFCC, N)),
        chroma=np.zeros((12, N)),
        extractor_version="1",
    )
    result = ComparisonResult(
        alignment=Alignment(path=np.stack([np.arange(N)] * 2, axis=1), cost=0.1, resolution=1),
        differences=FrameDifferences(
            np.arange(N) * 0.023, zeros, zeros, zeros, 0.0, 0, np.zeros(N, bool)
        ),
        scores=scores,
        regions=[],
    )
    clip = AudioClip(y=np.zeros(1000, dtype=np.float32), sr=config.SAMPLE_RATE, name="x.wav")
    tip = FeedbackItem("pitch", "You were sharp.", "Tune up.", 0.5, 0.9, SOURCE)
    return store.save_session(
        fingerprint=key,
        ref=features,
        user=features,
        result=result,
        context=PlayerContext(instrument=instrument, piece=piece),
        feedback=[tip],
        feedback_source=SOURCE,
        ref_clip=clip,
        user_clip=clip,
        now=START + timedelta(days=day),
    )


def scores(pitch: float, tone: float | None = 80.0) -> dict[Dimension, float | None]:
    return {"pitch": pitch, "dynamics": 70.0, "tone": tone}


def history_page() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    app.switch_page("history_page.py").run()
    return app


def pitch_column(app: AppTest) -> list[float]:
    return [row["Pitch"] for row in app.dataframe[0].value.to_dict("records")]


def test_empty_history_explains_how_to_start(store: SessionStore) -> None:
    app = history_page()
    assert not app.exception
    assert "No saved sessions yet" in app.info[0].value


def test_history_shows_latest_scores_and_progress(store: SessionStore) -> None:
    add_session(store, "a", {"pitch": 60.0, "dynamics": 70.0, "tone": None}, day=0)
    add_session(store, "b", {"pitch": 75.0, "dynamics": 65.0, "tone": 90.0}, day=1)

    app = history_page()

    assert not app.exception
    metrics = {m.label: (m.value, m.delta) for m in app.metric}
    assert metrics == {
        "Pitch": ("75/100", "+15"),
        "Dynamics": ("65/100", "-5"),
        "Tone": ("90/100", ""),  # no earlier tone score, so no change shown
    }
    assert len(app.get("image")) == 1  # the score chart
    assert pitch_column(app) == [75.0, 60.0]  # newest first
    assert app.dataframe[0].value.to_dict("records")[0]["Instrument"] == "Viola"
    assert app.subheader[1].value == "Sessions (2)"


def test_history_filters_by_piece_and_instrument(store: SessionStore) -> None:
    add_session(store, "a", scores(60.0), piece="Scale")
    add_session(store, "b", scores(90.0), day=1, piece="")
    add_session(store, "c", scores(50.0), day=2, instrument="electric_guitar")

    app = history_page()
    assert app.selectbox(key="piece_filter").options == ["All", "(no piece name)", "Scale"]

    app.selectbox(key="piece_filter").set_value("(no piece name)").run()
    assert pitch_column(app) == [90.0]

    app.selectbox(key="piece_filter").set_value("Scale").run()
    app.selectbox(key="instrument_filter").set_value("electric_guitar").run()
    assert pitch_column(app) == [50.0]
    assert app.dataframe[0].value.to_dict("records")[0]["Instrument"] == "Electric guitar"

    app.selectbox(key="piece_filter").set_value("(no piece name)").run()
    assert not app.exception
    assert "No saved sessions match" in app.info[0].value


def test_history_counts_ratings(store: SessionStore) -> None:
    first = add_session(store, "a", scores(60.0))
    second = add_session(store, "b", scores(60.0), day=1)

    assert any("haven't rated" in c.value for c in history_page().caption)

    store.rate(first, 0, 1)
    store.rate(second, 0, -1)
    app = history_page()
    assert any("rated **2** tips: 1 👍 and 1 👎" in m.value for m in app.markdown)


def test_deleting_history_needs_confirmation(store: SessionStore) -> None:
    add_session(store, "a", scores(60.0))
    app = history_page()

    assert app.button(key="delete_all").disabled  # one stray click can't delete it
    app.checkbox(key="confirm_delete").check().run()
    app.button(key="delete_all").click().run()

    assert not app.exception
    assert store.history() == []
    assert "No saved sessions yet" in app.info[0].value


def test_history_page_survives_an_unusable_data_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("a file where the data folder should be")
    monkeypatch.setenv(config.DATA_DIR_ENV, str(blocker))

    app = history_page()
    assert not app.exception
    assert "can't be opened" in app.warning[0].value


def test_deleting_resets_the_confirmation(store: SessionStore) -> None:
    """Otherwise the next deletion would need just one click."""
    add_session(store, "a", scores(60.0))
    app = history_page()
    app.checkbox(key="confirm_delete").check().run()
    app.button(key="delete_all").click().run()

    add_session(store, "b", scores(70.0))
    app.run()

    assert not app.checkbox(key="confirm_delete").value
    assert app.button(key="delete_all").disabled
