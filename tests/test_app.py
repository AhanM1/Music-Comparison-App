"""Smoke tests for the Streamlit app, run headlessly with Streamlit's AppTest.

These don't check layout details (the UI is meant to change freely). They check
that the script runs without crashing in the main situations a user hits.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from musiccompare import config
from musiccompare.data.store import SessionStore
from musiccompare.feedback.rules import SPOT_TIPS

# Build the path from this file's location so tests work from any directory.
APP = str(Path(__file__).parents[1] / "app" / "streamlit_app.py")
TIMEOUT_S = 120  # first run imports librosa/numba, which is slow


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every app test saves its history into a throwaway folder, never ./data."""
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture
def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=TIMEOUT_S)


def upload_sharp_take(app: AppTest, make_melody, to_wav_bytes) -> None:
    """Upload a reference and a take with notes 5-8 (1.6 s - 3.2 s) 35 cents sharp."""
    notes = [60, 64, 67, 72, 71, 67, 65, 69, 74, 72]
    sharp = [0, 0, 0, 0, 35, 35, 35, 35, 0, 0]
    app.file_uploader(key="ref").upload("ref.wav", to_wav_bytes(make_melody(notes)), "audio/wav")
    user = make_melody(notes, cents=sharp)
    app.file_uploader(key="user").upload("me.wav", to_wav_bytes(user), "audio/wav")
    app.run()


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


def test_app_compares_two_takes_of_the_same_melody(app: AppTest, make_melody, to_wav_bytes) -> None:
    notes = [60, 62, 64, 65, 67]
    app.run()
    app.file_uploader(key="ref").upload("ref.wav", to_wav_bytes(make_melody(notes)), "audio/wav")
    slower = make_melody(notes, durations=0.5)
    app.file_uploader(key="user").upload("me.wav", to_wav_bytes(slower), "audio/wav")
    app.run()

    assert not app.exception
    assert any(h.value == "Comparison" for h in app.header)
    assert not app.warning  # same tune: no "different music" warning
    captions = " ".join(c.value for c in app.caption)
    assert "+25% length" in captions  # 0.5 s notes vs 0.4 s
    # Only the tempo differs, so every score is perfect and nothing is flagged.
    assert {m.label: m.value for m in app.metric} == {
        "Pitch": "100/100",
        "Dynamics": "100/100",
        "Tone": "100/100",
    }
    successes = [s.value for s in app.success]
    assert any("Nothing to work on" in s for s in successes)
    assert any("No problem spots" in s for s in successes)


def test_app_gives_advice_for_the_chosen_instrument(
    app: AppTest, make_melody, to_wav_bytes
) -> None:
    notes = [60, 64, 67, 72, 71, 67, 65, 69, 74, 72]
    sharp = [0, 0, 0, 0, 35, 35, 35, 35, 0, 0]  # notes 5-8 (1.6 s - 3.2 s) 35 cents sharp
    app.run()
    app.file_uploader(key="ref").upload("ref.wav", to_wav_bytes(make_melody(notes)), "audio/wav")
    user = make_melody(notes, cents=sharp)
    app.file_uploader(key="user").upload("me.wav", to_wav_bytes(user), "audio/wav")
    app.run()

    def advice() -> list[str]:
        return [m.value for m in app.markdown]

    assert not app.exception
    assert app.selectbox(key="instrument").value == "viola"
    assert "🎯 **At 0:01.6 – 0:03.2 you were 35 cents sharp.**" in advice()
    assert SPOT_TIPS[("pitch", True)]["bowed"] in advice()

    app.selectbox(key="instrument").set_value("acoustic_guitar").run()

    assert not app.exception
    assert SPOT_TIPS[("pitch", True)]["guitar"] in advice()
    assert SPOT_TIPS[("pitch", True)]["bowed"] not in advice()


def test_app_folds_away_extra_advice(app: AppTest, make_melody, to_wav_bytes) -> None:
    notes = [60, 64, 67, 72, 71, 67, 65, 69, 74, 72, 67, 64, 60, 65]
    cents = [40, 0, -40, 0] * 3 + [40, 0]  # 7 separate sharp/flat notes
    app.run()
    app.file_uploader(key="ref").upload("ref.wav", to_wav_bytes(make_melody(notes)), "audio/wav")
    user = make_melody(notes, cents=cents)
    app.file_uploader(key="user").upload("me.wav", to_wav_bytes(user), "audio/wav")
    app.run()

    assert not app.exception
    # 7 tips: the app shows 5 (MAX_TIPS_SHOWN) and folds the rest away.
    assert app.expander[0].label == "2 more"


def test_app_lists_a_sharp_passage(app: AppTest, make_melody, to_wav_bytes) -> None:
    notes = [60, 64, 67, 72, 71, 67, 65, 69, 74, 72]
    sharp = [0, 0, 0, 0, 35, 35, 35, 35, 0, 0]  # notes 5-8 (1.6 s - 3.2 s) 35 cents sharp
    app.run()
    app.file_uploader(key="ref").upload("ref.wav", to_wav_bytes(make_melody(notes)), "audio/wav")
    user = make_melody(notes, cents=sharp)
    app.file_uploader(key="user").upload("me.wav", to_wav_bytes(user), "audio/wav")
    app.run()

    assert not app.exception
    [row] = app.dataframe[0].value.to_dict("records")
    assert row["What"] == "Pitch"
    assert row["How far off"] == "35 cents sharp"
    assert row["Reference time"] == "0:01.6 – 0:03.2"
    # The comparison charts are drawn: timeline + 5 tabs + 2x3 per-recording plots.
    assert len(app.get("image")) == 1 + 5 + 6
    # Two full players + one snippet player per recording.
    assert len(app.get("audio")) == 4
    assert app.selectbox(key="snippet").value.dimension == "pitch"


def test_app_plays_the_selected_problem_spot(app: AppTest, make_melody, to_wav_bytes) -> None:
    notes = [60, 64, 67, 72, 71, 67, 65, 69, 74, 72]
    cents = [0, 40, 40, 0, 0, 0, 0, -40, -40, 0]  # sharp at 0.4-1.2 s, flat at 2.8-3.6 s
    app.run()
    app.file_uploader(key="ref").upload("ref.wav", to_wav_bytes(make_melody(notes)), "audio/wav")
    user = make_melody(notes, cents=cents)
    app.file_uploader(key="user").upload("me.wav", to_wav_bytes(user), "audio/wav")
    app.run()

    def snippet_times() -> list[str]:
        return [m.value for m in app.markdown if m.value.startswith(("**Reference**", "**You**"))]

    assert snippet_times()[0] == "**Reference** (0:00.4 – 0:01.2)"
    app.selectbox(key="snippet").select_index(1).run()

    assert not app.exception
    assert snippet_times()[0] == "**Reference** (0:02.8 – 0:03.6)"


def test_app_warns_when_pieces_differ(app: AppTest, make_melody, to_wav_bytes) -> None:
    app.run()
    ref = make_melody([60, 64, 67, 72, 67, 64])
    other = make_melody([61, 70, 63, 66, 59, 68])
    app.file_uploader(key="ref").upload("ref.wav", to_wav_bytes(ref), "audio/wav")
    app.file_uploader(key="user").upload("other.wav", to_wav_bytes(other), "audio/wav")
    app.run()

    assert not app.exception
    assert "same music" in app.warning[0].value


def test_app_reports_unreadable_upload(app: AppTest) -> None:
    app.run()
    app.file_uploader(key="ref").upload("broken.wav", b"not audio at all", "audio/wav")
    app.run()

    assert not app.exception  # handled gracefully...
    assert "broken.wav" in app.error[0].value  # ...with a message naming the file
    assert "[in#" not in app.error[0].value  # no FFmpeg internals in the message


def test_app_rejects_a_too_short_recording(app: AppTest, make_tone, to_wav_bytes) -> None:
    app.run()
    app.file_uploader(key="ref").upload("click.wav", to_wav_bytes(make_tone(duration_s=0.1)))
    app.run()

    assert not app.exception
    assert "at least 1 s" in app.error[0].value


# --- Practice history (Phase 7) ----------------------------------------------


def test_app_saves_the_session_once(
    app: AppTest, make_melody, to_wav_bytes, data_dir: Path
) -> None:
    app.run()
    upload_sharp_take(app, make_melody, to_wav_bytes)
    app.run()  # a rerun (any click) must not save the take a second time

    assert not app.exception
    [session] = SessionStore(data_dir).history()
    assert session.instrument == "viola"
    assert session.n_tips == len(app.feedback) > 0
    assert not session.audio_saved


def test_app_stores_tip_ratings(app: AppTest, make_melody, to_wav_bytes, data_dir: Path) -> None:
    app.run()
    upload_sharp_take(app, make_melody, to_wav_bytes)
    store = SessionStore(data_dir)
    [session] = store.history()

    app.feedback[0].set_value(1).run()  # 👍
    assert not app.exception
    assert store.ratings(session.id) == {0: 1}

    app.feedback[0].set_value(0).run()  # changed to 👎
    assert store.ratings(session.id) == {0: -1}


def test_app_shows_earlier_ratings_again(make_melody, to_wav_bytes, data_dir: Path) -> None:
    first = AppTest.from_file(APP, default_timeout=TIMEOUT_S)
    first.run()
    upload_sharp_take(first, make_melody, to_wav_bytes)
    first.feedback[0].set_value(1).run()

    # A fresh browser session with the same take: the 👍 is still there.
    later = AppTest.from_file(APP, default_timeout=TIMEOUT_S)
    later.run()
    upload_sharp_take(later, make_melody, to_wav_bytes)

    assert not later.exception
    assert later.feedback[0].value == 1
    assert len(SessionStore(data_dir).history()) == 1


def test_app_saves_nothing_when_history_is_off(
    app: AppTest, make_melody, to_wav_bytes, data_dir: Path
) -> None:
    app.run()
    app.toggle(key="save_history").set_value(False)
    upload_sharp_take(app, make_melody, to_wav_bytes)

    assert not app.exception
    assert app.checkbox(key="keep_audio").disabled
    assert not app.feedback  # nothing to rate without a saved session
    assert any("35 cents sharp" in m.value for m in app.markdown)  # advice still shown
    assert not (data_dir / "sessions.db").exists()


def test_app_keeps_recordings_only_when_asked(
    app: AppTest, make_melody, to_wav_bytes, data_dir: Path
) -> None:
    app.run()
    app.checkbox(key="keep_audio").check()
    upload_sharp_take(app, make_melody, to_wav_bytes)

    assert not app.exception
    store = SessionStore(data_dir)
    [session] = store.history()
    assert session.audio_saved
    assert store.audio_paths(session.id) is not None


def test_app_still_works_when_history_cant_be_saved(
    app: AppTest, make_melody, to_wav_bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("a file where the data folder should be")
    monkeypatch.setenv(config.DATA_DIR_ENV, str(blocker))
    app.run()
    upload_sharp_take(app, make_melody, to_wav_bytes)

    assert not app.exception
    assert any("history can't be opened" in w.value for w in app.warning)
    assert any("35 cents sharp" in m.value for m in app.markdown)
    assert not app.feedback


def test_app_warns_about_a_busy_reference(app: AppTest, make_melody, to_wav_bytes) -> None:
    import numpy as np

    from musiccompare import config as cfg

    notes = [60, 64, 67, 72, 71, 67, 65, 69, 74, 72]
    melody = make_melody(notes)
    t = np.arange(melody.size) / cfg.SAMPLE_RATE
    drone = 0.25 * np.sin(2 * np.pi * 110 * t) + 0.15 * np.sin(2 * np.pi * 165 * t)
    app.run()
    reference = (melody + drone).astype(np.float32)
    app.file_uploader(key="ref").upload("orchestra.wav", to_wav_bytes(reference), "audio/wav")
    app.file_uploader(key="user").upload("me.wav", to_wav_bytes(melody), "audio/wav")
    app.run()

    assert not app.exception
    assert any("accompaniment, a drone, or a lot of reverb" in w.value for w in app.warning)
