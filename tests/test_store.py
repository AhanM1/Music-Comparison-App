"""Tests for the session store: saving, re-saving, ratings, history, and deleting.

Every test gets its own empty folder (pytest's ``tmp_path``), so tests never
touch your real practice history or each other's data.
"""

import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from musiccompare import config
from musiccompare.analysis.alignment import align
from musiccompare.analysis.compare import compare
from musiccompare.data.store import SCHEMA_VERSION, SessionStore, default_data_dir, fingerprint
from musiccompare.features.extract import extract_features
from musiccompare.feedback.rules import SOURCE, RuleBasedFeedback
from musiccompare.models import (
    AudioClip,
    ComparisonResult,
    FeatureSet,
    FeedbackItem,
    PlayerContext,
)

SR = config.SAMPLE_RATE
VIOLA = PlayerContext(instrument="viola", piece="Scale")


def melody(cents: list[float]) -> np.ndarray:
    """Eight 0.4 s notes with overtones; ``cents`` detunes each note."""
    notes = []
    for midi, off in zip([60, 64, 67, 72, 71, 67, 65, 69], cents, strict=True):
        t = np.arange(int(0.4 * SR)) / SR
        f0 = 440.0 * 2 ** ((midi - 69) / 12 + off / 1200)
        tone = sum(a * np.sin(2 * np.pi * f0 * (h + 1) * t) for h, a in enumerate((1, 0.5, 0.3)))
        notes.append(0.3 * tone * np.minimum(1, t / 0.02) * np.minimum(1, (0.4 - t) / 0.03))
    return np.concatenate(notes).astype(np.float32)


@dataclass(frozen=True)
class Analysis:
    ref_clip: AudioClip
    user_clip: AudioClip
    ref: FeatureSet
    user: FeatureSet
    result: ComparisonResult
    feedback: list[FeedbackItem]


@pytest.fixture(scope="module")
def analysis() -> Analysis:
    """One real comparison (notes 3-6 are 35 cents sharp), computed once per module."""
    ref_clip = AudioClip(y=melody([0] * 8), sr=SR, name="ref.wav")
    user_clip = AudioClip(y=melody([0, 0, 35, 35, 35, 35, 0, 0]), sr=SR, name="me.wav")
    ref, user = extract_features(ref_clip), extract_features(user_clip)
    result = compare(ref, user, align(ref, user))
    feedback = RuleBasedFeedback().generate(ref, user, result, VIOLA)
    assert result.regions and feedback  # the tests below need something to store
    return Analysis(ref_clip, user_clip, ref, user, result, feedback)


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path)


def save(
    store: SessionStore,
    a: Analysis,
    key: str = "take-1",
    context: PlayerContext = VIOLA,
    **kwargs: object,
) -> str:
    return store.save_session(
        fingerprint=key,
        ref=a.ref,
        user=a.user,
        result=a.result,
        context=context,
        feedback=a.feedback,
        feedback_source=SOURCE,
        ref_clip=a.ref_clip,
        user_clip=a.user_clip,
        **kwargs,  # type: ignore[arg-type]
    )


# --- Fingerprints ------------------------------------------------------------


def test_fingerprint_changes_with_what_changes_the_advice() -> None:
    base = fingerprint(b"ref", b"me", VIOLA, SOURCE)
    assert base == fingerprint(b"ref", b"me", VIOLA, SOURCE)  # deterministic
    assert base != fingerprint(b"ref", b"me2", VIOLA, SOURCE)
    assert base != fingerprint(b"me", b"ref", VIOLA, SOURCE)  # order matters
    assert base != fingerprint(b"ref", b"me", replace(VIOLA, instrument="other"), SOURCE)
    assert base != fingerprint(b"ref", b"me", replace(VIOLA, skill_level="advanced"), SOURCE)
    assert base != fingerprint(b"ref", b"me", VIOLA, "ml-v1")
    # The piece is only a label: renaming it is the same analysis.
    assert base == fingerprint(b"ref", b"me", replace(VIOLA, piece="Etude"), SOURCE)


def test_fingerprint_has_no_ambiguous_seams() -> None:
    """Joining raw bytes would make ("ab", "c") and ("a", "bc") collide."""
    assert fingerprint(b"ab", b"c", VIOLA, SOURCE) != fingerprint(b"a", b"bc", VIOLA, SOURCE)


def test_default_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(config.DATA_DIR_ENV, raising=False)
    assert default_data_dir() == Path("data")
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    assert default_data_dir() == tmp_path


# --- Saving ------------------------------------------------------------------


def test_a_new_store_is_empty(store: SessionStore) -> None:
    assert store.history() == []
    assert store.rated_tips() == []
    assert (store.root / "sessions.db").exists()


def test_saved_session_appears_in_history(store: SessionStore, analysis: Analysis) -> None:
    when = datetime(2026, 11, 10, 18, 30, tzinfo=UTC)
    session_id = save(store, analysis, now=when)

    [session] = store.history()
    assert session.id == session_id
    assert session.created_at == when
    assert (session.instrument, session.skill_level, session.piece) == (
        "viola",
        "intermediate",
        "Scale",
    )
    assert (session.ref_name, session.user_name) == ("ref.wav", "me.wav")
    assert session.scores == pytest.approx(analysis.result.scores)
    assert session.n_tips == len(analysis.feedback)
    assert not session.audio_saved


def test_session_row_records_versions(store: SessionStore, analysis: Analysis) -> None:
    session_id = save(store, analysis)
    with sqlite3.connect(store.root / "sessions.db") as db:
        row = db.execute(
            "SELECT schema_version, extractor_version, feedback_source, alignment_cost "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    assert row == (
        SCHEMA_VERSION,
        analysis.ref.extractor_version,
        SOURCE,
        pytest.approx(analysis.result.alignment.cost),
    )


def test_saving_again_is_the_same_session(store: SessionStore, analysis: Analysis) -> None:
    """Streamlit reruns the script on every click; that must not duplicate sessions."""
    first = save(store, analysis)
    again = save(store, analysis, context=replace(VIOLA, piece="Scale in C"))

    assert again == first
    [session] = store.history()
    assert session.piece == "Scale in C"  # the label is updated in place
    assert len(list(store.arrays_dir.iterdir())) == 1


def test_different_takes_are_different_sessions(store: SessionStore, analysis: Analysis) -> None:
    now = datetime(2026, 11, 10, tzinfo=UTC)
    second = save(store, analysis, key="take-2", now=now + timedelta(days=1))
    first = save(store, analysis, key="take-1", now=now)
    assert [s.id for s in store.history()] == [first, second]  # oldest first


def test_session_with_no_tips_still_records_its_source(
    store: SessionStore, analysis: Analysis
) -> None:
    session_id = save(store, replace(analysis, feedback=[]))
    [session] = store.history()
    assert session.n_tips == 0
    with sqlite3.connect(store.root / "sessions.db") as db:
        [(source,)] = db.execute("SELECT feedback_source FROM sessions WHERE id = ?", (session_id,))
    assert source == SOURCE


def test_scores_with_nothing_to_compare_are_stored_as_missing(
    store: SessionStore, analysis: Analysis
) -> None:
    scores = {**analysis.result.scores, "pitch": None}
    save(store, replace(analysis, result=replace(analysis.result, scores=scores)))
    [session] = store.history()
    assert session.scores["pitch"] is None


def test_saved_arrays_load_back_unchanged(store: SessionStore, analysis: Analysis) -> None:
    session_id = save(store, analysis)
    saved = store.load_arrays(session_id)

    for loaded, original in ((saved.ref, analysis.ref), (saved.user, analysis.user)):
        assert loaded.extractor_version == original.extractor_version
        for name in ("times", "f0_hz", "rms_db", "flatness", "mfcc", "chroma"):
            np.testing.assert_array_equal(getattr(loaded, name), getattr(original, name))
    assert saved.alignment == replace(analysis.result.alignment, path=saved.alignment.path)
    np.testing.assert_array_equal(saved.alignment.path, analysis.result.alignment.path)
    diff = analysis.result.differences
    np.testing.assert_array_equal(saved.differences.pitch_cents, diff.pitch_cents)
    assert saved.differences.level_offset_db == pytest.approx(diff.level_offset_db)
    assert saved.differences.octave_shift == diff.octave_shift


def test_no_partial_files_are_left_behind(store: SessionStore, analysis: Analysis) -> None:
    save(store, analysis)
    assert not list(store.arrays_dir.glob("*.partial*"))


# --- Recordings (opt-in) -----------------------------------------------------


def test_recordings_are_not_kept_by_default(store: SessionStore, analysis: Analysis) -> None:
    session_id = save(store, analysis)
    assert store.audio_paths(session_id) is None
    assert not store.audio_dir.exists()


def test_recordings_are_kept_when_asked(store: SessionStore, analysis: Analysis) -> None:
    session_id = save(store, analysis, keep_audio=True)
    paths = store.audio_paths(session_id)
    assert paths is not None
    assert all(p.stat().st_size > 0 for p in paths)
    [session] = store.history()
    assert session.audio_saved


def test_agreeing_later_keeps_the_recordings(store: SessionStore, analysis: Analysis) -> None:
    session_id = save(store, analysis)
    save(store, analysis, keep_audio=True)  # same take, consent given afterwards
    assert store.audio_paths(session_id) is not None
    assert store.history()[0].audio_saved


# --- Ratings -----------------------------------------------------------------


def test_rating_a_tip(store: SessionStore, analysis: Analysis) -> None:
    session_id = save(store, analysis)
    store.rate(session_id, 0, 1)
    assert store.ratings(session_id) == {0: 1}

    store.rate(session_id, 0, -1)  # change of mind
    assert store.ratings(session_id) == {0: -1}

    store.rate(session_id, 0, None)  # cleared
    assert store.ratings(session_id) == {}


def test_rating_a_missing_tip_fails_loudly(store: SessionStore, analysis: Analysis) -> None:
    session_id = save(store, analysis)
    with pytest.raises(KeyError):
        store.rate(session_id, len(analysis.feedback), 1)
    with pytest.raises(KeyError):
        store.rate("no-such-session", 0, 1)


def test_rated_tips_are_training_examples(store: SessionStore, analysis: Analysis) -> None:
    session_id = save(store, analysis)
    store.rate(session_id, 0, 1)

    [example] = store.rated_tips()
    tip = analysis.feedback[0]
    assert example.session_id == session_id
    assert (example.instrument, example.skill_level) == ("viola", "intermediate")
    assert (example.dimension, example.message, example.tip) == (
        tip.dimension,
        tip.message,
        tip.tip,
    )
    assert example.importance == pytest.approx(tip.importance)
    assert example.confidence == pytest.approx(tip.confidence)
    assert example.source == SOURCE
    assert example.rating == 1
    if tip.region is None:
        assert example.region_duration_s is None
        assert example.region_severity is None
    else:
        assert example.region_duration_s == pytest.approx(tip.region.duration_s)
        assert example.region_severity == pytest.approx(tip.region.severity)


def test_habit_and_spot_tips_store_their_regions(store: SessionStore, analysis: Analysis) -> None:
    habit = replace(analysis.feedback[0], region=None)
    spot = replace(analysis.feedback[0], region=analysis.result.regions[0])
    session_id = save(store, replace(analysis, feedback=[habit, spot]))
    store.rate(session_id, 0, -1)
    store.rate(session_id, 1, 1)

    by_rating = {t.rating: t for t in store.rated_tips()}
    assert by_rating[-1].region_duration_s is None
    assert by_rating[1].region_duration_s == pytest.approx(analysis.result.regions[0].duration_s)


# --- Deleting ----------------------------------------------------------------


def test_deleting_a_session_removes_everything(store: SessionStore, analysis: Analysis) -> None:
    kept = save(store, analysis, key="keep")
    gone = save(store, analysis, key="gone", keep_audio=True)
    store.rate(gone, 0, 1)

    store.delete_session(gone)

    assert [s.id for s in store.history()] == [kept]
    assert store.rated_tips() == []
    assert store.audio_paths(gone) is None
    assert not (store.arrays_dir / f"{gone}.npz").exists()
    with sqlite3.connect(store.root / "sessions.db") as db:
        # Cascade: the tips and regions went with the session.
        for table in ("feedback", "regions"):
            [(n,)] = db.execute(f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", (gone,))
            assert n == 0


def test_delete_all(store: SessionStore, analysis: Analysis) -> None:
    save(store, analysis, key="a")
    save(store, analysis, key="b", keep_audio=True)
    store.delete_all()
    assert store.history() == []
    assert not list(store.arrays_dir.iterdir())
    assert not list(store.audio_dir.iterdir())


def test_a_failed_save_leaves_nothing_behind(
    store: SessionStore, analysis: Analysis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The database transaction rolls back, and the files written first are removed."""

    def broken_insert(*args: object) -> None:
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(store, "_insert", broken_insert)
    with pytest.raises(sqlite3.OperationalError):
        save(store, analysis, keep_audio=True)

    assert store.history() == []
    assert not list(store.arrays_dir.iterdir())
    assert not list(store.audio_dir.iterdir())


def test_reopening_a_store_keeps_its_data(analysis: Analysis, tmp_path: Path) -> None:
    session_id = save(SessionStore(tmp_path), analysis)
    assert [s.id for s in SessionStore(tmp_path).history()] == [session_id]
