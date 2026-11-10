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
from musiccompare.analysis.compare import COMPARISON_VERSION, compare
from musiccompare.data import store as store_module
from musiccompare.data.store import SCHEMA_VERSION, SessionStore, default_data_dir, fingerprint
from musiccompare.features.extract import EXTRACTOR_VERSION as EXTRACTOR_VERSION
from musiccompare.features.extract import extract_features
from musiccompare.feedback.rules import SOURCE, RuleBasedFeedback
from musiccompare.models import (
    AudioClip,
    ComparisonResult,
    FeatureSet,
    FeedbackItem,
    PlayerContext,
)
from musiccompare.pipeline import ANALYSIS_VERSION

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


def test_fingerprint_is_the_recordings_and_the_analysis_version() -> None:
    base = fingerprint(b"ref", b"me", ANALYSIS_VERSION)
    assert base == fingerprint(b"ref", b"me", ANALYSIS_VERSION)  # deterministic
    assert base != fingerprint(b"ref", b"me2", ANALYSIS_VERSION)
    assert base != fingerprint(b"me", b"ref", ANALYSIS_VERSION)  # order matters
    # New analysis code gives new numbers: a different session, never stale scores.
    assert base != fingerprint(b"ref", b"me", ANALYSIS_VERSION + "-next")


def test_analysis_version_names_the_feature_and_comparison_code() -> None:
    assert EXTRACTOR_VERSION in ANALYSIS_VERSION
    assert COMPARISON_VERSION in ANALYSIS_VERSION


def test_fingerprint_has_no_ambiguous_seams() -> None:
    """Joining raw bytes would make ("ab", "c") and ("a", "bc") collide."""
    assert fingerprint(b"ab", b"c", "v") != fingerprint(b"a", b"bc", "v")


def test_default_data_dir_is_the_project_folder_wherever_you_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Starting Streamlit from another folder used to start a second, empty history."""
    monkeypatch.delenv(config.DATA_DIR_ENV, raising=False)
    project = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    assert default_data_dir() == project / "data"
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "elsewhere"))
    assert default_data_dir() == tmp_path / "elsewhere"


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
    alignment = analysis.result.alignment
    assert (saved.alignment.cost, saved.alignment.resolution) == (
        pytest.approx(alignment.cost),
        alignment.resolution,
    )
    for name in ("path", "extra_user", "skipped_ref"):
        np.testing.assert_array_equal(getattr(saved.alignment, name), getattr(alignment, name))
    diff = analysis.result.differences
    np.testing.assert_array_equal(saved.differences.pitch_cents, diff.pitch_cents)
    np.testing.assert_array_equal(saved.differences.missing, diff.missing)
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
    when = datetime(2026, 11, 10, 18, 30, tzinfo=UTC)
    session_id = save(store, analysis, now=when)
    store.rate(session_id, 0, 1)

    [example] = store.rated_tips()
    tip = analysis.feedback[0]
    assert example.session_id == session_id
    assert example.created_at == when
    assert (example.instrument, example.skill_level) == ("viola", "intermediate")
    assert example.piece == "Scale"
    assert example.extractor_version == analysis.ref.extractor_version
    assert example.scores == pytest.approx(analysis.result.scores)
    assert example.position == 0
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
        assert example.region_mean_error is None
        assert example.region_severity is None
    else:
        assert example.region_duration_s == pytest.approx(tip.region.duration_s)
        assert example.region_mean_error == pytest.approx(tip.region.mean_error)
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


def test_arrays_saved_before_coverage_tracking_still_load(
    store: SessionStore, analysis: Analysis
) -> None:
    """Old NPZ files have no missing/extra/skipped arrays; they load as "nothing missing"."""
    session_id = save(store, analysis)
    path = store.arrays_dir / f"{session_id}.npz"
    with np.load(path) as npz:
        old = {k: npz[k] for k in npz.files if k not in NEW_KEYS}
    np.savez_compressed(path, **old)

    saved = store.load_arrays(session_id)
    assert not saved.differences.missing.any()
    assert saved.alignment.extra_user.shape == saved.alignment.skipped_ref.shape == (0, 2)


NEW_KEYS = {"diff_missing", "alignment_extra_user", "alignment_skipped_ref"}


# --- Session identity: details change, the session stays -------------------------


def test_changing_instrument_updates_the_session_instead_of_duplicating_it(
    store: SessionStore, analysis: Analysis
) -> None:
    """Picking the right instrument after uploading used to add a second history point."""
    first = save(store, analysis)
    guitar = replace(VIOLA, instrument="acoustic_guitar", skill_level="advanced")
    new_tips = [replace(t, tip=f"guitar: {t.tip}") for t in analysis.feedback]

    again = save(store, replace(analysis, feedback=new_tips), context=guitar)

    assert again == first
    [session] = store.history()
    assert (session.instrument, session.skill_level) == ("acoustic_guitar", "advanced")
    with sqlite3.connect(store.root / "sessions.db") as db:
        tips = [t for (t,) in db.execute("SELECT tip FROM feedback ORDER BY idx")]
    assert tips == [t.tip for t in new_tips]


def test_ratings_survive_a_change_only_where_the_tip_is_the_same(
    store: SessionStore, analysis: Analysis
) -> None:
    habit = replace(analysis.feedback[0], message="Same habit.", tip="Same tip.", region=None)
    spot = replace(analysis.feedback[0], message="A spot.", tip="Viola tip.")
    session_id = save(store, replace(analysis, feedback=[habit, spot]))
    store.rate(session_id, 0, 1)
    store.rate(session_id, 1, -1)

    guitar_spot = replace(spot, tip="Guitar tip.")
    save(
        store,
        replace(analysis, feedback=[guitar_spot, habit]),  # new order, one tip changed
        context=replace(VIOLA, instrument="electric_guitar"),
    )

    # The unchanged habit keeps its 👍 at its new position; the changed tip starts fresh.
    assert store.ratings(session_id) == {1: 1}


def test_unchanged_settings_keep_tips_and_ratings_untouched(
    store: SessionStore, analysis: Analysis
) -> None:
    session_id = save(store, analysis)
    store.rate(session_id, 0, 1)
    save(store, analysis)  # a plain Streamlit rerun
    assert store.ratings(session_id) == {0: 1}


def test_withdrawing_consent_deletes_the_recordings(
    store: SessionStore, analysis: Analysis
) -> None:
    session_id = save(store, analysis, keep_audio=True)
    assert store.audio_paths(session_id) is not None

    save(store, analysis, keep_audio=False)  # the box was unticked

    assert store.audio_paths(session_id) is None
    assert not store.history()[0].audio_saved


# --- Concurrency (QA report M4) ------------------------------------------------


def test_simultaneous_saves_of_one_take_make_one_session(
    store: SessionStore, analysis: Analysis
) -> None:
    """Two browser tabs saving at once used to raise IntegrityError in all but one."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=6) as pool:
        ids = list(pool.map(lambda _: save(store, analysis), range(6)))

    assert len(set(ids)) == 1
    assert len(store.history()) == 1
    assert len(list(store.arrays_dir.iterdir())) == 1  # no orphaned files


# --- Schema versions and migrations ----------------------------------------------


def schema_version(store: SessionStore) -> int:
    with sqlite3.connect(store.root / "sessions.db") as db:
        [(version,)] = db.execute("PRAGMA user_version")
    return int(version)


def test_a_new_store_records_its_schema_version(store: SessionStore) -> None:
    assert schema_version(store) == SCHEMA_VERSION


def test_a_file_from_before_versioning_is_adopted(tmp_path: Path, analysis: Analysis) -> None:
    session_id = save(SessionStore(tmp_path), analysis)
    with sqlite3.connect(tmp_path / "sessions.db") as db:
        db.execute("PRAGMA user_version = 0")  # like the files Phase 7 first wrote

    reopened = SessionStore(tmp_path)

    assert schema_version(reopened) == SCHEMA_VERSION
    assert [s.id for s in reopened.history()] == [session_id]


def test_migrations_upgrade_an_older_file(
    tmp_path: Path, analysis: Analysis, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_id = save(SessionStore(tmp_path), analysis)
    monkeypatch.setattr(store_module, "SCHEMA_VERSION", 2)
    monkeypatch.setattr(
        store_module, "MIGRATIONS", {2: "ALTER TABLE sessions ADD COLUMN tempo REAL;"}
    )

    reopened = SessionStore(tmp_path)

    assert schema_version(reopened) == 2
    with sqlite3.connect(tmp_path / "sessions.db") as db:
        columns = [row[1] for row in db.execute("PRAGMA table_info(sessions)")]
    assert "tempo" in columns
    assert [s.id for s in reopened.history()] == [session_id]  # nothing lost


def test_a_file_from_a_newer_app_is_refused(tmp_path: Path) -> None:
    SessionStore(tmp_path)
    with sqlite3.connect(tmp_path / "sessions.db") as db:
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(sqlite3.DatabaseError, match="newer version of the app"):
        SessionStore(tmp_path)
