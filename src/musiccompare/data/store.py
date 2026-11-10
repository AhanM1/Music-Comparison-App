"""Save every practice session: for your history now, and as ML training data later.

Two kinds of data, two kinds of storage:

- **Small facts** (when, which instrument, the scores, the tips and how you
  rated them) go in a **SQLite** database: one file, ``sessions.db``. SQLite is
  a full SQL database that ships inside Python (``import sqlite3``); no server
  to install. Tables are like spreadsheets with typed columns, and SQL lets us
  ask questions like "every tip rated 👎 for viola players".
- **Big arrays** (every frame's pitch, loudness, ...) would bloat a database,
  so each session's arrays go in one compressed **NPZ** file (numpy's own
  format: several named arrays zipped together), ``arrays/<session id>.npz``.
- **Recordings** are saved only if the player opts in, as
  ``audio/<session id>_ref.wav`` and ``_user.wav``, and deleted again if they
  change their mind.

Why store measurements and not just scores? An ML model (Phase 8) learns from
examples. If we only kept scores, we couldn't later train on, say, "what the
pitch curve looked like when a tip was rated helpful". Arrays can't be
collected after the fact, so we keep them from day one, tagged with the
``extractor_version`` so data from different feature code never gets mixed.

What is a "session"? One analysis of one pair of recordings. Its identity is
a *fingerprint*: a hash of both recordings plus the version of the analysis
code. Streamlit reruns the script on every click, so saving must be
**idempotent**: saving the same fingerprint again returns the same session.
Your instrument, level, and piece name are *details* of a session, not part of
its identity: change them, and the session is updated (with new tips) instead
of a duplicate appearing in your history. New analysis code (a new version)
gives new scores, so that is a new session.
"""

import hashlib
import os
import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from musiccompare import config
from musiccompare.audio.export import to_wav_bytes
from musiccompare.models import (
    DIMENSIONS,
    Alignment,
    AudioClip,
    ComparisonResult,
    Dimension,
    FeatureSet,
    FeedbackItem,
    FrameDifferences,
    Instrument,
    PlayerContext,
    SkillLevel,
    no_spans,
)

# The database layout version, stored in the file itself (SQLite's
# "user_version"). Bump it and add a step to MIGRATIONS whenever a table
# changes, so existing history files are upgraded instead of breaking.
SCHEMA_VERSION = 1

# MIGRATIONS[v] = SQL that upgrades a version v-1 database to version v,
# e.g. {2: "ALTER TABLE sessions ADD COLUMN tempo REAL;"}. Run in order.
MIGRATIONS: dict[int, str] = {}

Rating = Literal[-1, 1]  # 👎 = -1, 👍 = +1

# The database layout. "IF NOT EXISTS" makes it safe to run on every start.
# REFERENCES ... ON DELETE CASCADE: deleting a session deletes its tips and
# regions too, so nothing is ever left pointing at a session that's gone.
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id                TEXT PRIMARY KEY,
    fingerprint       TEXT NOT NULL UNIQUE,
    created_at        TEXT NOT NULL,      -- ISO 8601, UTC
    schema_version    INTEGER NOT NULL,
    extractor_version TEXT NOT NULL,
    feedback_source   TEXT NOT NULL,
    instrument        TEXT NOT NULL,
    skill_level       TEXT NOT NULL,
    piece             TEXT NOT NULL,
    ref_name          TEXT NOT NULL,
    user_name         TEXT NOT NULL,
    ref_duration_s    REAL NOT NULL,
    user_duration_s   REAL NOT NULL,
    alignment_cost    REAL NOT NULL,
    score_pitch       REAL,               -- NULL = nothing to compare
    score_dynamics    REAL,
    score_tone        REAL,
    audio_saved       INTEGER NOT NULL    -- SQLite has no bool: 0 / 1
);
CREATE TABLE IF NOT EXISTS regions (
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    dimension    TEXT NOT NULL,
    start_s      REAL NOT NULL,
    end_s        REAL NOT NULL,
    user_start_s REAL NOT NULL,
    user_end_s   REAL NOT NULL,
    mean_error   REAL NOT NULL,
    severity     REAL NOT NULL,
    PRIMARY KEY (session_id, idx)
);
CREATE TABLE IF NOT EXISTS feedback (
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx               INTEGER NOT NULL,   -- position in the ranked list (0 = top)
    dimension         TEXT NOT NULL,
    message           TEXT NOT NULL,
    tip               TEXT NOT NULL,
    importance        REAL NOT NULL,
    confidence        REAL NOT NULL,
    source            TEXT NOT NULL,
    region_start_s    REAL,               -- all NULL for a whole-take habit
    region_end_s      REAL,
    region_mean_error REAL,
    region_severity   REAL,
    rating            INTEGER,            -- -1 / +1, NULL = not rated
    rated_at          TEXT,
    PRIMARY KEY (session_id, idx)
);
"""

FEATURE_ARRAYS = (
    "times",
    "f0_hz",
    "voiced_prob",
    "rms_db",
    "centroid_hz",
    "flatness",
    "mfcc",
    "chroma",
)
DIFFERENCE_ARRAYS = ("times", "pitch_cents", "dynamics_db", "tone_octaves")

# Seconds to wait for another save (e.g. a second browser tab) to finish.
LOCK_TIMEOUT_S = 10.0


def fingerprint(ref_data: bytes, user_data: bytes, analysis_version: str) -> str:
    """A hash identifying one analysis: these two recordings, this analysis code.

    ``analysis_version`` must change whenever the analysis would give
    different numbers (see pipeline.ANALYSIS_VERSION); otherwise re-uploading
    the same takes after an upgrade would show the old, stale scores.
    """
    h = hashlib.sha256()
    for part in (ref_data, user_data):
        h.update(hashlib.sha256(part).digest())  # hash of hashes: no ambiguity at the seams
    h.update(f"|{analysis_version}".encode())
    return h.hexdigest()


@dataclass(frozen=True)
class SessionSummary:
    """One row of your practice history."""

    id: str
    created_at: datetime
    instrument: Instrument
    skill_level: SkillLevel
    piece: str
    ref_name: str
    user_name: str
    scores: dict[Dimension, float | None]
    n_tips: int
    audio_saved: bool


@dataclass(frozen=True)
class RatedTip:
    """One labelled training example: a tip, everything around it, and your 👍/👎."""

    session_id: str
    created_at: datetime
    instrument: Instrument
    skill_level: SkillLevel
    piece: str
    extractor_version: str
    scores: dict[Dimension, float | None]  # the session's scores
    position: int  # where the tip was in the ranked list (0 = top)
    dimension: Dimension
    message: str
    tip: str
    importance: float
    confidence: float
    source: str
    region_duration_s: float | None  # None = a habit across the whole take
    region_mean_error: float | None  # signed: sharp/louder/brighter > 0
    region_severity: float | None
    rating: Rating


@dataclass(frozen=True)
class SavedArrays:
    """The measurements of a saved session, rebuilt as the app's own models."""

    ref: FeatureSet
    user: FeatureSet
    alignment: Alignment
    differences: FrameDifferences


def default_data_dir() -> Path:
    """The storage folder: $MUSICCOMPARE_DATA_DIR if set, else <project>/data.

    Anchored to the project folder, not the folder the app was started from,
    so starting Streamlit from somewhere else doesn't create a second, empty
    history. If the package isn't running from the project folder (an
    installed copy), it falls back to ~/.musiccompare.
    """
    if env := os.environ.get(config.DATA_DIR_ENV):
        return Path(env)
    project = Path(__file__).resolve().parents[3]  # src/musiccompare/data/store.py
    if (project / "pyproject.toml").exists():
        return project / config.DEFAULT_DATA_DIR
    return Path.home() / ".musiccompare"


class SessionStore:
    """Read and write saved sessions under one folder."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.arrays_dir = root / "arrays"
        self.audio_dir = root / "audio"
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._open()) as db:
            migrate(db)

    def _open(self) -> sqlite3.Connection:
        # isolation_level=None: we start transactions ourselves (BEGIN below).
        db = sqlite3.connect(
            self.root / "sessions.db", timeout=LOCK_TIMEOUT_S, isolation_level=None
        )
        db.row_factory = sqlite3.Row  # rows can be read by column name
        db.execute("PRAGMA foreign_keys = ON")  # SQLite leaves these off by default
        return db

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """A connection inside one all-or-nothing transaction.

        A *transaction* groups several reads and writes so they happen
        all-or-nothing: if saving the tips fails, the session row is undone
        too, so the database never holds half a session.

        ``BEGIN IMMEDIATE`` takes the write lock right away. Two saves at once
        (two browser tabs) then run one after the other: the second waits, then
        finds the session the first one saved, instead of both trying to add it.
        """
        with closing(self._open()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")

    @contextmanager
    def _reading(self) -> Iterator[sqlite3.Connection]:
        with closing(self._open()) as db:
            yield db

    def save_session(
        self,
        *,
        fingerprint: str,
        ref: FeatureSet,
        user: FeatureSet,
        result: ComparisonResult,
        context: PlayerContext,
        feedback: list[FeedbackItem],
        feedback_source: str,
        ref_clip: AudioClip,
        user_clip: AudioClip,
        keep_audio: bool = False,
        now: datetime | None = None,
    ) -> str:
        """Save one analysis and return its session id.

        Saving a fingerprint that already exists updates that session in place:
        the piece name; the tips, if your instrument, level, or the feedback
        provider changed (ratings carry over to tips that are word-for-word the
        same); and the recordings, which are saved or deleted to match
        ``keep_audio``.
        """
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM sessions WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            if row is not None:
                self._update(db, row, context, feedback, feedback_source)
                self._sync_audio(db, row, ref_clip, user_clip, keep_audio)
                return str(row["id"])

            session_id = uuid.uuid4().hex
            # Files first: if the database insert below fails, the transaction
            # rolls back and we delete the files we just wrote.
            self._save_arrays(session_id, ref, user, result)
            if keep_audio:
                self._save_audio(session_id, ref_clip, user_clip)
            try:
                self._insert(
                    db,
                    session_id,
                    fingerprint,
                    now or datetime.now(UTC),
                    ref,
                    result,
                    context,
                    feedback_source,
                    ref_clip,
                    user_clip,
                    keep_audio,
                )
                _insert_feedback(db, session_id, feedback)
            except BaseException:
                self._delete_files(session_id)
                raise
            return session_id

    def _insert(
        self,
        db: sqlite3.Connection,
        session_id: str,
        fingerprint: str,
        now: datetime,
        ref: FeatureSet,
        result: ComparisonResult,
        context: PlayerContext,
        feedback_source: str,
        ref_clip: AudioClip,
        user_clip: AudioClip,
        keep_audio: bool,
    ) -> None:
        # "?" placeholders let sqlite3 insert values safely. Never build SQL by
        # pasting strings together: a piece called "x'; DROP TABLE sessions"
        # would run as SQL (an "SQL injection"). Columns are always named, so
        # adding a column later can never silently shift values into the wrong one.
        _insert_row(
            db,
            "sessions",
            {
                "id": session_id,
                "fingerprint": fingerprint,
                "created_at": now.astimezone(UTC).isoformat(),
                "schema_version": SCHEMA_VERSION,
                "extractor_version": ref.extractor_version,
                "feedback_source": feedback_source,
                "instrument": context.instrument,
                "skill_level": context.skill_level,
                "piece": context.piece,
                "ref_name": ref_clip.name,
                "user_name": user_clip.name,
                "ref_duration_s": ref_clip.duration_s,
                "user_duration_s": user_clip.duration_s,
                "alignment_cost": float(result.alignment.cost),
                **{f"score_{d}": _optional_float(result.scores[d]) for d in DIMENSIONS},
                "audio_saved": int(keep_audio),
            },
        )
        for i, r in enumerate(result.regions):
            _insert_row(
                db,
                "regions",
                {
                    "session_id": session_id,
                    "idx": i,
                    "dimension": r.dimension,
                    "start_s": r.start_s,
                    "end_s": r.end_s,
                    "user_start_s": r.user_start_s,
                    "user_end_s": r.user_end_s,
                    "mean_error": r.mean_error,
                    "severity": r.severity,
                },
            )

    def _update(
        self,
        db: sqlite3.Connection,
        row: sqlite3.Row,
        context: PlayerContext,
        feedback: list[FeedbackItem],
        feedback_source: str,
    ) -> None:
        """Bring an existing session's details (and tips) up to date."""
        db.execute("UPDATE sessions SET piece = ? WHERE id = ?", (context.piece, row["id"]))
        settings = (context.instrument, context.skill_level, feedback_source)
        if settings == (row["instrument"], row["skill_level"], row["feedback_source"]):
            return
        db.execute(
            "UPDATE sessions SET instrument = ?, skill_level = ?, feedback_source = ? WHERE id = ?",
            (*settings, row["id"]),
        )
        # The tips were written for the old settings, so replace them. A rating
        # is kept only where the new tip is word-for-word the one that was rated.
        old_ratings = {
            (r["message"], r["tip"]): (r["rating"], r["rated_at"])
            for r in db.execute(
                "SELECT message, tip, rating, rated_at FROM feedback "
                "WHERE session_id = ? AND rating IS NOT NULL",
                (row["id"],),
            )
        }
        db.execute("DELETE FROM feedback WHERE session_id = ?", (row["id"],))
        _insert_feedback(db, row["id"], feedback, old_ratings)

    def _sync_audio(
        self,
        db: sqlite3.Connection,
        row: sqlite3.Row,
        ref_clip: AudioClip,
        user_clip: AudioClip,
        keep_audio: bool,
    ) -> None:
        """Save the recordings if newly agreed to; delete them if consent was withdrawn."""
        if keep_audio == bool(row["audio_saved"]):
            return
        if keep_audio:
            self._save_audio(row["id"], ref_clip, user_clip)
        else:
            self._delete_audio(row["id"])
        db.execute("UPDATE sessions SET audio_saved = ? WHERE id = ?", (int(keep_audio), row["id"]))

    def _save_arrays(
        self, session_id: str, ref: FeatureSet, user: FeatureSet, result: ComparisonResult
    ) -> None:
        # Any, not ndarray: numpy's type hints would otherwise mistake **arrays for its
        # allow_pickle flag.
        arrays: dict[str, Any] = {}
        for prefix, features in (("ref", ref), ("user", user)):
            for name in FEATURE_ARRAYS:
                arrays[f"{prefix}_{name}"] = getattr(features, name)
            arrays[f"{prefix}_extractor_version"] = np.array(features.extractor_version)
        diff = result.differences
        for name in DIFFERENCE_ARRAYS:
            arrays[f"diff_{name}"] = getattr(diff, name)
        arrays["diff_level_offset_db"] = np.array(diff.level_offset_db)
        arrays["diff_octave_shift"] = np.array(diff.octave_shift)
        arrays["diff_missing"] = diff.missing
        arrays["alignment_path"] = result.alignment.path
        arrays["alignment_cost"] = np.array(result.alignment.cost)
        arrays["alignment_resolution"] = np.array(result.alignment.resolution)
        arrays["alignment_extra_user"] = result.alignment.extra_user
        arrays["alignment_skipped_ref"] = result.alignment.skipped_ref
        # Write to a temporary name, then rename: a crash mid-write can never
        # leave a half-written file under the real name.
        final = self._arrays_path(session_id)
        partial = final.with_suffix(".partial.npz")
        np.savez_compressed(partial, **arrays)
        partial.replace(final)

    def _save_audio(self, session_id: str, ref_clip: AudioClip, user_clip: AudioClip) -> None:
        self.audio_dir.mkdir(exist_ok=True)
        for role, clip in (("ref", ref_clip), ("user", user_clip)):
            (self.audio_dir / f"{session_id}_{role}.wav").write_bytes(to_wav_bytes(clip))

    def _arrays_path(self, session_id: str) -> Path:
        return self.arrays_dir / f"{session_id}.npz"

    def audio_paths(self, session_id: str) -> tuple[Path, Path] | None:
        """The saved reference and user recordings, or None if they weren't kept."""
        paths = tuple(self.audio_dir / f"{session_id}_{role}.wav" for role in ("ref", "user"))
        return (paths[0], paths[1]) if all(p.exists() for p in paths) else None

    def load_arrays(self, session_id: str) -> SavedArrays:
        """Rebuild a saved session's measurements (e.g. to train a model on them)."""
        # allow_pickle=False: never run code hidden in a file, only read plain arrays.
        with np.load(self._arrays_path(session_id), allow_pickle=False) as npz:

            def features(prefix: str) -> FeatureSet:
                return FeatureSet(
                    **{name: npz[f"{prefix}_{name}"] for name in FEATURE_ARRAYS},
                    extractor_version=str(npz[f"{prefix}_extractor_version"]),
                )

            return SavedArrays(
                ref=features("ref"),
                user=features("user"),
                alignment=Alignment(
                    path=npz["alignment_path"],
                    cost=float(npz["alignment_cost"]),
                    resolution=int(npz["alignment_resolution"]),
                    # Files saved before coverage was tracked don't have these:
                    # read them as "nothing unmatched".
                    extra_user=_get(npz, "alignment_extra_user", no_spans()),
                    skipped_ref=_get(npz, "alignment_skipped_ref", no_spans()),
                ),
                differences=FrameDifferences(
                    **{name: npz[f"diff_{name}"] for name in DIFFERENCE_ARRAYS},
                    level_offset_db=float(npz["diff_level_offset_db"]),
                    octave_shift=int(npz["diff_octave_shift"]),
                    missing=_get(npz, "diff_missing", np.zeros(len(npz["diff_times"]), bool)),
                ),
            )

    def rate(self, session_id: str, index: int, rating: Rating | None) -> None:
        """Record 👍 (+1) / 👎 (-1) for one tip; None clears the rating."""
        rated_at = None if rating is None else datetime.now(UTC).isoformat()
        with self._transaction() as db:
            updated = db.execute(
                "UPDATE feedback SET rating = ?, rated_at = ? WHERE session_id = ? AND idx = ?",
                (rating, rated_at, session_id, index),
            ).rowcount
        if updated != 1:
            raise KeyError(f"No tip {index} in session {session_id}")

    def ratings(self, session_id: str) -> dict[int, Rating]:
        """The ratings given so far in one session, by tip position."""
        with self._reading() as db:
            rows = db.execute(
                "SELECT idx, rating FROM feedback WHERE session_id = ? AND rating IS NOT NULL",
                (session_id,),
            ).fetchall()
        return {row["idx"]: row["rating"] for row in rows}

    def history(self) -> list[SessionSummary]:
        """Every saved session, oldest first."""
        with self._reading() as db:
            rows = db.execute(
                """
                SELECT s.*, COUNT(f.idx) AS n_tips
                FROM sessions s LEFT JOIN feedback f ON f.session_id = s.id
                GROUP BY s.id
                ORDER BY s.created_at
                """
            ).fetchall()
        return [
            SessionSummary(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                instrument=row["instrument"],
                skill_level=row["skill_level"],
                piece=row["piece"],
                ref_name=row["ref_name"],
                user_name=row["user_name"],
                scores=_scores(row),
                n_tips=row["n_tips"],
                audio_saved=bool(row["audio_saved"]),
            )
            for row in rows
        ]

    def rated_tips(self) -> list[RatedTip]:
        """Every tip that was rated: the labelled examples for Phase 8."""
        with self._reading() as db:
            rows = db.execute(
                """
                SELECT f.*, s.created_at, s.instrument, s.skill_level, s.piece,
                       s.extractor_version, s.score_pitch, s.score_dynamics, s.score_tone
                FROM feedback f JOIN sessions s ON s.id = f.session_id
                WHERE f.rating IS NOT NULL
                ORDER BY s.created_at, f.idx
                """
            ).fetchall()
        return [
            RatedTip(
                session_id=row["session_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                instrument=row["instrument"],
                skill_level=row["skill_level"],
                piece=row["piece"],
                extractor_version=row["extractor_version"],
                scores=_scores(row),
                position=row["idx"],
                dimension=row["dimension"],
                message=row["message"],
                tip=row["tip"],
                importance=row["importance"],
                confidence=row["confidence"],
                source=row["source"],
                region_duration_s=(
                    None
                    if row["region_start_s"] is None
                    else row["region_end_s"] - row["region_start_s"]
                ),
                region_mean_error=row["region_mean_error"],
                region_severity=row["region_severity"],
                rating=row["rating"],
            )
            for row in rows
        ]

    def delete_session(self, session_id: str) -> None:
        """Remove a session, its tips and regions, and its files."""
        with self._transaction() as db:
            db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._delete_files(session_id)

    def delete_all(self) -> None:
        for session in self.history():
            self.delete_session(session.id)

    def _delete_audio(self, session_id: str) -> None:
        for role in ("ref", "user"):
            (self.audio_dir / f"{session_id}_{role}.wav").unlink(missing_ok=True)

    def _delete_files(self, session_id: str) -> None:
        self._arrays_path(session_id).unlink(missing_ok=True)
        self._delete_audio(session_id)


def migrate(db: sqlite3.Connection) -> None:
    """Create the tables in a new file, or upgrade an older file to SCHEMA_VERSION.

    The version lives in the database file itself (``PRAGMA user_version``,
    a number SQLite keeps for exactly this). Files made before versioning
    existed read as 0; their tables already match version 1.
    """
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise sqlite3.DatabaseError(
            f"This history was saved by a newer version of the app (format {version}; "
            f"this version understands up to {SCHEMA_VERSION}). Update the app to open it."
        )
    if version == 0:
        db.executescript(SCHEMA)
        version = 1
    for step in range(version + 1, SCHEMA_VERSION + 1):
        db.executescript(MIGRATIONS[step])
    # PRAGMA values can't use "?" placeholders; this one is always our own int.
    db.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")


def _insert_row(db: sqlite3.Connection, table: str, values: dict[str, Any]) -> None:
    """INSERT one row, naming every column. ``table`` and the keys are our own constants."""
    columns = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    db.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(values.values()))


def _insert_feedback(
    db: sqlite3.Connection,
    session_id: str,
    feedback: Sequence[FeedbackItem],
    ratings: dict[tuple[str, str], tuple[int, str]] | None = None,
) -> None:
    """Insert a session's tips in ranked order, restoring any matching earlier ratings."""
    for i, item in enumerate(feedback):
        rating, rated_at = (ratings or {}).get((item.message, item.tip), (None, None))
        r = item.region
        _insert_row(
            db,
            "feedback",
            {
                "session_id": session_id,
                "idx": i,
                "dimension": item.dimension,
                "message": item.message,
                "tip": item.tip,
                "importance": item.importance,
                "confidence": item.confidence,
                "source": item.source,
                "region_start_s": None if r is None else r.start_s,
                "region_end_s": None if r is None else r.end_s,
                "region_mean_error": None if r is None else r.mean_error,
                "region_severity": None if r is None else r.severity,
                "rating": rating,
                "rated_at": rated_at,
            },
        )


def _scores(row: sqlite3.Row) -> dict[Dimension, float | None]:
    return {d: row[f"score_{d}"] for d in DIMENSIONS}


def _get(npz: Any, key: str, default: np.ndarray) -> np.ndarray:
    """An array from an NPZ file, or ``default`` if an older file doesn't have it."""
    return npz[key] if key in npz.files else default


def _optional_float(value: float | None) -> float | None:
    """numpy floats -> plain Python floats, which sqlite3 knows how to store."""
    return None if value is None else float(value)
