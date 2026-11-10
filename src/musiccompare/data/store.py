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
  ``audio/<session id>_ref.wav`` and ``_user.wav``.

Why store measurements and not just scores? An ML model (Phase 8) learns from
examples. If we only kept scores, we couldn't later train on, say, "what the
pitch curve looked like when a tip was rated helpful". Arrays can't be
collected after the fact, so we keep them from day one, tagged with the
``extractor_version`` so data from different feature code never gets mixed.

Streamlit reruns the script on every click, so saving must be **idempotent**:
saving the same analysis twice must not create two sessions. Each session gets
a *fingerprint* (a hash of both recordings plus everything that changes the
advice); saving a fingerprint that already exists returns the existing session.
"""

import hashlib
import os
import sqlite3
import uuid
from collections.abc import Iterator
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
)

# Bump when the tables or the NPZ layout change, so old files can be migrated.
SCHEMA_VERSION = 1

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


def fingerprint(ref_data: bytes, user_data: bytes, context: PlayerContext, source: str) -> str:
    """A hash identifying one analysis: same recordings + same advice settings.

    The piece name is left out on purpose: it's a label, not something that
    changes the advice, so renaming the piece updates the session instead of
    creating a new one.
    """
    h = hashlib.sha256()
    for part in (ref_data, user_data):
        h.update(hashlib.sha256(part).digest())  # hash of hashes: no ambiguity at the seams
    h.update(f"|{context.instrument}|{context.skill_level}|{source}".encode())
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
    """One labelled training example: a tip, its context, and your 👍/👎."""

    session_id: str
    instrument: Instrument
    skill_level: SkillLevel
    dimension: Dimension
    message: str
    tip: str
    importance: float
    confidence: float
    source: str
    region_duration_s: float | None  # None = a habit across the whole take
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
    """The storage folder: $MUSICCOMPARE_DATA_DIR if set, else ./data."""
    return Path(os.environ.get(config.DATA_DIR_ENV, config.DEFAULT_DATA_DIR))


class SessionStore:
    """Read and write saved sessions under one folder."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.arrays_dir = root / "arrays"
        self.audio_dir = root / "audio"
        self.arrays_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits on success and rolls back on error.

        A *transaction* groups several writes so they happen all-or-nothing: if
        saving the tips fails, the session row is undone too, so the database
        never holds half a session. ``with db:`` is sqlite3's transaction.
        """
        with closing(sqlite3.connect(self.root / "sessions.db")) as db:
            db.row_factory = sqlite3.Row  # rows can be read by column name
            db.execute("PRAGMA foreign_keys = ON")  # SQLite leaves these off by default
            with db:
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

        Saving the same fingerprint again returns the existing id. It still
        updates the piece name, and saves the recordings if the player has
        only now agreed to keep them.
        """
        with self._connect() as db:
            row = db.execute(
                "SELECT id, audio_saved FROM sessions WHERE fingerprint = ?", (fingerprint,)
            ).fetchone()
            if row is not None:
                session_id: str = row["id"]
                db.execute(
                    "UPDATE sessions SET piece = ? WHERE id = ?", (context.piece, session_id)
                )
                if keep_audio and not row["audio_saved"]:
                    self._save_audio(session_id, ref_clip, user_clip)
                    db.execute("UPDATE sessions SET audio_saved = 1 WHERE id = ?", (session_id,))
                return session_id

            session_id = uuid.uuid4().hex
            # Arrays first: if the database insert below then fails, the
            # transaction rolls back and we delete the orphaned files.
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
                    feedback,
                    feedback_source,
                    ref_clip,
                    user_clip,
                    keep_audio,
                )
            except Exception:
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
        feedback: list[FeedbackItem],
        feedback_source: str,
        ref_clip: AudioClip,
        user_clip: AudioClip,
        keep_audio: bool,
    ) -> None:
        # "?" placeholders let sqlite3 insert values safely. Never build SQL by
        # pasting strings together: a piece called "x'; DROP TABLE sessions"
        # would run as SQL (an "SQL injection").
        db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                fingerprint,
                now.astimezone(UTC).isoformat(),
                SCHEMA_VERSION,
                ref.extractor_version,
                feedback_source,
                context.instrument,
                context.skill_level,
                context.piece,
                ref_clip.name,
                user_clip.name,
                ref_clip.duration_s,
                user_clip.duration_s,
                float(result.alignment.cost),
                *(_optional_float(result.scores[d]) for d in DIMENSIONS),
                int(keep_audio),
            ),
        )
        db.executemany(
            "INSERT INTO regions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (session_id, i, r.dimension, r.start_s, r.end_s)
                + (r.user_start_s, r.user_end_s, r.mean_error, r.severity)
                for i, r in enumerate(result.regions)
            ],
        )
        db.executemany(
            "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
            [
                (session_id, i, item.dimension, item.message, item.tip)
                + (item.importance, item.confidence, item.source)
                + _region_columns(item)
                for i, item in enumerate(feedback)
            ],
        )

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
        arrays["alignment_path"] = result.alignment.path
        arrays["alignment_cost"] = np.array(result.alignment.cost)
        arrays["alignment_resolution"] = np.array(result.alignment.resolution)
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
                ),
                differences=FrameDifferences(
                    **{name: npz[f"diff_{name}"] for name in DIFFERENCE_ARRAYS},
                    level_offset_db=float(npz["diff_level_offset_db"]),
                    octave_shift=int(npz["diff_octave_shift"]),
                ),
            )

    def rate(self, session_id: str, index: int, rating: Rating | None) -> None:
        """Record 👍 (+1) / 👎 (-1) for one tip; None clears the rating."""
        rated_at = None if rating is None else datetime.now(UTC).isoformat()
        with self._connect() as db:
            updated = db.execute(
                "UPDATE feedback SET rating = ?, rated_at = ? WHERE session_id = ? AND idx = ?",
                (rating, rated_at, session_id, index),
            ).rowcount
        if updated != 1:
            raise KeyError(f"No tip {index} in session {session_id}")

    def ratings(self, session_id: str) -> dict[int, Rating]:
        """The ratings given so far in one session, by tip position."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT idx, rating FROM feedback WHERE session_id = ? AND rating IS NOT NULL",
                (session_id,),
            ).fetchall()
        return {row["idx"]: row["rating"] for row in rows}

    def history(self) -> list[SessionSummary]:
        """Every saved session, oldest first."""
        with self._connect() as db:
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
                scores={d: row[f"score_{d}"] for d in DIMENSIONS},
                n_tips=row["n_tips"],
                audio_saved=bool(row["audio_saved"]),
            )
            for row in rows
        ]

    def rated_tips(self) -> list[RatedTip]:
        """Every tip that was rated: the labelled examples for Phase 8."""
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT f.*, s.instrument, s.skill_level
                FROM feedback f JOIN sessions s ON s.id = f.session_id
                WHERE f.rating IS NOT NULL
                ORDER BY s.created_at, f.idx
                """
            ).fetchall()
        return [
            RatedTip(
                session_id=row["session_id"],
                instrument=row["instrument"],
                skill_level=row["skill_level"],
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
                region_severity=row["region_severity"],
                rating=row["rating"],
            )
            for row in rows
        ]

    def delete_session(self, session_id: str) -> None:
        """Remove a session, its tips and regions, and its files."""
        with self._connect() as db:
            db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._delete_files(session_id)

    def delete_all(self) -> None:
        for session in self.history():
            self.delete_session(session.id)

    def _delete_files(self, session_id: str) -> None:
        self._arrays_path(session_id).unlink(missing_ok=True)
        for role in ("ref", "user"):
            (self.audio_dir / f"{session_id}_{role}.wav").unlink(missing_ok=True)


def _optional_float(value: float | None) -> float | None:
    """numpy floats -> plain Python floats, which sqlite3 knows how to store."""
    return None if value is None else float(value)


def _region_columns(item: FeedbackItem) -> tuple[float | None, ...]:
    """Where a tip applies; all None for a habit across the whole take."""
    r = item.region
    return (None,) * 4 if r is None else (r.start_s, r.end_s, r.mean_error, r.severity)
