"""SQLite persistence layer for MIR analysis cache and mix sessions."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dj_ai.config import DB_PATH, ensure_directories


@dataclass
class SectionRecord:
    """Structural segment of a track."""

    label: str  # 'intro', 'verse', 'buildup', 'chorus_drop', 'outro'
    start_sec: float
    end_sec: float
    energy_score: float
    is_climax: bool = False


@dataclass
class TrackRecord:
    """Analyzed audio track metadata and MIR features."""

    audio_hash: str
    file_path: str
    title: str
    artist: str
    genre: str
    duration_sec: float
    bpm: float
    bpm_confidence: float
    is_tempo_stable: bool
    camelot_key: str
    lufs_integrated: float
    energy_avg: float
    danceability: float
    first_downbeat_sec: float
    beat_positions: list[float]
    sections: list[SectionRecord]


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tracks (
    audio_hash TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    title TEXT,
    artist TEXT,
    genre TEXT,
    duration_sec REAL NOT NULL,
    bpm REAL NOT NULL,
    bpm_confidence REAL,
    is_tempo_stable INTEGER NOT NULL DEFAULT 1,
    camelot_key TEXT NOT NULL,
    lufs_integrated REAL NOT NULL,
    energy_avg REAL NOT NULL,
    danceability REAL NOT NULL,
    first_downbeat_sec REAL NOT NULL,
    beat_positions_json TEXT NOT NULL,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS track_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_hash TEXT NOT NULL REFERENCES tracks(audio_hash) ON DELETE CASCADE,
    label TEXT NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    energy_score REAL NOT NULL,
    is_climax INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS mix_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_context TEXT,
    target_style TEXT,
    output_file_path TEXT NOT NULL,
    total_duration_sec REAL NOT NULL,
    tracklist_json TEXT NOT NULL
);
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled."""
    ensure_directories()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize database tables if they do not exist."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def get_track(audio_hash: str, db_path: Path = DB_PATH) -> TrackRecord | None:
    """Fetch a cached TrackRecord and its sections by audio_hash."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM tracks WHERE audio_hash = ?", (audio_hash,)
        ).fetchone()
        if row is None:
            return None

        section_rows = conn.execute(
            """
            SELECT label, start_sec, end_sec, energy_score, is_climax
            FROM track_sections
            WHERE audio_hash = ?
            ORDER BY start_sec ASC
            """,
            (audio_hash,),
        ).fetchall()

        sections = [
            SectionRecord(
                label=s["label"],
                start_sec=float(s["start_sec"]),
                end_sec=float(s["end_sec"]),
                energy_score=float(s["energy_score"]),
                is_climax=bool(s["is_climax"]),
            )
            for s in section_rows
        ]

        return TrackRecord(
            audio_hash=row["audio_hash"],
            file_path=row["file_path"],
            title=row["title"] or "",
            artist=row["artist"] or "",
            genre=row["genre"] or "unknown",
            duration_sec=float(row["duration_sec"]),
            bpm=float(row["bpm"]),
            bpm_confidence=float(row["bpm_confidence"] or 0.0),
            is_tempo_stable=bool(row["is_tempo_stable"]),
            camelot_key=row["camelot_key"],
            lufs_integrated=float(row["lufs_integrated"]),
            energy_avg=float(row["energy_avg"]),
            danceability=float(row["danceability"]),
            first_downbeat_sec=float(row["first_downbeat_sec"]),
            beat_positions=json.loads(row["beat_positions_json"]),
            sections=sections,
        )


def update_track_path(audio_hash: str, file_path: str, db_path: Path = DB_PATH) -> None:
    """Update the file path of a cached track if the file was moved or renamed."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE tracks SET file_path = ? WHERE audio_hash = ?",
            (file_path, audio_hash),
        )


def save_track(track: TrackRecord, db_path: Path = DB_PATH) -> None:
    """Insert or replace a TrackRecord and its structural sections."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO tracks (
                audio_hash, file_path, title, artist, genre, duration_sec,
                bpm, bpm_confidence, is_tempo_stable, camelot_key,
                lufs_integrated, energy_avg, danceability,
                first_downbeat_sec, beat_positions_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                track.audio_hash,
                track.file_path,
                track.title,
                track.artist,
                track.genre,
                track.duration_sec,
                track.bpm,
                track.bpm_confidence,
                1 if track.is_tempo_stable else 0,
                track.camelot_key,
                track.lufs_integrated,
                track.energy_avg,
                track.danceability,
                track.first_downbeat_sec,
                json.dumps(track.beat_positions),
            ),
        )
        conn.execute(
            "DELETE FROM track_sections WHERE audio_hash = ?", (track.audio_hash,)
        )
        conn.executemany(
            """
            INSERT INTO track_sections (
                audio_hash, label, start_sec, end_sec, energy_score, is_climax
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    track.audio_hash,
                    s.label,
                    s.start_sec,
                    s.end_sec,
                    s.energy_score,
                    1 if s.is_climax else 0,
                )
                for s in track.sections
            ],
        )


def save_mix_session(
    session_id: str,
    user_context: str,
    target_style: str,
    output_file_path: str,
    total_duration_sec: float,
    tracklist: Iterable[dict],
    db_path: Path = DB_PATH,
) -> None:
    """Persist a generated mix session record."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO mix_sessions (
                session_id, user_context, target_style,
                output_file_path, total_duration_sec, tracklist_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_context,
                target_style,
                output_file_path,
                total_duration_sec,
                json.dumps(list(tracklist)),
            ),
        )
