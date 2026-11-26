"""Global configuration and genre profiles for the DJ-AI engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE: int = 44100
CHANNELS: int = 2
TARGET_LUFS: float = -14.0
MAX_BPM_STRETCH_RATIO: float = 0.08  # ±8% max time-stretch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "dj_ai_cache.db"
OUTPUT_DIR = DATA_DIR / "mixes"
UPLOAD_DIR = DATA_DIR / "uploads"


@dataclass(frozen=True)
class GenreProfile:
    """Pacing and mixing characteristics for a musical genre family."""

    name: str
    min_play_sec: float
    max_play_sec: float
    prefer_climax_focus: bool
    default_transition_bars: int
    usually_quantized: bool


GENRE_PROFILES: dict[str, GenreProfile] = {
    "reggaeton": GenreProfile("reggaeton", 75.0, 140.0, False, 8, True),
    "trap": GenreProfile("trap", 65.0, 120.0, True, 8, True),
    "dancehall": GenreProfile("dancehall", 75.0, 135.0, False, 8, True),
    "salsa": GenreProfile("salsa", 95.0, 165.0, False, 8, False),
    "banda": GenreProfile("banda", 85.0, 145.0, False, 4, False),
    "latin": GenreProfile("latin", 80.0, 145.0, False, 8, True),
    "pop": GenreProfile("pop", 60.0, 115.0, True, 8, True),
    "electronic": GenreProfile("electronic", 65.0, 125.0, True, 16, True),
    "hiphop": GenreProfile("hiphop", 65.0, 115.0, True, 8, True),
    "unknown": GenreProfile("unknown", 70.0, 125.0, True, 8, True),
}


def ensure_directories() -> None:
    """Create runtime data directories if they do not exist."""
    for path in (DATA_DIR, OUTPUT_DIR, UPLOAD_DIR):
        path.mkdir(parents=True, exist_ok=True)
