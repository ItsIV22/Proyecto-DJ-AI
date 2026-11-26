"""Music Information Retrieval (MIR) pipeline for BPM, Camelot key, structure, and climax."""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any

import librosa
import numpy as np
import pyloudnorm as pyln
import requests

from dj_ai.audio_io import compute_audio_hash, load_audio, read_metadata
from dj_ai.config import SAMPLE_RATE
from dj_ai.db import SectionRecord, TrackRecord, get_track, save_track, update_track_path

logger = logging.getLogger(__name__)

# Krumhansl-Schmuckler key profiles
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)

# Pitch class index (0=C .. 11=B) -> Camelot notation
_MAJOR_TO_CAMELOT = [
    "8B",  # C
    "3B",  # C# / Db
    "10B", # D
    "5B",  # Eb
    "12B", # E
    "7B",  # F
    "2B",  # F# / Gb
    "9B",  # G
    "4B",  # Ab
    "11B", # A
    "6B",  # Bb
    "1B",  # B
]
_MINOR_TO_CAMELOT = [
    "5A",  # Cm
    "12A", # C#m
    "7A",  # Dm
    "2A",  # Ebm
    "9A",  # Em
    "4A",  # Fm
    "11A", # F#m
    "6A",  # Gm
    "1A",  # G#m
    "8A",  # Am
    "3A",  # Bbm
    "10A", # Bm
]

_GENRE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("dancehall", ("salsa choque", "salsa choke", "choque", "dancehall", "moombahton", "afrobeat", "reggae", "shatta")),
    ("reggaeton", ("reggaeton", "reggaetón", "urbano latino", "perreo", "dembow", "latin urban")),
    ("trap", ("trap", "latin trap", "trap latino", "drill", "hip hop", "hip-hop", "rap", "r&b", "rnb")),
    ("banda", ("banda", "regional mexicano", "música mexicana", "musica mexicana", "corridos", "norteño", "norteno", "mariachi", "sierreño")),
    ("salsa", ("salsa", "timba", "mambo", "son cubano", "merengue", "bachata", "cumbia", "música tropical", "musica tropical")),
    ("electronic", ("electronic", "house", "techno", "edm", "trance", "electro", "dubstep", "dnb", "drum and bass", "dance", "club", "progressive")),
    ("pop", ("pop", "latin pop", "pop latino", "synthpop", "electropop", "indie pop", "rock")),
    ("latin", ("latin", "latino", "latina", "tropical", "urbano")),
]


def _clean_query_text(text: str) -> str:
    """Strip YouTube/download noise like (Official Video), [Audio], ft., etc."""
    cleaned = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)
    cleaned = re.sub(r"\b(feat\.?|ft\.?|official|video|audio|lyrics|letra|hd|4k)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_genre(raw_genre: str) -> str:
    """Map free-form genre strings to canonical DJ-AI genre families."""
    text = raw_genre.strip().lower()
    if not text:
        return "unknown"
    for canonical, keywords in _GENRE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return canonical
    return "unknown"


def _lookup_genre_online(artist: str, title: str) -> str:
    """Best-effort free genre lookup via iTunes Search API and Deezer (no key required)."""
    clean_artist = _clean_query_text(artist)
    clean_title = _clean_query_text(title)
    query_parts = [p for p in (clean_artist, clean_title) if p]
    if not query_parts:
        return "unknown"

    # Check title/artist keywords directly first (e.g. "salsa choque", "trap", "banda")
    direct = normalize_genre(" ".join(query_parts))
    if direct != "unknown":
        return direct

    q = urllib.parse.quote(" ".join(query_parts))

    # 1. iTunes Search API (excellent coverage for Latin, Urbano, Trap, Banda, Dancehall)
    try:
        it_resp = requests.get(
            f"https://itunes.apple.com/search?term={q}&entity=song&limit=3",
            timeout=2.5,
        )
        if it_resp.status_code == 200:
            results = it_resp.json().get("results", [])
            for item in results:
                raw_g = item.get("primaryGenreName", "")
                mapped = normalize_genre(raw_g)
                if mapped != "unknown":
                    return mapped
    except Exception:
        pass

    # 2. Deezer public API fallback
    try:
        resp = requests.get(f"https://api.deezer.com/search?q={q}&limit=1", timeout=2.0)
        if resp.status_code != 200:
            return "unknown"
        data = resp.json().get("data", [])
        if not data:
            return "unknown"
        album_id = data[0].get("album", {}).get("id")
        if not album_id:
            return "unknown"
        alb_resp = requests.get(f"https://api.deezer.com/album/{album_id}", timeout=2.0)
        if alb_resp.status_code != 200:
            return "unknown"
        genres = alb_resp.json().get("genres", {}).get("data", [])
        for g in genres:
            mapped = normalize_genre(g.get("name", ""))
            if mapped != "unknown":
                return mapped
    except Exception:
        pass
    return "unknown"


def _infer_genre_acoustic(bpm: float, is_stable: bool, danceability: float, energy: float) -> str:
    """Fallback genre heuristic from acoustic features when tags are missing."""
    if not is_stable and 80.0 <= bpm <= 125.0:
        return "salsa"
    if 86.0 <= bpm <= 104.0 and danceability >= 0.55:
        return "reggaeton"
    if 128.0 <= bpm <= 155.0 and energy < 0.65:
        return "trap"
    if 118.0 <= bpm <= 135.0 and energy >= 0.55:
        return "electronic"
    if 100.0 <= bpm <= 117.0:
        return "dancehall"
    return "pop"


def detect_camelot_key(mono_audio: np.ndarray, sr: int) -> str:
    """Estimate musical key and return its Camelot Wheel code (e.g., '8A', '11B')."""
    # Use a central slice up to 90s for fast, representative harmonic analysis
    max_samples = sr * 90
    if mono_audio.shape[0] > max_samples:
        start = (mono_audio.shape[0] - max_samples) // 2
        segment = mono_audio[start : start + max_samples]
    else:
        segment = mono_audio

    chroma = librosa.feature.chroma_cqt(y=segment, sr=sr, bins_per_octave=36)
    chroma_mean = np.mean(chroma, axis=1)
    if np.std(chroma_mean) < 1e-8:
        return "8A"

    best_corr = -np.inf
    best_camelot = "8A"
    for shift in range(12):
        maj_rot = np.roll(_MAJOR_PROFILE, shift)
        min_rot = np.roll(_MINOR_PROFILE, shift)
        maj_corr = float(np.corrcoef(chroma_mean, maj_rot)[0, 1])
        min_corr = float(np.corrcoef(chroma_mean, min_rot)[0, 1])
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_camelot = _MAJOR_TO_CAMELOT[shift]
        if min_corr > best_corr:
            best_corr = min_corr
            best_camelot = _MINOR_TO_CAMELOT[shift]

    return best_camelot


def _detect_beats_and_tempo(
    mono_audio: np.ndarray, sr: int
) -> tuple[float, float, bool, float, list[float]]:
    """Extract BPM, confidence, tempo stability, first downbeat, and beat timestamps."""
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=mono_audio, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="frames"
    )
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    # Normalize DJ tempo range (avoid half-time/double-time traps for common genres)
    if 45.0 <= bpm < 72.0:
        bpm *= 2.0
    elif bpm > 165.0:
        bpm /= 2.0

    if len(beat_times) < 8:
        duration = mono_audio.shape[0] / sr
        step = 60.0 / max(bpm, 90.0)
        beat_times = np.arange(0.0, duration, step)
        return round(bpm, 2), 0.3, True, float(beat_times[0]), [round(float(t), 4) for t in beat_times]

    ibis = np.diff(beat_times)
    median_ibi = float(np.median(ibis))
    # Use 10th-90th percentile IBIs so occasional beat-tracker half/double skips don't skew stability
    q10, q90 = np.percentile(ibis, [10, 90])
    trimmed = ibis[(ibis >= q10) & (ibis <= q90)]
    if len(trimmed) < 4:
        trimmed = ibis
    cv = float(np.std(trimmed) / max(median_ibi, 1e-6))
    is_stable = cv < 0.085
    confidence = float(np.clip(1.0 - cv * 3.5, 0.1, 0.99))

    # Downbeat phase estimation (0..3) using onset strength + low-frequency energy
    n_beats = len(beat_frames)
    phase_scores = np.zeros(4, dtype=np.float64)
    for phase in range(4):
        idx = beat_frames[phase::4]
        idx = idx[idx < len(onset_env)]
        if len(idx) > 0:
            phase_scores[phase] = float(np.mean(onset_env[idx]))
    best_phase = int(np.argmax(phase_scores))
    first_downbeat = float(beat_times[best_phase]) if best_phase < len(beat_times) else float(beat_times[0])

    return (
        round(bpm, 2),
        round(confidence, 3),
        is_stable,
        round(first_downbeat, 4),
        [round(float(t), 4) for t in beat_times],
    )


def _segment_and_find_climax(
    mono_audio: np.ndarray,
    sr: int,
    duration_sec: float,
    beat_times: list[float],
) -> tuple[list[SectionRecord], float, float]:
    """Segment track into musical sections and identify climax/drop regions."""
    # Compute 1-second resolution RMS and spectral contrast/novelty
    hop_1hz = sr
    frame_len = sr * 2
    rms = librosa.feature.rms(y=mono_audio, frame_length=frame_len, hop_length=hop_1hz)[0]
    onset_1hz = librosa.resample(
        librosa.onset.onset_strength(y=mono_audio, sr=sr, hop_length=512),
        orig_sr=sr / 512.0,
        target_sr=1.0,
    )
    n_sec = min(len(rms), len(onset_1hz), max(1, int(np.floor(duration_sec))))
    rms = rms[:n_sec]
    onset_1hz = onset_1hz[:n_sec]

    rms_norm = rms / (np.percentile(rms, 95) + 1e-8)
    onset_norm = onset_1hz / (np.percentile(onset_1hz, 95) + 1e-8)
    energy_curve = np.clip(0.7 * rms_norm + 0.3 * onset_norm, 0.0, 1.2)

    # Smooth energy curve over a 6-second window to reveal macro structure
    kernel = np.ones(6, dtype=np.float64) / 6.0
    smooth = np.convolve(energy_curve, kernel, mode="same") if n_sec >= 6 else energy_curve

    # Determine section boundaries every ~16-24s snapped to local energy minima or downbeats
    boundaries = [0.0]
    cursor = 16.0
    while cursor < duration_sec - 14.0:
        win_start = int(max(0, cursor - 5))
        win_end = int(min(n_sec, cursor + 6))
        if win_end > win_start:
            local_min_idx = win_start + int(np.argmin(smooth[win_start:win_end]))
            bound_sec = float(local_min_idx)
        else:
            bound_sec = cursor
        # Snap to nearest beat if available
        if beat_times:
            nearest_beat = min(beat_times, key=lambda b: abs(b - bound_sec))
            if abs(nearest_beat - bound_sec) < 2.0:
                bound_sec = nearest_beat
        if bound_sec - boundaries[-1] >= 10.0:
            boundaries.append(round(bound_sec, 3))
        cursor = boundaries[-1] + 20.0
    boundaries.append(round(duration_sec, 3))

    raw_sections: list[SectionRecord] = []
    for i in range(len(boundaries) - 1):
        s_start = boundaries[i]
        s_end = boundaries[i + 1]
        i0 = int(np.clip(np.floor(s_start), 0, max(0, n_sec - 1)))
        i1 = int(np.clip(np.ceil(s_end), i0 + 1, n_sec))
        sec_energy = float(np.mean(smooth[i0:i1])) if n_sec > 0 else 0.5
        raw_sections.append(
            SectionRecord(
                label="verse",
                start_sec=s_start,
                end_sec=s_end,
                energy_score=round(float(np.clip(sec_energy, 0.0, 1.0)), 3),
                is_climax=False,
            )
        )

    if raw_sections:
        energies = [s.energy_score for s in raw_sections]
        max_e = max(energies)
        median_e = float(np.median(energies))

        for idx, sec in enumerate(raw_sections):
            if idx == 0 and sec.energy_score < median_e * 0.95:
                sec.label = "intro"
            elif idx == len(raw_sections) - 1 and sec.energy_score < median_e * 0.95:
                sec.label = "outro"
            elif sec.energy_score >= max_e * 0.86:
                sec.label = "chorus_drop"
                sec.is_climax = True
            else:
                sec.label = "verse"

        # Mark section right before a chorus_drop as buildup if it's not intro
        for idx in range(1, len(raw_sections) - 1):
            if raw_sections[idx + 1].label == "chorus_drop" and raw_sections[idx].label == "verse":
                if raw_sections[idx].energy_score < raw_sections[idx + 1].energy_score:
                    raw_sections[idx].label = "buildup"

        # Guarantee at least one climax section
        if not any(s.is_climax for s in raw_sections):
            best_sec = max(raw_sections, key=lambda s: s.energy_score)
            best_sec.label = "chorus_drop"
            best_sec.is_climax = True

    energy_avg = float(np.clip(np.mean(energy_curve), 0.0, 1.0))
    # Danceability: combination of rhythmic regularity and mid-low energy
    if len(onset_1hz) > 4:
        ac = np.correlate(onset_norm - np.mean(onset_norm), onset_norm - np.mean(onset_norm), mode="full")
        ac = ac[len(ac) // 2 :]
        reg = float(np.max(ac[1:8]) / (ac[0] + 1e-8)) if len(ac) > 8 else 0.5
        danceability = float(np.clip(0.55 * reg + 0.45 * energy_avg, 0.1, 0.99))
    else:
        danceability = 0.5

    return raw_sections, round(energy_avg, 3), round(danceability, 3)


def analyze_track(file_path: str, allow_online_lookup: bool = True) -> TrackRecord:
    """Load, hash, check SQLite cache, and analyze an audio file if not cached."""
    audio, sr = load_audio(file_path, target_sr=SAMPLE_RATE)
    audio_hash = compute_audio_hash(audio)

    cached = get_track(audio_hash)
    if cached is not None:
        if cached.file_path != str(file_path):
            update_track_path(audio_hash, str(file_path))
            cached.file_path = str(file_path)
        if allow_online_lookup and cached.genre in ("unknown", "latin", "pop"):
            refined = _lookup_genre_online(cached.artist, cached.title)
            if refined != "unknown" and refined != cached.genre:
                cached.genre = refined
                save_track(cached)
        return cached

    meta = read_metadata(file_path)
    duration_sec = float(audio.shape[0] / sr)
    mono = np.mean(audio, axis=1)

    # Integrated LUFS (EBU R128)
    meter = pyln.Meter(sr)
    try:
        lufs = float(meter.integrated_loudness(audio))
        if not np.isfinite(lufs):
            lufs = -18.0
    except Exception:
        lufs = -18.0

    bpm, bpm_conf, is_stable, first_downbeat, beat_times = _detect_beats_and_tempo(mono, sr)
    camelot_key = detect_camelot_key(mono, sr)
    sections, energy_avg, danceability = _segment_and_find_climax(
        mono, sr, duration_sec, beat_times
    )

    genre = normalize_genre(meta["genre"])
    if genre == "unknown" and allow_online_lookup:
        genre = _lookup_genre_online(meta["artist"], meta["title"])
    if genre == "unknown":
        genre = _infer_genre_acoustic(bpm, is_stable, danceability, energy_avg)

    record = TrackRecord(
        audio_hash=audio_hash,
        file_path=str(file_path),
        title=meta["title"],
        artist=meta["artist"],
        genre=genre,
        duration_sec=round(duration_sec, 3),
        bpm=bpm,
        bpm_confidence=bpm_conf,
        is_tempo_stable=is_stable,
        camelot_key=camelot_key,
        lufs_integrated=round(lufs, 2),
        energy_avg=energy_avg,
        danceability=danceability,
        first_downbeat_sec=first_downbeat,
        beat_positions=beat_times,
        sections=sections,
    )
    save_track(record)
    return record
