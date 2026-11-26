"""End-to-end tests for DJ-AI MIR, curation, DSP, caching, and web API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from dj_ai.app import app
from dj_ai.audio_io import compute_audio_hash, load_audio
from dj_ai.config import SAMPLE_RATE
from dj_ai.curator import MixPreferences, camelot_score, order_tracks, parse_user_context, plan_mix
from dj_ai.db import init_db
from dj_ai.dsp import (
    apply_master_limiter,
    render_beatmatch_bass_swap,
    render_echo_out,
    render_filter_sweep,
    render_reverb_wash_drop,
    render_vinyl_brake_drop,
    time_stretch_audio,
)
from dj_ai.engine import generate_dj_mix
from dj_ai.mir import analyze_track, normalize_genre


def _synth_track(
    path: Path,
    bpm: float,
    freq_hz: float,
    duration_sec: float = 48.0,
    sr: int = SAMPLE_RATE,
) -> Path:
    """Synthesize a stereo musical test track with kick pulses, harmonic tone, and a climax."""
    n = int(duration_sec * sr)
    t = np.linspace(0.0, duration_sec, n, endpoint=False, dtype=np.float32)

    # Harmonic chord pad (root + fifth)
    pad = 0.12 * np.sin(2.0 * np.pi * freq_hz * t) + 0.08 * np.sin(
        2.0 * np.pi * (freq_hz * 1.5) * t
    )

    # Kick drum pulse on every beat
    beat_period = 60.0 / bpm
    beat_phase = (t % beat_period) / beat_period
    kick = 0.45 * np.exp(-beat_phase * 18.0) * np.sin(2.0 * np.pi * 65.0 * t)

    # Energy envelope: quiet intro (0-8s), verse (8-20s), climax (20-36s), outro (36-48s)
    env = np.ones_like(t) * 0.65
    env[t < 8.0] = 0.35
    env[(t >= 20.0) & (t < 36.0)] = 1.0
    env[t >= 36.0] = 0.40

    mono = (pad + kick) * env
    stereo = np.column_stack((mono, mono * 0.98)).astype(np.float32)
    sf.write(str(path), stereo, sr, subtype="PCM_16")
    return path


def test_dsp_primitives() -> None:
    sr = SAMPLE_RATE
    n = sr * 4
    t = np.linspace(0.0, 4.0, n, endpoint=False, dtype=np.float32)
    a = np.column_stack((np.sin(2 * np.pi * 100 * t), np.sin(2 * np.pi * 100 * t))) * 0.5
    b = np.column_stack((np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t))) * 0.5

    stretched = time_stretch_audio(a, sr, rate=1.05)
    assert stretched.ndim == 2 and stretched.shape[1] == 2
    assert stretched.shape[0] < a.shape[0]

    swap = render_beatmatch_bass_swap(a, b, sr=sr)
    assert swap.shape == a.shape and np.all(np.isfinite(swap))

    sweep = render_filter_sweep(a, b, sr=sr)
    assert sweep.shape == a.shape and np.all(np.isfinite(sweep))

    echo = render_echo_out(a, b, out_bpm=100.0, sr=sr)
    assert echo.shape == a.shape and np.all(np.isfinite(echo))

    wash = render_reverb_wash_drop(a, b, sr=sr)
    assert wash.shape == a.shape and np.all(np.isfinite(wash))

    brake = render_vinyl_brake_drop(a, b, sr=sr)
    assert brake.shape == a.shape and np.all(np.isfinite(brake))

    limited = apply_master_limiter(swap * 2.5, sr=sr)
    assert float(np.max(np.abs(limited))) <= 1.0


def test_camelot_and_context_parser() -> None:
    assert camelot_score("8A", "8A") == 1.0
    assert camelot_score("8A", "9A") == 0.90
    assert camelot_score("8A", "8B") == 0.85

    assert normalize_genre("Salsa Choque") == "dancehall"
    assert normalize_genre("Trap Latino") == "trap"
    assert normalize_genre("Regional Mexicano / Banda") == "banda"

    prefs = parse_user_context("Empieza con salsa, sigue con reggaetón y cierra con electrónica rápido")
    assert prefs.style == "quick_hype"
    assert prefs.genre_order_hint == ["salsa", "reggaeton", "electronic"]


def test_end_to_end_mix_and_cache(tmp_path: Path) -> None:
    init_db()
    t1 = _synth_track(tmp_path / "Artist A - Latin Groove.wav", bpm=96.0, freq_hz=220.0)
    t2 = _synth_track(tmp_path / "Artist B - Pop Hit.wav", bpm=100.0, freq_hz=261.63)
    t3 = _synth_track(tmp_path / "Artist C - Club Drop.wav", bpm=126.0, freq_hz=329.63)

    # First analysis populates cache
    rec1 = analyze_track(str(t1), allow_online_lookup=False)
    assert rec1.bpm > 50.0
    assert rec1.camelot_key.endswith(("A", "B"))
    assert any(s.is_climax for s in rec1.sections)

    # Hash stability check
    audio, _ = load_audio(t1)
    assert compute_audio_hash(audio) == rec1.audio_hash

    # Full mix generation
    result = generate_dj_mix(
        file_paths=[t1, t2, t3],
        user_context="Modo equilibrado",
        style="balanced",
        output_format="mp3",
        use_stems_on_jumps=False,
        allow_online_lookup=False,
    )
    out_file = Path(result["output_file"])
    assert out_file.exists() and out_file.stat().st_size > 10_000
    assert len(result["tracklist"]) == 3
    assert result["total_duration_sec"] > 40.0


def test_fastapi_routes() -> None:
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "DJ-AI Autonomous Mixer" in res.text

        bad = client.post("/api/mix", data={"style": "balanced"})
        assert bad.status_code == 400
