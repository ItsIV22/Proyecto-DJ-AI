"""Main autonomous DJ-AI orchestration engine."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from dj_ai.audio_io import SUPPORTED_EXTENSIONS, export_audio, load_audio
from dj_ai.config import OUTPUT_DIR, SAMPLE_RATE, TARGET_LUFS, ensure_directories
from dj_ai.curator import MixPreferences, TrackPlan, order_tracks, parse_user_context, plan_mix
from dj_ai.db import TrackRecord, init_db, save_mix_session
from dj_ai.dsp import (
    apply_master_limiter,
    normalize_lufs,
    render_beatmatch_bass_swap,
    render_echo_out,
    render_filter_sweep,
    render_reverb_wash_drop,
    render_stem_bridge,
    render_vinyl_brake_drop,
    time_stretch_audio,
)
from dj_ai.mir import analyze_track

ProgressCallback = Callable[[str, int, int, str], None]


def _default_progress(_stage: str, _cur: int, _tot: int, _msg: str) -> None:
    pass


def _micro_fade_in(audio: np.ndarray, sr: int, ms: float = 15.0) -> np.ndarray:
    n = min(audio.shape[0], int(sr * ms / 1000.0))
    if n <= 1:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, np.newaxis]
    out[:n] *= ramp
    return out


def _micro_fade_out(audio: np.ndarray, sr: int, ms: float = 30.0) -> np.ndarray:
    n = min(audio.shape[0], int(sr * ms / 1000.0))
    if n <= 1:
        return audio
    out = audio.copy()
    ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)[:, np.newaxis]
    out[-n:] *= ramp
    return out


def generate_dj_mix(
    file_paths: list[str | Path],
    user_context: str = "",
    style: str = "balanced",
    output_format: str = "mp3",
    use_stems_on_jumps: bool = True,
    allow_online_lookup: bool = True,
    progress_cb: ProgressCallback = _default_progress,
) -> dict:
    """Analyze tracks, curate setlist, render DJ transitions, and export final mix."""
    ensure_directories()
    init_db()

    valid_paths = [
        Path(p)
        for p in file_paths
        if Path(p).exists() and Path(p).suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if len(valid_paths) < 2:
        raise ValueError("Se necesitan al menos 2 canciones válidas para generar una mezcla.")

    # 1. MIR Analysis + SQLite Cache
    analyzed: list[TrackRecord] = []
    total = len(valid_paths)
    for idx, p in enumerate(valid_paths, start=1):
        progress_cb("analyze", idx, total, f"Analizando ({idx}/{total}): {p.name}")
        rec = analyze_track(str(p), allow_online_lookup=allow_online_lookup)
        analyzed.append(rec)

    # 2. Curation & Transition Planning
    progress_cb("curate", 1, 1, "Ordenando setlist por género, energía, BPM y Camelot...")
    prefs: MixPreferences = parse_user_context(user_context, style=style)
    prefs.use_stems_on_jumps = use_stems_on_jumps
    ordered = order_tracks(analyzed, prefs)
    plans: list[TrackPlan] = plan_mix(ordered, prefs)

    # 3. Render Audio & Transitions
    mix_chunks: list[np.ndarray] = []
    prev_tail_consumed_samples = 0

    for idx, plan in enumerate(plans):
        progress_cb(
            "render",
            idx + 1,
            len(plans),
            f"Mezclando ({idx + 1}/{len(plans)}): {plan.track.title or Path(plan.track.file_path).name}",
        )
        audio, sr = load_audio(plan.track.file_path, target_sr=SAMPLE_RATE)
        audio = normalize_lufs(audio, plan.track.lufs_integrated, TARGET_LUFS)

        s0 = int(plan.cue_in_sec * sr)
        s1 = min(audio.shape[0], int(plan.cue_out_sec * sr))
        track_seg = audio[s0:s1]

        # Skip the initial portion already mixed in the previous transition
        if idx > 0 and prev_tail_consumed_samples > 0:
            track_seg = track_seg[prev_tail_consumed_samples:]
        elif idx == 0 and plan.cue_in_sec > 0.05:
            track_seg = _micro_fade_in(track_seg, sr)

        if plan.transition_type is None or idx == len(plans) - 1:
            mix_chunks.append(_micro_fade_out(track_seg, sr, ms=500.0))
            break

        nxt_plan = plans[idx + 1]
        trans_samples = min(
            int(plan.transition_duration_sec * sr),
            max(sr * 2, track_seg.shape[0] // 2),
        )
        body = track_seg[:-trans_samples]
        out_seg = track_seg[-trans_samples:]

        # Load incoming segment of next track for the transition window
        nxt_audio, _ = load_audio(nxt_plan.track.file_path, target_sr=SAMPLE_RATE)
        nxt_audio = normalize_lufs(nxt_audio, nxt_plan.track.lufs_integrated, TARGET_LUFS)
        nxt_s0 = int(nxt_plan.cue_in_sec * sr)

        if plan.transition_type in ("beatmatch_bass_swap", "filter_sweep"):
            stretch_rate = float(np.clip(plan.track.bpm / max(plan.target_bpm_out, 1.0), 0.92, 1.08))
            raw_needed = int(trans_samples * stretch_rate) + int(sr * 0.5)
            raw_in = nxt_audio[nxt_s0 : nxt_s0 + raw_needed]
            in_seg = time_stretch_audio(raw_in, sr, rate=stretch_rate)[:trans_samples]
            prev_tail_consumed_samples = min(raw_needed, int(trans_samples * stretch_rate))
        else:
            in_seg = nxt_audio[nxt_s0 : nxt_s0 + trans_samples]
            prev_tail_consumed_samples = trans_samples

        if plan.transition_type == "beatmatch_bass_swap":
            trans_mix = render_beatmatch_bass_swap(out_seg, in_seg, sr=sr)
        elif plan.transition_type == "filter_sweep":
            trans_mix = render_filter_sweep(out_seg, in_seg, sr=sr)
        elif plan.transition_type == "reverb_wash_drop":
            trans_mix = render_reverb_wash_drop(out_seg, in_seg, sr=sr)
        elif plan.transition_type == "vinyl_brake_drop":
            trans_mix = render_vinyl_brake_drop(out_seg, in_seg, sr=sr)
        elif plan.transition_type == "stem_bridge":
            trans_mix = render_stem_bridge(out_seg, in_seg, out_bpm=plan.track.bpm, sr=sr)
        else:
            trans_mix = render_echo_out(out_seg, in_seg, out_bpm=plan.track.bpm, sr=sr)

        if body.shape[0] > 0:
            mix_chunks.append(body)
        mix_chunks.append(trans_mix)

    progress_cb("master", 1, 1, "Aplicando limitador Master y exportando mezcla...")
    full_mix = np.vstack(mix_chunks)
    full_mix = apply_master_limiter(full_mix, sr=SAMPLE_RATE)

    session_id = uuid.uuid4().hex[:10]
    ext = "wav" if output_format.lower() == "wav" else "mp3"
    out_path = OUTPUT_DIR / f"dj_ai_mix_{session_id}.{ext}"
    export_audio(full_mix, out_path, sr=SAMPLE_RATE)

    total_duration_sec = round(float(full_mix.shape[0] / SAMPLE_RATE), 2)
    tracklist_summary = [
        {
            "order": i + 1,
            "title": p.track.title,
            "artist": p.track.artist,
            "genre": p.track.genre,
            "bpm": p.track.bpm,
            "camelot_key": p.track.camelot_key,
            "cue_in_sec": p.cue_in_sec,
            "cue_out_sec": p.cue_out_sec,
            "transition_type": p.transition_type,
            "transition_duration_sec": p.transition_duration_sec,
            "reason": p.reason,
        }
        for i, p in enumerate(plans)
    ]

    save_mix_session(
        session_id=session_id,
        user_context=user_context,
        target_style=prefs.style,
        output_file_path=str(out_path),
        total_duration_sec=total_duration_sec,
        tracklist=tracklist_summary,
    )

    progress_cb("done", 1, 1, "¡Mezcla completada!")
    return {
        "session_id": session_id,
        "output_file": str(out_path),
        "output_filename": out_path.name,
        "total_duration_sec": total_duration_sec,
        "tracklist": tracklist_summary,
    }
