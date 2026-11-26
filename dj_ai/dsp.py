"""Digital Signal Processing (DSP) engine for DJ transitions, EQ, FX, and mastering."""

from __future__ import annotations

import importlib.util
import logging
from typing import Literal

import numpy as np
from scipy import signal

from dj_ai.config import SAMPLE_RATE, TARGET_LUFS

logger = logging.getLogger(__name__)

TransitionType = Literal[
    "beatmatch_bass_swap",
    "filter_sweep",
    "echo_out",
    "reverb_wash_drop",
    "vinyl_brake_drop",
    "stem_bridge",
]


def is_demucs_available() -> bool:
    """Return True if torch and demucs are installed in the environment."""
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("demucs") is not None
    )


def normalize_lufs(
    audio: np.ndarray,
    current_lufs: float,
    target_lufs: float = TARGET_LUFS,
) -> np.ndarray:
    """Scale audio to target integrated LUFS with headroom protection."""
    gain_db = float(np.clip(target_lufs - current_lufs, -18.0, 18.0))
    gain_lin = 10.0 ** (gain_db / 20.0)
    scaled = audio * gain_lin
    peak = float(np.max(np.abs(scaled)))
    if peak > 0.92:
        scaled = scaled * (0.92 / peak)
    return scaled.astype(np.float32)


def time_stretch_audio(
    audio: np.ndarray,
    sr: int,
    rate: float,
) -> np.ndarray:
    """Time-stretch stereo audio without altering pitch.

    rate > 1.0 speeds up (shorter duration); rate < 1.0 slows down.
    """
    if abs(rate - 1.0) < 0.002 or audio.shape[0] < 1024:
        return audio

    try:
        from pedalboard import time_stretch

        stretched = time_stretch(
            np.ascontiguousarray(audio.T, dtype=np.float32),
            sample_rate=float(sr),
            stretch_factor=float(rate),
            high_quality=True,
        )
        return np.ascontiguousarray(stretched.T, dtype=np.float32)
    except Exception as exc:
        logger.warning("pedalboard.time_stretch failed (%s); falling back to librosa.", exc)
        import librosa

        ch0 = librosa.effects.time_stretch(audio[:, 0], rate=rate)
        ch1 = librosa.effects.time_stretch(audio[:, 1], rate=rate)
        min_len = min(len(ch0), len(ch1))
        return np.column_stack((ch0[:min_len], ch1[:min_len])).astype(np.float32)


def _s_curve(n: int) -> np.ndarray:
    """Smooth cosine S-curve from 0.0 to 1.0."""
    t = np.linspace(0.0, np.pi, max(1, n), dtype=np.float32)
    return 0.5 * (1.0 - np.cos(t))


def _split_3band(
    audio: np.ndarray,
    sr: int,
    low_cut: float = 240.0,
    high_cut: float = 3800.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-phase 3-band crossover split (low, mid, high)."""
    sos_low = signal.butter(2, low_cut, btype="lowpass", fs=sr, output="sos")
    sos_high = signal.butter(2, high_cut, btype="highpass", fs=sr, output="sos")

    low = signal.sosfiltfilt(sos_low, audio, axis=0)
    high = signal.sosfiltfilt(sos_high, audio, axis=0)
    mid = audio - low - high
    return low.astype(np.float32), mid.astype(np.float32), high.astype(np.float32)


def _smooth_filter_sweep(
    audio: np.ndarray,
    sr: int,
    btype: Literal["highpass", "lowpass"],
    cutoff_hz: float,
    fade_in_effect: bool,
) -> np.ndarray:
    """Click-free progressive filter blend (avoids biquad coefficient zipper noise)."""
    sos = signal.butter(2, cutoff_hz, btype=btype, fs=sr, output="sos")
    filtered = signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)
    mix = _s_curve(audio.shape[0])[:, np.newaxis]
    if fade_in_effect:
        # Transitions smoothly from dry audio -> filtered audio
        return (audio * (1.0 - mix) + filtered * mix).astype(np.float32)
    # Transitions smoothly from filtered audio -> dry audio
    return (filtered * (1.0 - mix) + audio * mix).astype(np.float32)


def render_beatmatch_bass_swap(
    out_seg: np.ndarray,
    in_seg: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Mix two beat-aligned segments with early ducking on A and a clean Bass Swap."""
    n = min(out_seg.shape[0], in_seg.shape[0])
    out_seg = out_seg[:n]
    in_seg = in_seg[:n]

    low_a, mid_a, high_a = _split_3band(out_seg, sr)
    low_b, mid_b, high_b = _split_3band(in_seg, sr)

    # Outgoing A ducks noticeably earlier so B enters without clutter
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, np.newaxis]
    fade_out_mid = np.clip(1.0 - (t ** 0.85), 0.0, 1.0)
    fade_in_mid = np.sqrt(_s_curve(n))[:, np.newaxis]

    # Equal-power headroom scaling during overlap
    mid_mix = (mid_a * fade_out_mid + mid_b * fade_in_mid) * 0.88
    high_mix = (high_a * fade_out_mid + high_b * fade_in_mid) * 0.88

    # Bass swap around 42%-54%
    swap_start = int(n * 0.42)
    swap_end = int(n * 0.54)
    swap_len = max(1, swap_end - swap_start)

    bass_gain_a = np.ones((n, 1), dtype=np.float32)
    bass_gain_b = np.zeros((n, 1), dtype=np.float32)
    swap_curve = _s_curve(swap_len)[:, np.newaxis]

    bass_gain_a[swap_start:swap_end] = 1.0 - swap_curve
    bass_gain_a[swap_end:] = 0.0
    bass_gain_b[swap_start:swap_end] = swap_curve
    bass_gain_b[swap_end:] = 1.0

    low_mix = low_a * bass_gain_a + low_b * bass_gain_b
    return (low_mix + mid_mix + high_mix).astype(np.float32)


def render_filter_sweep(
    out_seg: np.ndarray,
    in_seg: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Click-free HPF sweep out on outgoing track while LPF opens on incoming track."""
    n = min(out_seg.shape[0], in_seg.shape[0])
    out_seg = out_seg[:n]
    in_seg = in_seg[:n]

    swept_out = _smooth_filter_sweep(out_seg, sr, "highpass", cutoff_hz=1200.0, fade_in_effect=True)
    swept_in = _smooth_filter_sweep(in_seg, sr, "lowpass", cutoff_hz=850.0, fade_in_effect=False)

    curve = _s_curve(n)[:, np.newaxis]
    # Outgoing fades down earlier (power 0.8) while incoming rises smoothly
    gain_a = np.sqrt(1.0 - curve) * 0.90
    gain_b = np.sqrt(curve) * 0.92
    return (swept_out * gain_a + swept_in * gain_b).astype(np.float32)


def render_echo_out(
    out_seg: np.ndarray,
    in_seg: np.ndarray,
    out_bpm: float,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply tempo-synced echo tail on outgoing track over clean entry of incoming track."""
    n = min(out_seg.shape[0], in_seg.shape[0])
    out_seg = out_seg[:n]
    in_seg = in_seg[:n]

    beat_sec = 60.0 / max(out_bpm, 70.0)
    delay_sec = float(np.clip(beat_sec * 0.75, 0.22, 0.62))
    delay_samples = int(delay_sec * sr)

    # Dry outgoing track lowers volume early (first 38% of window)
    dry_cut = int(n * 0.38)
    dry_env = np.zeros((n, 1), dtype=np.float32)
    dry_env[:dry_cut] = (1.0 - _s_curve(dry_cut))[:, np.newaxis]

    source = out_seg * dry_env
    wet = np.zeros_like(out_seg)
    feedback = 0.42
    for tap in range(1, 6):
        shift = tap * delay_samples
        if shift >= n:
            break
        wet[shift:] += source[: n - shift] * (feedback ** tap)

    sos_hp = signal.butter(2, 320.0, btype="highpass", fs=sr, output="sos")
    wet = signal.sosfiltfilt(sos_hp, wet, axis=0).astype(np.float32)

    # Incoming track enters smoothly from 28% with a 1.0s ramp
    in_start = int(n * 0.28)
    in_env = np.zeros((n, 1), dtype=np.float32)
    ramp_len = min(int(sr * 1.2), n - in_start)
    if ramp_len > 0:
        in_env[in_start : in_start + ramp_len] = _s_curve(ramp_len)[:, np.newaxis]
        in_env[in_start + ramp_len :] = 1.0

    # Fade wet echo down as incoming track reaches full volume
    wet_duck = 1.0 - 0.65 * in_env
    return (source * 0.90 + wet * 0.65 * wet_duck + in_seg * in_env * 0.95).astype(np.float32)


def render_reverb_wash_drop(
    out_seg: np.ndarray,
    in_seg: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Wash outgoing track in smooth reverb + high-pass fade, then drop incoming track."""
    n = min(out_seg.shape[0], in_seg.shape[0])
    out_seg = out_seg[:n]
    in_seg = in_seg[:n]

    try:
        from pedalboard import HighpassFilter, Pedalboard, Reverb

        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=280.0),
            Reverb(room_size=0.78, damping=0.45, wet_level=0.65, dry_level=0.35, width=1.0),
        ])
        washed = board(np.ascontiguousarray(out_seg.T, dtype=np.float32), sample_rate=float(sr)).T
        washed = np.ascontiguousarray(washed, dtype=np.float32)
    except Exception:
        washed = _smooth_filter_sweep(out_seg, sr, "highpass", 600.0, fade_in_effect=True)

    # Outgoing fades down to 0 by 60% of the window
    fade_out_end = int(n * 0.62)
    out_env = np.zeros((n, 1), dtype=np.float32)
    out_env[:fade_out_end] = (1.0 - _s_curve(fade_out_end))[:, np.newaxis]

    # Incoming fades in from 38% over 1.2s
    in_start = int(n * 0.38)
    in_env = np.zeros((n, 1), dtype=np.float32)
    ramp = min(int(sr * 1.2), n - in_start)
    if ramp > 0:
        in_env[in_start : in_start + ramp] = _s_curve(ramp)[:, np.newaxis]
        in_env[in_start + ramp :] = 1.0

    return (washed * out_env * 0.85 + in_seg * in_env * 0.95).astype(np.float32)


def render_vinyl_brake_drop(
    out_seg: np.ndarray,
    in_seg: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Turntable slowdown brake on outgoing track right before incoming track enters."""
    n = min(out_seg.shape[0], in_seg.shape[0])
    out_seg = out_seg[:n]
    in_seg = in_seg[:n]

    # Play outgoing track normally for first 40%, then brake over ~1.4s
    drop_point = int(n * 0.55)
    brake_len = min(int(sr * 1.4), drop_point)
    brake_start = drop_point - brake_len

    out_processed = np.zeros_like(out_seg)
    out_processed[:brake_start] = out_seg[:brake_start]

    if brake_len > 16:
        # Non-linear time-warping curve simulating turntable platter stopping
        u = np.linspace(0.0, 1.0, brake_len, dtype=np.float64)
        src_pos = brake_start + (u - 0.48 * (u ** 2)) * (brake_len * 0.7)
        src_pos = np.clip(src_pos, 0, n - 2)
        idx0 = np.floor(src_pos).astype(np.int64)
        frac = (src_pos - idx0)[:, np.newaxis].astype(np.float32)
        braked = out_seg[idx0] * (1.0 - frac) + out_seg[idx0 + 1] * frac
        fade = (1.0 - _s_curve(brake_len))[:, np.newaxis]
        out_processed[brake_start:drop_point] = braked * fade

    # Incoming track enters smoothly right at drop_point (with 80ms de-click)
    in_env = np.zeros((n, 1), dtype=np.float32)
    ramp = min(int(sr * 0.15), n - drop_point)
    if ramp > 0:
        in_env[drop_point : drop_point + ramp] = _s_curve(ramp)[:, np.newaxis]
        in_env[drop_point + ramp :] = 1.0

    return (out_processed * 0.92 + in_seg * in_env * 0.95).astype(np.float32)


def render_stem_bridge(
    out_seg: np.ndarray,
    in_seg: np.ndarray,
    out_bpm: float,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Use selective Demucs stem separation on the transition window (GPU/CPU)."""
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model

        n = min(out_seg.shape[0], in_seg.shape[0])
        out_seg = out_seg[:n]
        in_seg = in_seg[:n]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model("htdemucs")
        model.to(device)
        model.eval()

        def _separate(seg: np.ndarray) -> dict[str, np.ndarray]:
            tensor = torch.from_numpy(seg.T).unsqueeze(0).to(device)
            with torch.no_grad():
                if device == "cuda":
                    with torch.amp.autocast("cuda"):
                        sources = apply_model(model, tensor, shifts=0, split=True, overlap=0.25)
                else:
                    sources = apply_model(model, tensor, shifts=0, split=True, overlap=0.25)
            stems_np = sources.squeeze(0).cpu().numpy()
            return {
                name: np.ascontiguousarray(stems_np[i].T, dtype=np.float32)
                for i, name in enumerate(model.sources)
            }

        stems_a = _separate(out_seg)
        stems_b = _separate(in_seg)

        if device == "cuda":
            torch.cuda.empty_cache()

        early_cut = int(n * 0.38)
        early_out = np.zeros((n, 1), dtype=np.float32)
        early_out[:early_cut] = (1.0 - _s_curve(early_cut))[:, np.newaxis]

        vocal_out = (1.0 - _s_curve(n)[:, np.newaxis]) * 0.85

        mix_a = (
            (stems_a["drums"] + stems_a["bass"]) * early_out
            + (stems_a["vocals"] + stems_a["other"]) * vocal_out
        )
        b_rhythm_start = int(n * 0.25)
        b_rhythm_env = np.zeros((n, 1), dtype=np.float32)
        b_rhythm_env[b_rhythm_start:] = _s_curve(n - b_rhythm_start)[:, np.newaxis]

        b_mel_start = int(n * 0.55)
        b_mel_env = np.zeros((n, 1), dtype=np.float32)
        b_mel_env[b_mel_start:] = _s_curve(n - b_mel_start)[:, np.newaxis]

        mix_b = (
            (stems_b["drums"] + stems_b["bass"]) * b_rhythm_env
            + (stems_b["vocals"] + stems_b["other"]) * b_mel_env
        )
        return (mix_a + mix_b).astype(np.float32)
    except Exception as exc:
        logger.info("Stem bridge unavailable (%s); using reverb_wash_drop fallback.", exc)
        return render_reverb_wash_drop(out_seg, in_seg, sr=sr)


def apply_master_limiter(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply gentle master bus lookahead limiter to prevent digital clipping."""
    try:
        from pedalboard import Limiter

        limiter = Limiter(threshold_db=-1.5, release_ms=150.0)
        processed = limiter(np.ascontiguousarray(audio.T, dtype=np.float32), sample_rate=float(sr))
        return np.clip(processed.T, -0.98, 0.98).astype(np.float32)
    except Exception:
        return np.tanh(audio * 0.92).astype(np.float32)
