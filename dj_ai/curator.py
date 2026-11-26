"""Curation engine: user context parsing, setlist ordering, cue planning, and transition selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dj_ai.config import GENRE_PROFILES, MAX_BPM_STRETCH_RATIO
from dj_ai.db import SectionRecord, TrackRecord
from dj_ai.dsp import TransitionType, is_demucs_available


@dataclass
class MixPreferences:
    """User-configurable mixing and pacing preferences."""

    style: str = "balanced"  # 'balanced', 'quick_hype', 'dance_extended'
    energy_curve: str = "ascending"  # 'ascending', 'peak', 'wave'
    user_context: str = ""
    use_stems_on_jumps: bool = True
    genre_order_hint: list[str] | None = None


@dataclass
class TrackPlan:
    """Planned playback window and outgoing transition for a track in the mix."""

    track: TrackRecord
    cue_in_sec: float
    cue_out_sec: float
    transition_type: TransitionType | None
    transition_duration_sec: float
    target_bpm_out: float
    reason: str


def parse_user_context(user_context: str, style: str = "balanced") -> MixPreferences:
    """Parse free-text user instructions into deterministic MixPreferences."""
    text = user_context.lower().strip()
    resolved_style = style

    if any(w in text for w in ("rápido", "rapido", "coros", "viral", "hype", "corto")):
        resolved_style = "quick_hype"
    elif any(w in text for w in ("bailable", "largo", "pista", "fiesta larga", "completo")):
        resolved_style = "dance_extended"

    energy_curve = "ascending"
    if any(w in text for w in ("pico", "fuerte", "siempre arriba", "alto", "intenso")):
        energy_curve = "peak"
    elif any(w in text for w in ("ola", "sube y baja", "dinámico", "dinamico", "variado")):
        energy_curve = "wave"

    # Detect explicit genre progression hints if mentioned in order
    genre_aliases = [
        ("salsa", ("salsa", "tropical")),
        ("pop", ("pop",)),
        ("reggaeton", ("reggaeton", "reggaetón", "urbano")),
        ("dancehall", ("dancehall",)),
        ("electronic", ("electrónica", "electronica", "edm", "house", "techno")),
    ]
    found_genres: list[tuple[int, str]] = []
    for canonical, aliases in genre_aliases:
        positions = [text.find(a) for a in aliases if text.find(a) != -1]
        if positions:
            found_genres.append((min(positions), canonical))
    found_genres.sort(key=lambda x: x[0])
    genre_order = [g for _, g in found_genres] if len(found_genres) >= 2 else None

    return MixPreferences(
        style=resolved_style,
        energy_curve=energy_curve,
        user_context=user_context,
        use_stems_on_jumps=True,
        genre_order_hint=genre_order,
    )


def _parse_camelot(key: str) -> tuple[int, str]:
    key = key.strip().upper()
    if len(key) < 2 or key[-1] not in ("A", "B"):
        return 8, "A"
    try:
        num = int(key[:-1])
        return max(1, min(12, num)), key[-1]
    except ValueError:
        return 8, "A"


def camelot_score(key_a: str, key_b: str) -> float:
    """Return harmonic compatibility score in [0.15, 1.0] on the Camelot Wheel."""
    num_a, let_a = _parse_camelot(key_a)
    num_b, let_b = _parse_camelot(key_b)
    diff = min((num_a - num_b) % 12, (num_b - num_a) % 12)

    if num_a == num_b and let_a == let_b:
        return 1.0
    if let_a == let_b and diff == 1:
        return 0.90
    if num_a == num_b and let_a != let_b:
        return 0.85
    if let_a == let_b and diff in (2, 5):  # +2 energy boost or +7/-5
        return 0.70
    if diff == 1 and let_a != let_b:
        return 0.50
    return 0.20


_GENRE_AFFINITY: dict[tuple[str, str], float] = {
    ("reggaeton", "dancehall"): 0.85,
    ("reggaeton", "trap"): 0.78,
    ("reggaeton", "latin"): 0.82,
    ("reggaeton", "pop"): 0.68,
    ("reggaeton", "hiphop"): 0.75,
    ("trap", "hiphop"): 0.90,
    ("trap", "dancehall"): 0.72,
    ("trap", "electronic"): 0.65,
    ("trap", "pop"): 0.65,
    ("dancehall", "latin"): 0.72,
    ("dancehall", "pop"): 0.62,
    ("salsa", "latin"): 0.85,
    ("salsa", "dancehall"): 0.68,
    ("salsa", "reggaeton"): 0.50,
    ("banda", "latin"): 0.65,
    ("banda", "reggaeton"): 0.45,
    ("pop", "electronic"): 0.74,
    ("pop", "latin"): 0.68,
    ("electronic", "hiphop"): 0.42,
}


def genre_affinity(genre_a: str, genre_b: str) -> float:
    """Return genre transition affinity in [0.2, 1.0]."""
    if genre_a == genre_b:
        return 1.0
    if "unknown" in (genre_a, genre_b):
        return 0.60
    pair = (genre_a, genre_b)
    rev = (genre_b, genre_a)
    return _GENRE_AFFINITY.get(pair, _GENRE_AFFINITY.get(rev, 0.25))


def bpm_compatibility(bpm_a: float, bpm_b: float) -> float:
    """Return BPM proximity score, favoring <= ±8% delta or clean half/double."""
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.5
    ratio = abs(bpm_b - bpm_a) / bpm_a
    if ratio <= MAX_BPM_STRETCH_RATIO:
        return float(1.0 - (ratio / MAX_BPM_STRETCH_RATIO) * 0.35)
    # Check half/double time compatibility
    if abs(bpm_b * 2.0 - bpm_a) / bpm_a <= MAX_BPM_STRETCH_RATIO:
        return 0.65
    if abs(bpm_b * 0.5 - bpm_a) / bpm_a <= MAX_BPM_STRETCH_RATIO:
        return 0.65
    return float(max(0.1, 0.5 - ratio))


def _target_energy_at(pos: float, curve: str) -> float:
    """Expected normalized energy at setlist position pos in [0.0, 1.0]."""
    if curve == "peak":
        return 0.75 + 0.15 * np.sin(np.pi * pos)
    if curve == "wave":
        return 0.55 + 0.30 * np.sin(2.0 * np.pi * pos)
    # Default: ascending warm-up -> peak
    return 0.35 + 0.55 * pos


def order_tracks(tracks: list[TrackRecord], prefs: MixPreferences) -> list[TrackRecord]:
    """Order tracks by genre grouping, energy curve, BPM proximity, and Camelot key."""
    if len(tracks) <= 2:
        return list(tracks)

    n = len(tracks)

    def _genre_rank(g: str) -> int:
        if prefs.genre_order_hint and g in prefs.genre_order_hint:
            return prefs.genre_order_hint.index(g)
        return 99

    # Pick starting track: lowest energy (for ascending) or earliest hinted genre
    start_idx = min(
        range(n),
        key=lambda i: (_genre_rank(tracks[i].genre), tracks[i].energy_avg),
    )

    remaining = set(range(n))
    remaining.remove(start_idx)
    order = [start_idx]

    for step in range(1, n):
        pos = step / max(1, n - 1)
        target_e = _target_energy_at(pos, prefs.energy_curve)
        prev = tracks[order[-1]]

        best_idx = -1
        best_score = -1e9
        for cand_idx in remaining:
            cand = tracks[cand_idx]
            g_score = genre_affinity(prev.genre, cand.genre)
            e_score = 1.0 - abs(cand.energy_avg - target_e)
            b_score = bpm_compatibility(prev.bpm, cand.bpm)
            k_score = camelot_score(prev.camelot_key, cand.camelot_key)

            hint_bonus = 0.0
            if prefs.genre_order_hint:
                r_prev = _genre_rank(prev.genre)
                r_cand = _genre_rank(cand.genre)
                if r_cand < r_prev:
                    hint_bonus = -0.35
                elif r_cand == r_prev or r_cand == r_prev + 1:
                    hint_bonus = 0.15

            total = (
                0.35 * g_score
                + 0.25 * e_score
                + 0.25 * b_score
                + 0.15 * k_score
                + hint_bonus
            )
            if total > best_score:
                best_score = total
                best_idx = cand_idx

        order.append(best_idx)
        remaining.remove(best_idx)

    return [tracks[i] for i in order]


def _snap_to_beat(target_sec: float, beats: list[float]) -> float:
    if not beats:
        return target_sec
    return float(min(beats, key=lambda b: abs(b - target_sec)))


def _choose_cue_window(track: TrackRecord, prefs: MixPreferences) -> tuple[float, float, str]:
    """Select balanced Cue-In and Cue-Out timestamps respecting genre danceability and climax."""
    profile = GENRE_PROFILES.get(track.genre, GENRE_PROFILES["unknown"])
    dur = track.duration_sec

    if dur <= 45.0:
        return 0.0, max(0.0, dur - 2.0), "Pista corta: reproducción casi completa"

    min_len = profile.min_play_sec
    max_len = profile.max_play_sec
    if prefs.style == "quick_hype":
        min_len *= 0.65
        max_len *= 0.65
    elif prefs.style == "dance_extended":
        min_len *= 1.15
        max_len *= 1.20

    climax_sections: list[SectionRecord] = [s for s in track.sections if s.is_climax]
    main_climax = (
        max(climax_sections, key=lambda s: s.energy_score)
        if climax_sections
        else SectionRecord("chorus_drop", dur * 0.35, dur * 0.65, track.energy_avg, True)
    )

    # Cue-in: enter well BEFORE the climax (intro/verse/buildup) so transitions never slam into the chorus
    intro_sec = next((s for s in track.sections if s.label == "intro"), None)
    if prefs.style == "quick_hype" or profile.prefer_climax_focus:
        # Start ~24-28s before climax (catching verse/buildup)
        ideal_in = max(0.0, main_climax.start_sec - 26.0)
        if intro_sec and ideal_in > intro_sec.end_sec:
            ideal_in = max(intro_sec.start_sec, min(ideal_in, intro_sec.end_sec * 0.6))
        cue_in = ideal_in
        ideal_out = min(dur - 4.0, max(main_climax.end_sec + 18.0, cue_in + min_len))
        cue_out = min(ideal_out, cue_in + max_len)
        reason = "Entrada temprana en verso/build-up + clímax + salida suave"
    else:
        # Danceable genres (salsa, reggaeton, dancehall, banda): let the groove breathe from early intro/verse
        cue_in = min(6.0, intro_sec.end_sec * 0.25) if intro_sec else 0.0
        target_end = max(main_climax.end_sec + 22.0, cue_in + min_len)
        cue_out = min(dur - 4.0, target_end, cue_in + max_len)
        reason = "Duración bailable extendida respetando el groove"

    if cue_out - cue_in < 32.0:
        cue_in = max(0.0, main_climax.start_sec - 22.0)
        cue_out = min(dur - 2.0, cue_in + 55.0)

    cue_in = _snap_to_beat(cue_in, track.beat_positions)
    cue_out = _snap_to_beat(cue_out, track.beat_positions)
    if cue_out <= cue_in + 20.0:
        cue_in = 0.0
        cue_out = max(20.0, dur - 2.0)

    return round(cue_in, 3), round(cue_out, 3), reason


def _best_matched_bpm(bpm_a: float, bpm_b: float) -> tuple[float, float]:
    """Return (effective_bpm_b, relative_delta) considering 1x, 2x, and 0.5x tempo."""
    candidates = [bpm_b, bpm_b * 2.0, bpm_b * 0.5]
    best_bpm = min(candidates, key=lambda c: abs(c - bpm_a) / max(bpm_a, 1e-6))
    delta = abs(best_bpm - bpm_a) / max(bpm_a, 1e-6)
    return best_bpm, delta


def plan_mix(ordered_tracks: list[TrackRecord], prefs: MixPreferences) -> list[TrackPlan]:
    """Create full Cue-In/Cue-Out and transition strategy plan for an ordered setlist."""
    plans: list[TrackPlan] = []
    n = len(ordered_tracks)
    demucs_ready = is_demucs_available()
    prev_transition: TransitionType | None = None

    for idx, track in enumerate(ordered_tracks):
        cue_in, cue_out, reason = _choose_cue_window(track, prefs)
        if idx == n - 1:
            plans.append(
                TrackPlan(
                    track=track,
                    cue_in_sec=cue_in,
                    cue_out_sec=min(track.duration_sec, cue_out + 8.0),
                    transition_type=None,
                    transition_duration_sec=0.0,
                    target_bpm_out=track.bpm,
                    reason=reason,
                )
            )
            continue

        nxt = ordered_tracks[idx + 1]
        matched_nxt_bpm, bpm_delta = _best_matched_bpm(track.bpm, nxt.bpm)
        both_stable = track.is_tempo_stable and nxt.is_tempo_stable
        g_aff = genre_affinity(track.genre, nxt.genre)

        profile = GENRE_PROFILES.get(track.genre, GENRE_PROFILES["unknown"])
        bars = profile.default_transition_bars
        beat_sec = 60.0 / max(track.bpm, 75.0)

        if both_stable and bpm_delta <= MAX_BPM_STRETCH_RATIO and g_aff >= 0.45:
            if g_aff >= 0.70 and prev_transition != "beatmatch_bass_swap":
                trans_type: TransitionType = "beatmatch_bass_swap"
            else:
                trans_type = "filter_sweep" if prev_transition != "filter_sweep" else "beatmatch_bass_swap"
            trans_dur = float(np.clip(bars * 2 * beat_sec, 8.0, 18.0))
            target_bpm = matched_nxt_bpm
        else:
            # Diverse non-beatmatched transitions for >8% BPM jumps or abrupt genre changes
            jump_pool: list[TransitionType] = ["reverb_wash_drop", "echo_out", "vinyl_brake_drop"]
            if prefs.use_stems_on_jumps and demucs_ready:
                jump_pool.insert(0, "stem_bridge")
            candidates = [t for t in jump_pool if t != prev_transition] or jump_pool
            if nxt.energy_avg > track.energy_avg + 0.12 and "vinyl_brake_drop" in candidates:
                trans_type = "vinyl_brake_drop"
            else:
                trans_type = candidates[idx % len(candidates)]
            trans_dur = float(np.clip(4 * 4 * beat_sec, 7.0, 13.0))
            target_bpm = track.bpm

        prev_transition = trans_type
        max_allowed = max(4.0, (cue_out - cue_in) * 0.35)
        trans_dur = round(min(trans_dur, max_allowed), 3)

        plans.append(
            TrackPlan(
                track=track,
                cue_in_sec=cue_in,
                cue_out_sec=cue_out,
                transition_type=trans_type,
                transition_duration_sec=trans_dur,
                target_bpm_out=round(target_bpm, 2),
                reason=reason,
            )
        )

    return plans
