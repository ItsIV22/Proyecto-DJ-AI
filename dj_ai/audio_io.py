"""Audio decoding, content hashing, tag reading, and export utilities."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from mutagen import File as MutagenFile

from dj_ai.config import CHANNELS, SAMPLE_RATE

SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac"}


def get_ffmpeg_exe() -> str:
    """Resolve ffmpeg executable from PATH or imageio-ffmpeg."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is not available. Install imageio-ffmpeg or add ffmpeg to PATH."
        ) from exc


def load_audio(
    file_path: str | Path,
    target_sr: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> tuple[np.ndarray, int]:
    """Decode an audio file to float32 stereo PCM with shape (n_samples, channels)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Fast path for WAV/FLAC already at target sample rate
    if path.suffix.lower() in {".wav", ".flac"}:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            if sr == target_sr:
                if data.shape[1] == 1 and channels == 2:
                    data = np.repeat(data, 2, axis=1)
                elif data.shape[1] > channels:
                    data = data[:, :channels]
                return np.ascontiguousarray(data, dtype=np.float32), sr
        except Exception:
            pass

    ffmpeg_exe = get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed decoding '{path.name}': {err}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Decoded audio is empty for '{path.name}'")

    n_frames = audio.size // channels
    audio = audio[: n_frames * channels].reshape(n_frames, channels)
    return np.ascontiguousarray(audio, dtype=np.float32), target_sr


def compute_audio_hash(audio: np.ndarray) -> str:
    """Compute a deterministic SHA-256 hash of decoded audio PCM content.

    Uses quantized 16-bit representation sampled deterministically so that
    renaming, moving, or editing ID3 tags does not invalidate the MIR cache.
    """
    if audio.size == 0:
        return hashlib.sha256(b"empty").hexdigest()

    # Quantize to int16 to avoid tiny float decoder differences across platforms
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    step = max(1, pcm16.shape[0] // 200_000)
    sampled = np.ascontiguousarray(pcm16[::step])
    hasher = hashlib.sha256()
    hasher.update(str(audio.shape[0]).encode("utf-8"))
    hasher.update(sampled.tobytes())
    return hasher.hexdigest()


def read_metadata(file_path: str | Path) -> dict[str, str]:
    """Extract title, artist, and genre from audio tags with filename fallback."""
    path = Path(file_path)
    title = ""
    artist = ""
    genre = ""

    try:
        tags = MutagenFile(str(path), easy=True)
        if tags:
            title = str(tags.get("title", [""])[0]).strip()
            artist = str(tags.get("artist", [""])[0]).strip()
            genre = str(tags.get("genre", [""])[0]).strip()
    except Exception:
        pass

    if not title:
        stem = path.stem.strip()
        if " - " in stem:
            parts = stem.split(" - ", 1)
            if not artist:
                artist = parts[0].strip()
            title = parts[1].strip()
        else:
            title = stem

    return {"title": title, "artist": artist, "genre": genre}


def export_audio(
    audio: np.ndarray,
    output_path: str | Path,
    sr: int = SAMPLE_RATE,
    bitrate: str = "320k",
) -> Path:
    """Export float32 stereo audio to WAV or MP3."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    safe_audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    if out.suffix.lower() == ".wav":
        sf.write(str(out), safe_audio, sr, subtype="PCM_16")
        return out

    ffmpeg_exe = get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        str(safe_audio.shape[1]),
        "-i",
        "pipe:0",
        "-b:a",
        bitrate,
        str(out),
    ]
    proc = subprocess.run(
        cmd,
        input=np.ascontiguousarray(safe_audio).tobytes(),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed exporting '{out.name}': {err}")
    return out
