"""FastAPI backend and job runner for the DJ-AI web interface."""

from __future__ import annotations

from contextlib import asynccontextmanager
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from dj_ai.audio_io import SUPPORTED_EXTENSIONS
from dj_ai.config import OUTPUT_DIR, UPLOAD_DIR, ensure_directories
from dj_ai.db import init_db
from dj_ai.engine import generate_dj_mix


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    ensure_directories()
    init_db()
    yield


app = FastAPI(title="DJ-AI Autonomous Mixer", version="0.1.0", lifespan=_lifespan)

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


def _run_mix_job(
    job_id: str,
    file_paths: list[Path],
    user_context: str,
    style: str,
    output_format: str,
    use_stems: bool,
) -> None:
    def _progress(stage: str, cur: int, tot: int, msg: str) -> None:
        with _JOBS_LOCK:
            if job_id in _JOBS:
                pct = int((cur / max(1, tot)) * 100)
                _JOBS[job_id].update(
                    {
                        "stage": stage,
                        "current": cur,
                        "total": tot,
                        "percent": pct,
                        "message": msg,
                    }
                )

    try:
        result = generate_dj_mix(
            file_paths=file_paths,
            user_context=user_context,
            style=style,
            output_format=output_format,
            use_stems_on_jumps=use_stems,
            progress_cb=_progress,
        )
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "completed"
            _JOBS[job_id]["percent"] = 100
            _JOBS[job_id]["result"] = result
    except Exception as exc:
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["message"] = str(exc)


@app.post("/api/mix")
async def create_mix(
    files: list[UploadFile] = File(default=[]),
    folder_path: str = Form(default=""),
    user_context: str = Form(default=""),
    style: str = Form(default="balanced"),
    output_format: str = Form(default="mp3"),
    use_stems: bool = Form(default=True),
) -> dict[str, str]:
    """Start an asynchronous DJ mix generation job from uploaded files or a local folder."""
    ensure_directories()
    job_id = uuid.uuid4().hex[:12]
    target_paths: list[Path] = []

    if folder_path.strip():
        folder = Path(folder_path.strip())
        if not folder.exists() or not folder.is_dir():
            raise HTTPException(status_code=400, detail="La carpeta especificada no existe.")
        for item in sorted(folder.iterdir()):
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                target_paths.append(item)

    if files:
        job_upload_dir = UPLOAD_DIR / job_id
        job_upload_dir.mkdir(parents=True, exist_ok=True)
        for uf in files:
            if not uf.filename:
                continue
            ext = Path(uf.filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            dest = job_upload_dir / Path(uf.filename).name
            with dest.open("wb") as f:
                shutil.copyfileobj(uf.file, f)
            target_paths.append(dest)

    if len(target_paths) < 2:
        raise HTTPException(
            status_code=400,
            detail="Debes seleccionar al menos 2 canciones válidas (MP3, M4A, WAV o FLAC).",
        )

    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "running",
            "stage": "init",
            "current": 0,
            "total": len(target_paths),
            "percent": 0,
            "message": "Iniciando motor DJ-AI...",
            "result": None,
        }

    thread = threading.Thread(
        target=_run_mix_job,
        args=(job_id, target_paths, user_context, style, output_format, use_stems),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
        return dict(job)


@app.get("/api/mixes/{filename}")
def download_mix(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    file_path = OUTPUT_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Archivo de mezcla no encontrado.")
    return FileResponse(path=str(file_path), filename=safe_name)
