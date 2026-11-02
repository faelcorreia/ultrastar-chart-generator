"""REST API (no auth): upload a song, follow the job, download the chart.

    POST   /jobs              multipart: file (+ artist, title, language, year)
    GET    /jobs              list jobs (?status=queued|running|done|...)
    GET    /jobs/{id}         job state, current stage and reason
    GET    /jobs/{id}/chart   the UltraStar song folder as a zip
    DELETE /jobs/{id}         cancel a queued job / remove a finished one
    GET    /health

Run: ``python -m src.api`` (env OUTPUT_DIR, API_HOST, API_PORT,
API_MAX_UPLOAD_MB).
"""
import os
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from src.api.store import (CANCELLED, HAS_CHART, QUEUED, RUNNING,
                           JobStore, Worker)
from src.readers.metadata import parse_filename, read_metadata
from src.model.song_job import SongJob
from src.planner import Stage

AUDIO_EXTS = (".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus")


def _public(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if k != "folder"}
    out["chart_ready"] = rec["status"] in HAS_CHART and bool(rec.get("folder"))
    return out


def _resolve_identity(audio: Path, filename: str, artist, title, probe):
    """Form values win; then tags; then the uploaded file name."""
    job = SongJob(artist=artist, title=title)
    try:
        read_metadata(audio, job, probe=probe)
    except Exception:  # noqa: BLE001 - no ffprobe etc.: fall back to name
        pass
    # read_metadata saw only our storage name ("audio.mp3"): drop its guesses
    if job.sources.get("artist") == "filename":
        job.artist = None
    if job.sources.get("title") == "filename":
        job.title = None
    if not (job.artist and job.title):
        name_artist, name_title = parse_filename(filename)
        job.artist = job.artist or name_artist
        job.title = job.title or name_title
    return job.artist, job.title, job.year


def create_app(output_dir=None, stages: Optional[Dict[str, Stage]] = None,
               runner=None, probe=None,
               max_upload_mb: Optional[int] = None) -> FastAPI:
    out = Path(output_dir or os.environ.get("OUTPUT_DIR", "output")).resolve()
    limit = (max_upload_mb or int(os.environ.get("API_MAX_UPLOAD_MB", "200"))
             ) * 1024 * 1024
    store = JobStore(out / ".api")
    holder: dict = {}

    def get_worker() -> Worker:
        if "worker" not in holder:
            real = stages
            if real is None:
                from src.planner import default_stages
                from src.providers.lrclib import LrclibClient
                from src.providers.musicbrainz import MusicBrainzClient
                real = default_stages(
                    out,
                    lrclib=LrclibClient(cache_dir=out / ".cache" / "lrclib"),
                    musicbrainz=MusicBrainzClient(
                        "ultrastar-song-creator/0.1 (REST API)",
                        cover_dir=out / ".cache" / "covers"))
            kwargs = {"runner": runner} if runner else {}
            holder["worker"] = Worker(store, out, real, **kwargs)
        return holder["worker"]

    @asynccontextmanager
    async def lifespan(_app):
        worker = get_worker()
        worker.start()
        yield
        worker.stop()

    app = FastAPI(title="UltraStar Chart Generator", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok",
                "queued": len(store.list(QUEUED)),
                "running": len(store.list(RUNNING))}

    @app.post("/jobs", status_code=202)
    async def create_job(file: UploadFile = File(...),
                         artist: Optional[str] = Form(None),
                         title: Optional[str] = Form(None),
                         language: Optional[str] = Form(None),
                         year: Optional[int] = Form(None)):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in AUDIO_EXTS:
            raise HTTPException(
                415, f"unsupported audio type '{ext}'; use one of "
                     f"{', '.join(AUDIO_EXTS)}")
        job_id = store.new_id()
        folder = store.job_dir(job_id)
        folder.mkdir(parents=True)
        audio = folder / f"audio{ext}"
        size = 0
        with audio.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    fh.close()
                    shutil.rmtree(folder, ignore_errors=True)
                    raise HTTPException(
                        413, f"upload larger than {limit // 1024 // 1024} MB")
                fh.write(chunk)
        if size == 0:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(422, "empty file")
        artist, title, tag_year = _resolve_identity(
            audio, file.filename or "", (artist or "").strip() or None,
            (title or "").strip() or None, probe)
        if not (artist and title):
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(
                422, "artist and title are required (form fields, file tags "
                     "or a 'Artist - Title' file name)")
        rec = store.create({
            "id": job_id, "artist": artist, "title": title,
            "language": (language or "").strip().lower() or None,
            "year": year or tag_year, "filename": file.filename})
        get_worker().submit(job_id)
        return _public(rec)

    @app.get("/jobs")
    def list_jobs(status: Optional[str] = None):
        return [_public(r) for r in store.list(status)]

    def _find(job_id: str) -> dict:
        rec = store.get(job_id)
        if rec is None:
            raise HTTPException(404, "job not found")
        return rec

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        return _public(_find(job_id))

    @app.get("/jobs/{job_id}/chart")
    def get_chart(job_id: str):
        rec = _find(job_id)
        if rec["status"] not in HAS_CHART:
            raise HTTPException(409, f"chart not ready (status: "
                                     f"{rec['status']})")
        folder = Path(rec["folder"]) if rec.get("folder") else None
        if folder is None or not folder.is_dir():
            raise HTTPException(404, "chart folder is gone")
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            for path in sorted(folder.rglob("*")):
                rel = path.relative_to(folder)
                if path.is_file() and ".work" not in rel.parts:
                    zf.write(path, Path(folder.name) / rel)
        tmp.close()
        return FileResponse(
            tmp.name, media_type="application/zip",
            filename=f"{folder.name}.zip",
            background=BackgroundTask(os.unlink, tmp.name))

    @app.delete("/jobs/{job_id}")
    def delete_job(job_id: str):
        rec = _find(job_id)
        if rec["status"] == RUNNING:
            raise HTTPException(409, "job is running; wait for it to finish")
        if rec["status"] == QUEUED:
            store.update(job_id, status=CANCELLED, finished_at=None)
            return _public(store.get(job_id))
        store.delete(job_id)
        return {"id": job_id, "deleted": True}

    return app
