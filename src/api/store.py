"""REST API job store and single-worker queue.

Each API job lives in ``<data_dir>/<id>/job.json`` next to its uploaded audio,
rewritten atomically on every change, so the queue survives a restart: jobs
that were queued or running go back to the queue (the per-stage cache resumes
them). One worker thread runs one song at a time (the pipeline is heavy).
"""
import json
import os
import queue
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.batch.report import now_iso
from src.model.song_job import SongJob
from src.planner import JobResult, Stage, run_job

QUEUED, RUNNING, CANCELLED = "queued", "running", "cancelled"
ACTIVE = (QUEUED, RUNNING)
# planner statuses that leave a chart on disk ("review" = chart to check)
HAS_CHART = ("done", "review")


class JobStore:
    def __init__(self, data_dir):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: Dict[str, dict] = {}
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        for path in sorted(self.root.glob("*/job.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                self._jobs[rec["id"]] = rec
            except (ValueError, OSError, KeyError):
                continue  # unreadable record: ignore, never crash

    def _save(self, rec: dict) -> None:
        folder = self.root / rec["id"]
        folder.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, folder / "job.json")

    # -- access ------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def new_id(self) -> str:
        return uuid.uuid4().hex

    def create(self, rec: dict) -> dict:
        with self._lock:
            rec = dict(rec, status=QUEUED, created_at=now_iso(), stage=None,
                       reason=None, folder=None)
            self._jobs[rec["id"]] = rec
            self._save(rec)
            return dict(rec)

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            rec = self._jobs.get(job_id)
            return dict(rec) if rec else None

    def list(self, status: Optional[str] = None) -> List[dict]:
        with self._lock:
            recs = [dict(r) for r in self._jobs.values()
                    if status in (None, r["status"])]
        return sorted(recs, key=lambda r: r["created_at"], reverse=True)

    def update(self, job_id: str, **fields) -> Optional[dict]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            rec.update(fields)
            self._save(rec)
            return dict(rec)

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(self.root / job_id, ignore_errors=True)

    def recover(self) -> List[str]:
        """Ids to (re)queue after a start: queued and interrupted jobs."""
        with self._lock:
            ids = [r["id"] for r in self.list() if r["status"] in ACTIVE]
            for job_id in ids:
                self.update(job_id, status=QUEUED, stage=None)
        return list(reversed(ids))  # oldest first


class Worker:
    """One thread, one song at a time, in submission order."""

    def __init__(self, store: JobStore, output_dir, stages: Dict[str, Stage],
                 runner: Callable = run_job):
        self.store, self.output_dir = store, output_dir
        self.stages, self.runner = stages, runner
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        for job_id in self.store.recover():
            self._queue.put(job_id)
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="api-worker")
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            rec = self.store.get(job_id)
            if rec is None or rec["status"] != QUEUED:
                continue  # cancelled or deleted while waiting
            self._run(rec)

    def _tracked(self, job_id: str) -> Dict[str, Stage]:
        def wrap(name: str, func: Stage) -> Stage:
            def stage(job):
                self.store.update(job_id, stage=name)
                func(job)
            return stage
        return {n: wrap(n, f) for n, f in self.stages.items()}

    def _run(self, rec: dict) -> None:
        job_id = rec["id"]
        self.store.update(job_id, status=RUNNING, started_at=now_iso())
        audio = next(self.store.job_dir(job_id).glob("audio.*"), None)
        job = SongJob(artist=rec["artist"], title=rec["title"],
                      year=rec.get("year"), language=rec.get("language"),
                      audio_path=audio)
        try:
            res: JobResult = self.runner(job, self.output_dir,
                                         self._tracked(job_id))
            folder = job.extra.get("output_folder")
            self.store.update(
                job_id, status=res.status, reason=res.reason, stage=None,
                folder=str(folder) if folder else None,
                language=job.language, ran=res.ran, cached=res.cached,
                notes=list(job.notes), finished_at=now_iso())
        except Exception as exc:  # noqa: BLE001 - a job never kills the worker
            self.store.update(job_id, status="failed", stage=None,
                              reason=f"{type(exc).__name__}: {exc}",
                              finished_at=now_iso())
