"""F2: batch report/state file (batch_report.json), crash-safe and resumable.

One entry per song (keyed by accent/case tolerant artist|title). Rewritten
atomically (temp file + os.replace) after every song, so a crash never leaves
a truncated file. On re-run, entries with status done/skipped are not
reprocessed; failed/review/awaiting_audio are retried.
"""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPORT_NAME = "batch_report.json"
RESUME_DONE = ("done", "skipped")
SUMMARY_ORDER = ("done", "skipped", "failed", "review", "awaiting_audio")
_VERSION = 1


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BatchReport:
    def __init__(self, path):
        self.path = Path(path)
        self.songs: Dict[str, dict] = {}
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.songs = dict(data.get("songs", {}))
            except (ValueError, OSError, AttributeError):
                self.songs = {}  # unreadable state: start over, never crash

    def is_finished(self, key: str) -> bool:
        return self.songs.get(key, {}).get("status") in RESUME_DONE

    def get(self, key: str) -> Optional[dict]:
        return self.songs.get(key)

    def record(self, key: str, status: str, reason: str, artist: str,
               title: str, language: Optional[str],
               started_at: Optional[str] = None) -> dict:
        entry = {"artist": artist, "title": title, "language": language,
                 "status": status, "reason": reason,
                 "started_at": started_at or now_iso(),
                 "finished_at": now_iso()}
        self.songs[key] = entry
        self.save()
        return entry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"version": _VERSION, "updated_at": now_iso(),
                              "songs": self.songs}, ensure_ascii=False,
                             indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".report-",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def counts(self, keys: Optional[List[str]] = None) -> Dict[str, int]:
        out: Dict[str, int] = {s: 0 for s in SUMMARY_ORDER}
        for k, e in self.songs.items():
            if keys is None or k in keys:
                out[e["status"]] = out.get(e["status"], 0) + 1
        return out

    def summary(self, keys: Optional[List[str]] = None) -> str:
        lines = ["Batch summary: " + ", ".join(
            f"{s}={n}" for s, n in self.counts(keys).items())]
        for k, e in self.songs.items():
            if keys is not None and k not in keys:
                continue
            if e["status"] in ("failed", "review"):
                lines.append(f"  [{e['status']}] {e['artist']} - "
                             f"{e['title']}: {e['reason']}")
        return "\n".join(lines)
