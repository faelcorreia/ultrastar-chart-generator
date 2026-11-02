"""F1: run a song list one song at a time through the planner."""
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.batch.locate import (find_audio, find_cdg, index_audio, index_cdg,
                              library_keys)
from src.batch.report import BatchReport, now_iso
from src.batch.songlist import ListEntry, song_key
from src.model.song_job import SongJob
from src.planner import DONE, JobResult, Stage, run_job

SKIPPED = "skipped"


def run_batch(entries: List[ListEntry], audio_dir, output_dir,
              stages: Dict[str, Stage], library=None,
              force_existing: bool = False, force: bool = False,
              on_result: Optional[Callable[[ListEntry, JobResult], None]] = None,
              report: Optional[BatchReport] = None, retry_all: bool = False,
              foreign_audio: bool = False) -> List[JobResult]:
    """One JobResult per entry, in list order. Songs already in `library`
    get status "skipped" (unless `force_existing`). The language is the list
    value or None: the language stage resolves it, never a silent default.
    With a `report` (F2) every result is saved right away; entries already
    done/skipped in it are not reprocessed unless `retry_all`."""
    audio = index_audio(audio_dir) if audio_dir else {}
    cdg = index_cdg(audio_dir) if audio_dir else {}
    existing = set() if force_existing else library_keys(library)
    results: List[JobResult] = []
    for entry in entries:
        key = song_key(entry.artist, entry.title)
        if report and not retry_all and report.is_finished(key):
            prev = report.get(key)
            res = JobResult(prev["status"], "resumed: " + prev["reason"],
                            SongJob(artist=entry.artist, title=entry.title,
                                    year=entry.year,
                                    language=prev.get("language")))
            results.append(res)
            if on_result:
                on_result(entry, res)
            continue
        started = now_iso()
        job = SongJob(artist=entry.artist, title=entry.title,
                      year=entry.year, language=entry.language,
                      audio_path=find_audio(audio, entry.artist, entry.title),
                      cdg_path=find_cdg(cdg, entry.artist, entry.title))
        if foreign_audio:
            job.extra["cdg_foreign_audio"] = True
        if key in existing:
            res = JobResult(SKIPPED, "already in library", job)
        else:
            res = run_job(job, output_dir, stages, force=force)
        results.append(res)
        if report:
            report.record(key, res.status, res.reason, entry.artist,
                          entry.title, job.language or entry.language, started)
        if on_result:
            on_result(entry, res)
    return results
