"""F1: run a song list one song at a time through the planner."""
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.batch.locate import (find_audio, index_audio, library_keys)
from src.batch.songlist import ListEntry, song_key
from src.model.song_job import SongJob
from src.planner import DONE, JobResult, Stage, run_job

SKIPPED = "skipped"


def run_batch(entries: List[ListEntry], audio_dir, output_dir,
              stages: Dict[str, Stage], library=None,
              force_existing: bool = False, force: bool = False,
              on_result: Optional[Callable[[ListEntry, JobResult], None]] = None
              ) -> List[JobResult]:
    """One JobResult per entry, in list order. Songs already in `library`
    get status "skipped" (unless `force_existing`). The language is the list
    value or None: the language stage resolves it, never a silent default."""
    audio = index_audio(audio_dir) if audio_dir else {}
    existing = set() if force_existing else library_keys(library)
    results: List[JobResult] = []
    for entry in entries:
        job = SongJob(artist=entry.artist, title=entry.title,
                      year=entry.year, language=entry.language,
                      audio_path=find_audio(audio, entry.artist, entry.title))
        if song_key(entry.artist, entry.title) in existing:
            res = JobResult(SKIPPED, "already in library", job)
        else:
            res = run_job(job, output_dir, stages, force=force)
        results.append(res)
        if on_result:
            on_result(entry, res)
    return results
