"""Bridge between SongJob and the dict the current stages still use.

The stages in src/core read and write a per-song dict. Until they take a
SongJob directly, the pipeline builds that dict from the job and copies what
the stages produced back into the job. Nothing here changes any value.
"""
from pathlib import Path
from typing import Any, Dict

from src.model.lyrics import Lyrics
from src.model.song_job import SongJob

# Dict keys that map to first-class SongJob fields (key -> field).
_PATH_FIELDS = {"full_track": "audio_path", "vocals": "vocals_path",
                "cover": "cover_path"}
_IDENTITY_KEYS = {"artist", "title", "language"}


def job_from_song(song: Dict[str, Any]) -> SongJob:
    """Build a SongJob from a collected song dict, keeping language as given."""
    job = SongJob(
        artist=song.get("artist"),
        title=song.get("title"),
        language=song.get("language"),
    )
    absorb(job, song)
    return job


def absorb(job: SongJob, song: Dict[str, Any]) -> SongJob:
    """Copy the values of a stage dict into the job."""
    for key, value in song.items():
        if key in _IDENTITY_KEYS:
            setattr(job, key, value)
        elif key in _PATH_FIELDS:
            setattr(job, _PATH_FIELDS[key], Path(value))
        elif key == "lyrics":
            job.extra["lyrics_path"] = Path(value)
            job.sources["lyrics"] = str(value)
            if job.lyrics is None:
                job.lyrics = Lyrics(language=job.language, source=str(value))
        else:
            job.extra[key] = value
    return job


def song_from_job(job: SongJob) -> Dict[str, Any]:
    """Rebuild the stage dict from the job."""
    song: Dict[str, Any] = {
        "artist": job.artist,
        "title": job.title,
        "language": job.language,
    }
    for key, attr in _PATH_FIELDS.items():
        value = getattr(job, attr)
        if value is not None:
            song[key] = value
    if "lyrics_path" in job.extra:
        song["lyrics"] = job.extra["lyrics_path"]
    for key, value in job.extra.items():
        if key != "lyrics_path":
            song[key] = value
    return song
