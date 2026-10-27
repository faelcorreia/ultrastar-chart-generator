"""B2: read song metadata from file tags and the file name.

Tags win over the file name (`Artista - Título`). Tags and the embedded cover
come from ffprobe/ffmpeg (mutagen is not a dependency). Both are injectable.
"""
import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from src.model import SongJob
from src.providers.name_clean import strip_number_prefix

Probe = Callable[[Path], Optional[dict]]
CoverExtractor = Callable[[Path, Path], bool]


def parse_filename(path) -> Tuple[Optional[str], Optional[str]]:
    """Split `Artista - Título`; a leading track number is ignored."""
    stem = Path(path).stem.strip()
    # "01. Artist - Title" / "01 Artist - Title" / "01-Artist - Title"
    stem = strip_number_prefix(stem) or stem
    parts = [p.strip() for p in re.split(r"\s+-\s+|\s+–\s+", stem)]
    if len(parts) >= 3 and re.fullmatch(r"\d{1,3}\.?", parts[0]):
        parts = parts[1:]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None, (stem or None)
    return parts[0], " - ".join(parts[1:])


def ffprobe(path: Path) -> Optional[dict]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, check=True, timeout=60)
        return json.loads(out.stdout.decode("utf-8", "replace"))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def ffmpeg_extract_cover(audio: Path, dest: Path) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-v", "quiet", "-y", "-i", str(audio), "-an",
             "-map", "0:v:0", "-frames:v", "1", str(dest)],
            capture_output=True, check=True, timeout=60)
        return dest.exists() and dest.stat().st_size > 0
    except (OSError, subprocess.SubprocessError):
        return False


def _tags(info: dict) -> Dict[str, str]:
    raw = dict((info.get("format") or {}).get("tags") or {})
    for stream in info.get("streams") or []:
        for key, value in (stream.get("tags") or {}).items():
            raw.setdefault(key, value)
    return {k.lower(): str(v).strip() for k, v in raw.items() if str(v).strip()}


def _year(value: Optional[str]) -> Optional[int]:
    match = re.match(r"\s*(\d{4})", value or "")
    return int(match.group(1)) if match else None


def read_metadata(path, job: Optional[SongJob] = None,
                  probe: Optional[Probe] = None,
                  extract_cover: Optional[CoverExtractor] = None,
                  cover_dir=None) -> SongJob:
    """Fill artist/title/album/year/duration/cover of `job` (new if None).

    Values already on the job are kept; otherwise tags beat the file name.
    The cover is extracted only when `cover_dir` is given.
    """
    path = Path(path)
    job = job or SongJob()
    if job.audio_path is None:
        job.audio_path = path
    info = (probe or ffprobe)(path) or {}
    tags = _tags(info)
    name_artist, name_title = parse_filename(path)

    def fill(attr, tag_value, name_value=None):
        if getattr(job, attr):
            return
        if tag_value:
            setattr(job, attr, tag_value)
            job.sources[attr] = "tags"
        elif name_value:
            setattr(job, attr, name_value)
            job.sources[attr] = "filename"

    fill("artist", tags.get("artist") or tags.get("album_artist"), name_artist)
    fill("title", tags.get("title"), name_title)
    fill("album", tags.get("album"))
    if job.year is None:
        year = _year(tags.get("date") or tags.get("year"))
        if year:
            job.year = year
            job.sources["year"] = "tags"
    if job.duration is None:
        try:
            job.duration = float((info.get("format") or {})["duration"])
        except (KeyError, TypeError, ValueError):
            pass
    if job.cover_path is None and cover_dir is not None:
        has_cover = any(s.get("codec_type") == "video"
                        for s in info.get("streams") or [])
        if has_cover:
            dest = Path(cover_dir) / f"{path.stem}-cover.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if (extract_cover or ffmpeg_extract_cover)(path, dest):
                job.cover_path = dest
                job.sources["cover"] = "tags"
    return job
