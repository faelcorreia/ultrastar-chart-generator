"""C2: MusicBrainz + Cover Art Archive, only for what is still missing."""
import json
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote, urlencode

from src.model import SongJob
from src.providers.http import HttpGet, urllib_get

MB_URL = "https://musicbrainz.org/ws/2"
CAA_URL = "https://coverartarchive.org"
MIN_INTERVAL = 1.1  # MusicBrainz allows ~1 request/second
MIN_SCORE = 90
MAX_COVER_TRIES = 3


class MusicBrainzClient:
    def __init__(self, user_agent: str, http: Optional[HttpGet] = None,
                 cover_dir=None, min_interval: float = MIN_INTERVAL,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if not user_agent or not user_agent.strip():
            raise ValueError("MusicBrainz requires a descriptive User-Agent")
        self.http = http or urllib_get
        self.headers = {"User-Agent": user_agent, "Accept": "application/json"}
        self.cover_dir = Path(cover_dir) if cover_dir else None
        self.min_interval = min_interval
        self._clock, self._sleep = clock, sleep
        self._last: Optional[float] = None

    def fill(self, job: SongJob) -> SongJob:
        """Fill missing year/album/cover. Makes no call if nothing is missing."""
        need_cover = job.cover_path is None and self.cover_dir is not None
        if not (job.artist and job.title):
            return job
        if job.year and job.album and not need_cover:
            return job
        releases = self._search(job.artist, job.title)
        if not releases:
            return job
        year, album = releases["year"], releases["album"]
        if not job.year and year:
            job.year, job.sources["year"] = year, "musicbrainz"
        if not job.album and album:
            job.album, job.sources["album"] = album, "musicbrainz"
        if need_cover:
            for mbid in releases["release_ids"][:MAX_COVER_TRIES]:
                dest = self._cover(mbid, f"{mbid}.jpg")
                if dest:
                    job.cover_path = dest
                    job.sources["cover"] = "coverartarchive"
                    break
        return job

    def _search(self, artist: str, title: str) -> Optional[dict]:
        esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
        query = f'recording:"{esc(title)}" AND artist:"{esc(artist)}"'
        self._throttle()
        response = self.http(
            f"{MB_URL}/recording?{urlencode({'query': query, 'fmt': 'json', 'limit': 5})}",
            self.headers)
        if response is None or response.status != 200:
            return None
        try:
            recordings = response.json().get("recordings") or []
        except (ValueError, AttributeError):
            return None
        for rec in recordings:
            if int(rec.get("score", 0)) < MIN_SCORE or not rec.get("releases"):
                continue
            releases = sorted(
                rec["releases"],
                key=lambda r: (r.get("status") != "Official",
                               r.get("date") or "9999"))
            dated = [r.get("date") for r in releases if r.get("date")]
            date = rec.get("first-release-date") or (min(dated) if dated else "")
            year = int(date[:4]) if date[:4].isdigit() else None
            return {"year": year, "album": releases[0].get("title"),
                    "release_ids": [r["id"] for r in releases if r.get("id")]}
        return None

    def _cover(self, mbid: str, name: str) -> Optional[Path]:
        response = self.http(f"{CAA_URL}/release/{quote(mbid)}/front-500",
                             {"User-Agent": self.headers["User-Agent"]})
        if response is None or response.status != 200 or not response.body:
            return None
        self.cover_dir.mkdir(parents=True, exist_ok=True)
        dest = self.cover_dir / name
        dest.write_bytes(response.body)
        return dest

    def _throttle(self) -> None:
        now = self._clock()
        if self._last is not None:
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last = now
