"""C1: LRCLIB client with disk cache and duration validation."""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode

from src.model import Line, Lyrics
from src.providers.http import HttpGet, urllib_get
from src.providers.name_clean import lookup_variants

BASE_URL = "https://lrclib.net"
DURATION_TOLERANCE = 3.0
USER_AGENT = "UltraStarSongCreator/0.1 (https://github.com/ultrastar-song-creator)"
_LRC_TIME = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


@dataclass
class LrclibResult:
    id: Optional[int]
    artist: str
    title: str
    duration: Optional[float]
    synced: Optional[str] = None
    plain: Optional[str] = None

    @property
    def kind(self) -> str:
        return "synced" if self.synced else "plain"

    def to_lyrics(self, language: Optional[str] = None) -> Lyrics:
        source = f"lrclib:{self.id}" if self.id is not None else "lrclib"
        if self.synced:
            return Lyrics(lines=parse_synced(self.synced), language=language,
                          source=source)
        return Lyrics.from_text(self.plain or "", language=language,
                                source=source)


def parse_synced(lrc: str) -> List[Line]:
    """Minimal LRC parse: one Line per timestamped, non-empty text row."""
    lines = []
    for raw in lrc.splitlines():
        stamps = _LRC_TIME.findall(raw)
        text = _LRC_TIME.sub("", raw).strip()
        if stamps and text:
            minutes, seconds = stamps[0]
            lines.append(Line(text=text, start=int(minutes) * 60 + float(seconds)))
    return lines


class LrclibClient:
    def __init__(self, http: Optional[HttpGet] = None, cache_dir=None,
                 user_agent: str = USER_AGENT, base_url: str = BASE_URL,
                 tolerance: float = DURATION_TOLERANCE):
        self.http = http or urllib_get
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.headers = {"User-Agent": user_agent}
        self.base_url = base_url.rstrip("/")
        self.tolerance = tolerance

    def find(self, artist: str, title: str, album: Optional[str] = None,
             duration: Optional[float] = None) -> Optional[LrclibResult]:
        """Best lyrics for the song: synced first, plain only if no synced."""
        key = self._key(artist, title, album, duration)
        cached = self._cache_read(key)
        if cached is not None:
            return cached.get("result") and LrclibResult(**cached["result"])
        result = None
        # I7: exact name first, then cleaned/looser variants in cascade
        for v_artist, v_title in lookup_variants(artist, title):
            result = self._lookup(v_artist, v_title, album, duration)
            if result is not None:
                break
        self._cache_write(key, result)
        return result

    # -- lookup ----------------------------------------------------------
    def _lookup(self, artist, title, album, duration):
        params = {"artist_name": artist, "track_name": title}
        if album:
            params["album_name"] = album
        if duration:
            params["duration"] = str(int(round(duration)))
        direct = None
        data = self._get("/api/get", params)
        if isinstance(data, dict):
            direct = self._accept(data, duration)
            if direct and direct.synced:
                return direct
        search = {"artist_name": artist, "track_name": title}
        found = self._get("/api/search", search)
        candidates = [r for r in (self._accept(d, duration)
                                  for d in found or [] if isinstance(d, dict))
                      if r]
        synced = [r for r in candidates if r.synced]
        if synced:
            return min(synced, key=lambda r: self._distance(r, duration))
        if direct:
            return direct
        plain = [r for r in candidates if r.plain]
        if plain:
            return min(plain, key=lambda r: self._distance(r, duration))
        return None

    def _accept(self, data: dict, duration) -> Optional[LrclibResult]:
        if data.get("instrumental"):
            return None
        synced = (data.get("syncedLyrics") or "").strip() or None
        plain = (data.get("plainLyrics") or "").strip() or None
        if not synced and not plain:
            return None
        got = data.get("duration")
        if duration and got is not None:
            if abs(float(got) - float(duration)) > self.tolerance:
                return None
        return LrclibResult(
            id=data.get("id"), artist=data.get("artistName") or "",
            title=data.get("trackName") or "",
            duration=float(got) if got is not None else None,
            synced=synced, plain=plain)

    @staticmethod
    def _distance(result: LrclibResult, duration) -> float:
        if not duration or result.duration is None:
            return 0.0
        return abs(result.duration - duration)

    def _get(self, path: str, params: dict):
        response = self.http(f"{self.base_url}{path}?{urlencode(params)}",
                             self.headers)
        if response is None or response.status != 200:
            return None
        try:
            return response.json()
        except ValueError:
            return None

    # -- cache -----------------------------------------------------------
    @staticmethod
    def _key(artist, title, album, duration) -> str:
        raw = "|".join([artist.casefold().strip(), title.casefold().strip(),
                        (album or "").casefold().strip(),
                        str(int(round(duration))) if duration else ""])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_read(self, key):
        if not self.cache_dir:
            return None
        try:
            return json.loads((self.cache_dir / f"{key}.json").read_text("utf-8"))
        except (OSError, ValueError):
            return None

    def _cache_write(self, key, result):
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {"result": result.__dict__ if result else None}
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(payload, ensure_ascii=False), "utf-8")
