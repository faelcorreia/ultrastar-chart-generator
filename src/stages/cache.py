"""Per-stage cache: one JSON per stage in <output>/<song>/.work.

An entry stores the fingerprint of the stage inputs and the values the stage
produced. A stage is skipped only when the fingerprint matches and every file
it produced still exists, so changed inputs or deleted outputs recompute.
"""
import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from src.model.lyrics import Line, Lyrics, Syllable
from src.model.song_job import SongJob

VERSION = 1
_TYPES = {"Lyrics": Lyrics, "Line": Line, "Syllable": Syllable,
          "SongJob": SongJob}


def encode(value: Any) -> Any:
    """Turn paths, Lyrics, SongJob and containers into plain JSON data."""
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name not in _TYPES:
            raise TypeError(f"Cannot cache {name}")
        fields = {f.name: encode(getattr(value, f.name))
                  for f in dataclasses.fields(value)}
        return {"__type__": name, "fields": fields}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot cache {type(value).__name__}")


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        if "__path__" in value:
            return Path(value["__path__"])
        if "__type__" in value:
            cls = _TYPES[value["__type__"]]
            return cls(**{k: decode(v) for k, v in value["fields"].items()})
        return {k: decode(v) for k, v in value.items()}
    return value


def _paths(value: Any):
    if isinstance(value, Path):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _paths(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _paths(v)


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(*parts: Any) -> str:
    """Hash of the stage inputs. Existing files count by content, other
    Paths by name, everything else by its encoded value."""
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, Path):
            digest = _file_digest(part) if part.is_file() else "missing"
            h.update(f"file:{part.name}:{digest}".encode("utf-8"))
        else:
            h.update(json.dumps(encode(part), sort_keys=True,
                                ensure_ascii=False).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip() or "song"


class StageCache:
    """Load/save the result of each stage for one song."""

    def __init__(self, output_dir: Path, song_name: str,
                 enabled: bool = True):
        self.dir = Path(output_dir) / _safe(song_name) / ".work"
        self.enabled = enabled

    def path(self, stage: str) -> Path:
        return self.dir / f"{_safe(stage)}.json"

    def load(self, stage: str, key: str) -> Optional[Dict[str, Any]]:
        """Return the stored result, or None when absent, stale or broken."""
        if not self.enabled:
            return None
        try:
            with open(self.path(stage), encoding="utf-8") as handle:
                entry = json.load(handle)
            if entry.get("version") != VERSION or entry.get("key") != key:
                return None
            result = decode(entry["result"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if any(not p.exists() for p in _paths(result)):
            return None
        return result

    def save(self, stage: str, key: str, result: Dict[str, Any]) -> None:
        """Always writes, so a forced run refreshes the stored entry."""
        self.dir.mkdir(parents=True, exist_ok=True)
        entry = {"version": VERSION, "stage": stage, "key": key,
                 "result": encode(result)}
        tmp = self.path(stage).with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(entry, handle, ensure_ascii=False, indent=1)
        tmp.replace(self.path(stage))


def run_stage(cache: StageCache, stage: str, key: str,
              song: Dict[str, Any], func) -> bool:
    """Run func() unless cached; either way `song` ends with the stage's
    outputs. Returns True when the cache was used."""
    cached = cache.load(stage, key)
    if cached is not None:
        song.update(cached)
        return True
    before = dict(song)
    func()
    delta = {k: v for k, v in song.items()
             if k not in before or before[k] != v}
    cache.save(stage, key, delta)
    return False
