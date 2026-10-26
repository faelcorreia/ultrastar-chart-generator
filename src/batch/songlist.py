"""F1: parse the batch list `artista;título;ano[;idioma]`."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.core.text_norm import fold_for_compare

_HEADER_ARTIST = {"artista", "artist"}


@dataclass(frozen=True)
class ListEntry:
    artist: str
    title: str
    year: Optional[int] = None
    language: Optional[str] = None
    line: int = 0


def match_key(text: str) -> str:
    """Accent, case and punctuation tolerant key (any of en/pt/es)."""
    return fold_for_compare(text or "", "pt")


def song_key(artist: str, title: str) -> str:
    return f"{match_key(artist)}|{match_key(title)}"


def parse_song_list(path) -> List[ListEntry]:
    """Read the list file (UTF-8). Blank lines and `#` comments are skipped,
    as is a header row. Malformed lines raise ValueError with the line no."""
    entries: List[ListEntry] = []
    text = Path(path).read_text(encoding="utf-8-sig")
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = [c.strip() for c in line.split(";")]
        if len(cols) < 2 or not cols[0] or not cols[1]:
            raise ValueError(f"line {no}: expected 'artist;title;year': {raw!r}")
        if not entries and match_key(cols[0]) in _HEADER_ARTIST:
            continue
        year = None
        if len(cols) > 2 and cols[2]:
            if not cols[2].isdigit():
                raise ValueError(f"line {no}: invalid year {cols[2]!r}")
            year = int(cols[2])
        language = cols[3] if len(cols) > 3 and cols[3] else None
        entries.append(ListEntry(cols[0], cols[1], year, language, no))
    return entries
