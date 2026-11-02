"""F1: find audio/CDG files and library duplicates by `Artista - Título`."""
import re
from pathlib import Path
from typing import Dict, Iterator, Optional, Set

from src.batch.songlist import song_key
from src.readers.metadata import parse_filename

AUDIO_EXTS = {".mp3", ".mp2", ".flac", ".wav", ".m4a", ".ogg", ".opus"}
CDG_EXTS = {".cdg"}
DEFAULT_LIBRARY = "/run/media/ghost/ROMS/ultrastar-songs"

# Karaoke files carry the disc label in the name: "Title [HM Karaoke]",
# "Title (With Vocals)". The song list has the clean title, so index both.
_BRACKET_SUFFIX = re.compile(r"\s*[\[(][^\[\]()]*[\])]\s*$")
# Disc catalog prefix: "SF337-01 - Artist - Title", "SFWS072-01 - ...".
# At least two digits so "U2 - One" and "Blink-182 - ..." are left alone.
_CATALOG_PREFIX = re.compile(r"^[A-Za-z]{2,6}\d{2,5}(?:[-_]\d+)?\s+-\s+")


def strip_label(title: str) -> str:
    """Drop a trailing karaoke label: 'Title [HM Karaoke]' -> 'Title'."""
    return _BRACKET_SUFFIX.sub("", title).strip() or title


def split_song_name(path) -> tuple:
    """Artist/title of a karaoke file, ignoring a disc catalog prefix.

    `SF337-01 - Saturdays - Not Giving Up.cdg` gives
    ("Saturdays", "Not Giving Up"). A name that is a real artist with a
    number (U2, AKB48) keeps its own split when the stripped one is invalid.
    """
    artist, title = parse_filename(path)
    stem = Path(path).stem
    cleaned = _CATALOG_PREFIX.sub("", stem, count=1)
    if cleaned != stem:
        a, t = parse_filename(cleaned)
        if a and t:
            return a, t
    return artist, title


def _index_keys(artist: str, title: str) -> Iterator[str]:
    yield song_key(artist, title)
    clean = strip_label(title)
    if clean != title:
        yield song_key(artist, clean)


def _index(folder, exts: Set[str], prefer: str) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    root = Path(folder)
    if not root.is_dir():
        return found
    files = sorted((p for p in root.rglob("*")
                    if p.is_file() and p.suffix.lower() in exts),
                   key=lambda p: (p.suffix.lower() != prefer, str(p)))
    for p in files:
        artist, title = split_song_name(p)
        if artist and title:
            for key in _index_keys(artist, title):
                found.setdefault(key, p)
    return found


def index_audio(folder) -> Dict[str, Path]:
    """Map normalized artist+title to the audio file (recursive). With
    duplicates the first in sorted order wins, preferring .mp3."""
    return _index(folder, AUDIO_EXTS, ".mp3")


def index_cdg(folder) -> Dict[str, Path]:
    """Map normalized artist+title to the .cdg file, same rules as audio."""
    return _index(folder, CDG_EXTS, ".cdg")


def find_audio(index: Dict[str, Path], artist: str, title: str
               ) -> Optional[Path]:
    for key in _index_keys(artist, title):
        found = index.get(key)
        if found is not None:
            return found
    return None


def find_cdg(index: Dict[str, Path], artist: str, title: str
             ) -> Optional[Path]:
    for key in _index_keys(artist, title):
        found = index.get(key)
        if found is not None:
            return found
    return None


def library_keys(library) -> Set[str]:
    """Keys of songs already in the library: names of entries (folders or
    files) up to two levels deep, expected as `Artista - Título`."""
    keys: Set[str] = set()
    root = Path(library) if library else None
    if root is None or not root.is_dir():
        return keys

    def add(name: str):
        artist, title = parse_filename(name + ".x")
        if artist and title:
            keys.add(song_key(artist, title))

    for child in root.iterdir():
        if child.is_dir():
            add(child.name)
            for sub in child.iterdir():
                if sub.is_dir():
                    add(sub.name)
                elif sub.suffix.lower() == ".txt":
                    add(sub.stem)
        else:
            add(child.stem)
    return keys
