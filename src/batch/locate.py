"""F1: find audio files and library duplicates by `Artista - Título`."""
from pathlib import Path
from typing import Dict, Optional, Set

from src.batch.songlist import song_key
from src.readers.metadata import parse_filename

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus"}
DEFAULT_LIBRARY = "/run/media/ghost/ROMS/ultrastar-songs"


def index_audio(folder) -> Dict[str, Path]:
    """Map normalized artist+title to the audio file (recursive). With
    duplicates the first in sorted order wins, preferring .mp3."""
    found: Dict[str, Path] = {}
    root = Path(folder)
    if not root.is_dir():
        return found
    files = sorted((p for p in root.rglob("*")
                    if p.is_file() and p.suffix.lower() in AUDIO_EXTS),
                   key=lambda p: (p.suffix.lower() != ".mp3", str(p)))
    for p in files:
        artist, title = parse_filename(p)
        if artist and title:
            found.setdefault(song_key(artist, title), p)
    return found


def find_audio(index: Dict[str, Path], artist: str, title: str
               ) -> Optional[Path]:
    return index.get(song_key(artist, title))


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
