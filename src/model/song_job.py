from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.model.lyrics import Lyrics


@dataclass
class SongJob:
    """Everything known about a song so far.

    Fields stay None until some reader, provider or stage fills them, so a
    planner can look at what is missing and pick the stages to run.
    `language` never defaults: it is None until resolved.
    """

    artist: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    album: Optional[str] = None
    language: Optional[str] = None
    audio_path: Optional[Path] = None
    vocals_path: Optional[Path] = None
    cover_path: Optional[Path] = None
    cdg_path: Optional[Path] = None
    duration: Optional[float] = None
    lyrics: Optional[Lyrics] = None
    sources: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("audio_path", "vocals_path", "cover_path", "cdg_path"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                setattr(self, name, Path(value))

    @property
    def resolved_language(self) -> Optional[str]:
        if self.language:
            return self.language
        if self.lyrics is not None:
            return self.lyrics.language
        return None

    def require_language(self) -> str:
        language = self.resolved_language
        if not language:
            raise ValueError(
                f"Language not resolved for '{self.artist} - {self.title}'"
            )
        return language

    def missing(self) -> List[str]:
        """Names of the basic pieces not known yet."""
        wanted = {
            "artist": self.artist,
            "title": self.title,
            "language": self.resolved_language,
            "audio": self.audio_path,
            "lyrics": self.lyrics,
            "cover": self.cover_path,
            "year": self.year,
        }
        return [name for name, value in wanted.items() if not value]

    def note(self, message: str) -> None:
        self.notes.append(message)
