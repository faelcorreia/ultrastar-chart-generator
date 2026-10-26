from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Syllable:
    """A sung unit of text. Times are in seconds, never beats."""

    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    pitch: Optional[int] = None

    @property
    def is_timed(self) -> bool:
        return self.start is not None and self.end is not None

    @property
    def duration(self) -> Optional[float]:
        if not self.is_timed:
            return None
        return self.end - self.start


@dataclass
class Line:
    """A lyric line. Without syllable timing it only carries its text."""

    text: str = ""
    start: Optional[float] = None
    end: Optional[float] = None
    syllables: List[Syllable] = field(default_factory=list)
    # True when a stanza break (blank line) precedes this line (I5).
    stanza_break_before: bool = False

    def __post_init__(self):
        if not self.text and self.syllables:
            self.text = "".join(s.text for s in self.syllables)
        timed = [s for s in self.syllables if s.is_timed]
        if self.start is None and timed:
            self.start = timed[0].start
        if self.end is None and timed:
            self.end = timed[-1].end


@dataclass
class Lyrics:
    """Lyrics of a song. `language` stays None until it is resolved."""

    lines: List[Line] = field(default_factory=list)
    language: Optional[str] = None
    source: Optional[str] = None

    @property
    def is_synced(self) -> bool:
        return bool(self.lines) and all(
            line.start is not None for line in self.lines
        )

    @property
    def has_syllable_timing(self) -> bool:
        return bool(self.lines) and all(
            line.syllables and all(s.is_timed for s in line.syllables)
            for line in self.lines
        )

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @classmethod
    def from_text(cls, text: str, language: Optional[str] = None,
                  source: Optional[str] = None) -> "Lyrics":
        lines = [Line(text=raw.strip()) for raw in text.splitlines()
                 if raw.strip()]
        return cls(lines=lines, language=language, source=source)
