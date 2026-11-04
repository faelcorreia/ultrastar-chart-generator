"""cdg_adapt: clean OCR noise in CDG lyric text and turn a CDG into a timing guess.

The CDG text is trusted blindly otherwise, so only unambiguous OCR slips are
fixed here (a lone ``|`` is an ``I``; a line without letters is not a lyric).
"""
import re
from typing import List

from src.model.lyrics import Line, Lyrics

SOURCE_FOREIGN = "cdg_foreign_audio"
_PIPE = re.compile(r"\|")


def clean_ocr_text(text: str) -> str:
    text = _PIPE.sub("I", text)
    return " ".join(text.split())


def _has_letters(text: str, minimum: int = 2) -> bool:
    return sum(c.isalpha() for c in text) >= minimum


def as_timing_guess(lyrics: Lyrics) -> Lyrics:
    """CDG lyrics for audio of another version: text and line order are kept,
    the line start is only a first guess and the syllable timing is dropped
    so the audio is aligned afresh. Noise lines (no letters) are dropped."""
    lines: List[Line] = []
    for line in lyrics.lines:
        text = clean_ocr_text(line.text)
        if not _has_letters(text) or line.start is None:
            continue
        lines.append(Line(text=text, start=line.start,
                          stanza_break_before=line.stanza_break_before))
    return Lyrics(lines=lines, language=lyrics.language,
                  source=SOURCE_FOREIGN)
