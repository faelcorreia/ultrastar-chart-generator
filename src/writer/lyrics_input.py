"""Turn `Lyrics` (or the legacy syllable dicts) into writer entries."""
from typing import Any, Dict, List

from src.model.lyrics import Lyrics


def lyrics_to_syllable_dicts(lyrics: Lyrics) -> List[Dict[str, Any]]:
    """Flatten timed Lyrics into the dict form the writer works on.

    A syllable starts a word unless it has `word_start = False`
    (default True). The last syllable of each line gets `line_break`.
    Untimed syllables keep start/end None and are interpolated later.
    """
    result: List[Dict[str, Any]] = []
    for line in lyrics.lines:
        for index, syllable in enumerate(line.syllables):
            result.append({
                "syllable": syllable.text,
                "start": syllable.start,
                "end": syllable.end,
                "line_break": index == len(line.syllables) - 1,
                "first_syllable": getattr(syllable, "word_start", True),
            })
    return result


def syllable_dicts_to_lyrics(items: List[Dict[str, Any]],
                             language: str = None) -> Lyrics:
    """Inverse of `lyrics_to_syllable_dicts` (lines split at line_break)."""
    from src.model.lyrics import Line, Syllable
    lines, current = [], []
    for item in items:
        syllable = Syllable(text=item["syllable"], start=item.get("start"),
                            end=item.get("end"))
        if not item.get("first_syllable", True):
            syllable.word_start = False
        current.append(syllable)
        if item.get("line_break", False):
            lines.append(Line(syllables=current))
            current = []
    if current:
        lines.append(Line(syllables=current))
    return Lyrics(lines=lines, language=language)
