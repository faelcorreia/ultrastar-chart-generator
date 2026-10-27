"""D2: align words/syllables inside each LRC line's time window.

CTC is restricted to the window of each line (its start to the next line's
start), so a wrong repetition elsewhere in the song cannot capture the words.
MFA keeps handling syllables: pass ``syllable_refiner`` to replace the
orthographic split with phone-based timing.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from src.model.lyrics import Line, Lyrics
from src.stages.word_alignment import (
    Aligner, Syllabifier, build_line, ctc_window_aligner, even_bounds,
    normalize_language, pyphen_syllabifier, split_words,
)


@dataclass
class WindowAlignment:
    lyrics: Lyrics
    failed_lines: List[int] = field(default_factory=list)


def line_windows(lyrics: Lyrics, audio_duration: Optional[float] = None,
                 tail: float = 8.0) -> List[Optional[tuple]]:
    """(start, end) per line; None for lines without a start time."""
    windows: List[Optional[tuple]] = []
    lines = lyrics.lines
    for i, line in enumerate(lines):
        if line.start is None:
            windows.append(None)
            continue
        nxt = next((l.start for l in lines[i + 1:] if l.start is not None),
                   None)
        if nxt is not None:
            end = nxt
        elif audio_duration is not None:
            end = audio_duration
        else:
            end = line.end if line.end and line.end > line.start \
                else line.start + tail
        if audio_duration is not None:
            end = min(end, audio_duration)
        windows.append((line.start, max(end, line.start + 0.05)))
    return windows


def _closer_to_start(bounds, aligner, alt_audio, words, lang, window):
    """``bounds`` or the same line aligned on ``alt_audio``, whichever starts
    nearer the window start (the line's own time)."""
    try:
        alt = aligner(alt_audio, words, lang, window)
    except Exception:
        return bounds
    if len(alt) != len(words):
        return bounds
    if abs(alt[0][0] - window[0]) < abs(bounds[0][0] - window[0]):
        return alt
    return bounds


def align_in_windows(
    lyrics: Lyrics,
    audio_path: Path,
    language: Optional[str] = None,
    aligner: Aligner = ctc_window_aligner,
    syllabifier: Optional[Syllabifier] = pyphen_syllabifier,
    syllable_refiner: Optional[Callable[[Line, str], Line]] = None,
    audio_duration: Optional[float] = None,
    alt_audio_path: Optional[Path] = None,
) -> WindowAlignment:
    """Return a new Lyrics with word/syllable times from per-line CTC.

    A line whose alignment fails keeps even timing inside its window and its
    index is reported in ``failed_lines``.

    ``audio_path`` should be the separated vocals. When ``alt_audio_path``
    (the full mix) is also given, each line is aligned on both and the result
    whose first word lands closer to the line's own start time wins: the
    stem can lose a soft onset the mix still carries, and the mix can place
    a line seconds off where instruments mask the voice (I13).
    """
    lang = normalize_language(language or lyrics.language)
    if not lyrics.is_synced:
        raise ValueError("align_in_windows needs line times (synced lyrics)")
    out_lines: List[Line] = []
    failed: List[int] = []
    for i, (line, window) in enumerate(
            zip(lyrics.lines, line_windows(lyrics, audio_duration))):
        words = split_words(line.text)
        if not words:
            out_lines.append(Line(text=line.text, start=line.start,
                                  end=line.end))
            continue
        try:
            bounds = aligner(Path(audio_path), words, lang, window)
            if len(bounds) != len(words):
                raise RuntimeError("word count mismatch")
            if alt_audio_path is not None:
                bounds = _closer_to_start(
                    bounds, aligner, Path(alt_audio_path), words, lang,
                    window)
        except Exception:
            failed.append(i)
            bounds = even_bounds(len(words), *window)
        new = build_line(words, bounds, lang, syllabifier)
        if syllable_refiner is not None:
            new = syllable_refiner(new, lang)
        out_lines.append(new)
    return WindowAlignment(
        Lyrics(lines=out_lines, language=lang, source=lyrics.source), failed)
