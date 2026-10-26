"""D3: plain lyrics + audio -> forced alignment without Whisper."""
from pathlib import Path
from typing import List, Optional

from src.model.lyrics import Line, Lyrics
from src.stages.word_alignment import (
    Aligner, Syllabifier, build_line, ctc_window_aligner, normalize_language,
    pyphen_syllabifier, split_words,
)


def align_plain_lyrics(
    lyrics: Lyrics,
    audio_path: Path,
    language: Optional[str] = None,
    aligner: Aligner = ctc_window_aligner,
    syllabifier: Optional[Syllabifier] = pyphen_syllabifier,
) -> Lyrics:
    """Align every word of the lyrics over the whole audio in one CTC pass.

    The language must come from the argument or ``lyrics.language``; it is
    never guessed. Lines without words are kept untimed.
    """
    lang = normalize_language(language or lyrics.language)
    per_line = [split_words(l.text) for l in lyrics.lines]
    flat = [w for ws in per_line for w in ws]
    if not flat:
        raise ValueError("Lyrics have no words to align")
    bounds = aligner(Path(audio_path), flat, lang, None)
    if len(bounds) != len(flat):
        raise RuntimeError(
            f"Aligner returned {len(bounds)} bounds for {len(flat)} words")
    lines: List[Line] = []
    pos = 0
    for line, words in zip(lyrics.lines, per_line):
        if not words:
            lines.append(Line(text=line.text))
            continue
        lines.append(build_line(words, bounds[pos:pos + len(words)], lang,
                                syllabifier))
        pos += len(words)
    return Lyrics(lines=lines, language=lang, source=lyrics.source)
