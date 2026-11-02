"""Shared helpers for the D2/D3 alignment stages.

Heavy models are injected: an *aligner* is any callable
``(audio_path, words, language, window) -> [(start, end), ...]`` returning
absolute seconds in the full audio. ``ctc_window_aligner`` is the default and
wraps ``src.core.ctc_word_aligner`` (imported lazily).
"""
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from src.model.lyrics import Line, Syllable

SUPPORTED_LANGUAGES = ("en", "pt", "es")

Bounds = List[Tuple[float, float]]
Window = Optional[Tuple[float, float]]
Aligner = Callable[[Path, List[str], str, Window], Bounds]
Syllabifier = Callable[[str, str], List[str]]


def normalize_language(language: Optional[str]) -> str:
    """Return en/pt/es; never guesses a default."""
    if not language:
        raise ValueError("Language not resolved: pass en, pt or es")
    base = language.lower().replace("_", "-").split("-")[0]
    if base not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}' (supported: "
            f"{', '.join(SUPPORTED_LANGUAGES)})"
        )
    return base


def split_words(text: str) -> List[str]:
    return [w for w in text.split() if any(c.isalnum() for c in w)]


def ctc_window_aligner(audio_path: Path, words: List[str], language: str,
                       window: Window = None) -> Bounds:
    """Default aligner: CTC emissions of the whole file (computed once),
    aligned on the span of the window."""
    from src.core import ctc_word_aligner

    return ctc_word_aligner.align_words(audio_path, words, language, window)


def pyphen_syllabifier(word: str, language: str) -> List[str]:
    """Orthographic split; whole word when pyphen has no dictionary."""
    try:
        import pyphen
        dic = pyphen.Pyphen(lang={"en": "en_US", "pt": "pt_BR",
                                  "es": "es"}[language])
        parts = dic.inserted(word).split("-")
        return [p for p in parts if p] or [word]
    except Exception:
        return [word]


def even_bounds(count: int, start: float, end: float) -> Bounds:
    step = (end - start) / max(count, 1)
    return [(start + i * step, start + (i + 1) * step) for i in range(count)]


def build_line(words: Sequence[str], bounds: Bounds, language: str,
               syllabifier: Optional[Syllabifier] = None) -> Line:
    """A Line whose syllables carry the word times, split by character count."""
    syllables: List[Syllable] = []
    for i, (word, (start, end)) in enumerate(zip(words, bounds)):
        end = max(end, start + 0.001)
        parts = syllabifier(word, language) if syllabifier else [word]
        if not parts or "".join(parts) != word:
            parts = [word]
        total = sum(len(p) for p in parts)
        cursor = start
        for j, part in enumerate(parts):
            seg_end = end if j == len(parts) - 1 else (
                cursor + (end - start) * len(part) / total)
            text = part + (" " if j == len(parts) - 1 and i < len(words) - 1
                           else "")
            syllables.append(Syllable(text=text, start=cursor, end=seg_end))
            cursor = seg_end
    return Line(text=" ".join(words), syllables=syllables)
