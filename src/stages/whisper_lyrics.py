"""I3: lyrics from Whisper when LRCLIB has none.

The audio is transcribed with word timestamps, the words are split into
lines/stanzas with the I5 segmenter, and the result is timed (line level)
lyrics that continue through the normal stages. The job is flagged for review
because transcribed lyrics are never fully trustworthy.

The language comes from the job (list/tags) or, failing that, from Whisper's
own detection; it is never defaulted. Unsupported or missing -> LanguageError.
"""
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from src.core.language import (
    SUPPORTED, LanguageError, normalize_code, tag_language)
from src.core.text_norm import minimal_case
from src.core.whisper_model import WHISPER_MODEL
from src.model.lyrics import Line, Lyrics
from src.model.song_job import SongJob
from src.segmentation import Word, segment_words

TimedWord = Tuple[str, float, float]
# (audio_path, language or None) -> (words, language detected or None)
Transcriber = Callable[[Path, Optional[str]],
                       Tuple[Sequence[TimedWord], Optional[str]]]

SOURCE = "whisper"
# A transcript that is mostly one repeated token is a hallucination (I4:
# "e e e e ..." on a Portuguese song), not lyrics.
DEGENERATE_MIN_WORDS = 8
DEGENERATE_SHARE = 0.6
NO_LYRICS_WARNING = (
    "WARNING: no lyrics were provided, so the words come from Whisper "
    f"({WHISPER_MODEL}); providing the lyrics is STRONGLY RECOMMENDED "
    "(much better quality)")
# Lines should cover at least this share of the song (or have at least
# LOW_COVERAGE_MIN_LINES lines), else the transcript is flagged as thin.
LOW_COVERAGE_SHARE = 0.25
LOW_COVERAGE_MIN_LINES = 4
LOW_COVERAGE_REASON = ("low line coverage: {lines} line(s) cover "
                       "{share:.0%} of the song")
REVIEW_REASON = ("lyrics not found on LRCLIB; transcribed with Whisper "
                 "(language {lang}, {origin}); review the text. "
                 + NO_LYRICS_WARNING)


def line_coverage(lines: Sequence[Line], duration: Optional[float]
                  ) -> Optional[float]:
    """Share of the song duration covered by the transcribed lines."""
    if not duration or duration <= 0:
        return None
    return min(1.0, sum(max(0.0, l.end - l.start) for l in lines) / duration)


def whisper_transcriber(audio_path, language: Optional[str],
                        model_size: str = WHISPER_MODEL):
    """Default transcriber (loads faster-whisper lazily)."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(audio_path), language=language, word_timestamps=True,
        # VAD drops most sung phrases (Pegasus Fantasy: 4 words with VAD vs
        # 48 without, I4), so it stays off.
        vad_filter=False)
    words: List[TimedWord] = []
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if text and w.end > w.start:
                words.append((text, float(w.start), float(w.end)))
    return words, getattr(info, "language", None)


def lyrics_from_whisper(job: SongJob, transcriber: Transcriber = whisper_transcriber,
                        tag_reader=tag_language) -> Lyrics:
    """Transcribe `job.audio_path`, segment, and flag the job for review."""
    label = f"'{job.artist} - {job.title}'"
    given = job.resolved_language
    origin = "from the song list" if given else None
    lang = None
    if given:
        lang = normalize_code(given)
        if lang is None:
            raise LanguageError(
                f"Unsupported language '{given}' for {label}. "
                f"Supported: {', '.join(SUPPORTED)}.")
    else:
        lang = tag_reader(job.audio_path)
        origin = "from file tags" if lang else None
    vocals = job.vocals_path
    audio = Path(vocals) if vocals and Path(vocals).is_file() \
        else Path(job.audio_path)
    words, detected = transcriber(audio, lang)
    if lang is None:
        lang = normalize_code(detected)
        if lang is None:
            raise LanguageError(
                f"Could not determine the language of {label}: Whisper "
                f"detected '{detected}'. Give it in the song list "
                "(en|pt|es).")
        origin = "DETECTED by Whisper"
        job.note(f"language '{lang}' DETECTED by Whisper (not given); "
                 "please verify")
    elif origin == "from file tags":
        job.note(f"language '{lang}' taken from file tags")
    words = [w for w in words if w[0].strip()]
    if not words:
        raise RuntimeError("Whisper found no words in the audio")
    from collections import Counter
    common = Counter(w[0].lower().strip(".,!?") for w in words).most_common(1)
    if len(words) >= DEGENERATE_MIN_WORDS and \
            common[0][1] / len(words) > DEGENERATE_SHARE:
        raise RuntimeError(
            f"Whisper transcription is degenerate ('{common[0][0]}' is "
            f"{common[0][1]} of {len(words)} words); no usable lyrics")
    seg = segment_words([Word(t, s, e) for t, s, e in words], lang)
    lines = []
    for i, part in enumerate(seg.lines):
        lines.append(Line(text=minimal_case(" ".join(w.text for w in part), lang),
                          start=part[0].start, end=part[-1].end,
                          stanza_break_before=i in seg.stanza_starts))
    job.sources["lyrics"] = SOURCE
    job.extra["review"] = REVIEW_REASON.format(lang=lang, origin=origin)
    job.note(NO_LYRICS_WARNING)
    print(f"[WARN] {label}: {NO_LYRICS_WARNING}", file=sys.stderr)
    share = line_coverage(lines, job.duration)
    low = len(lines) < LOW_COVERAGE_MIN_LINES or (
        share is not None and share < LOW_COVERAGE_SHARE)
    if low:
        job.extra["review"] += "; " + LOW_COVERAGE_REASON.format(
            lines=len(lines), share=share or 0.0)
    return Lyrics(lines=lines, language=lang, source=SOURCE)
