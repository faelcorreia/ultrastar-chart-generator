"""LRC reader (D1) and Whisper-based global offset correction.

parse_lrc builds a Lyrics with per-line times (and per-word times for
enhanced LRC `<mm:ss.xx>` tags). correct_offset is a separate pure function:
it estimates a global shift from Whisper words and applies it only when the
match with Whisper improves.
"""
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.model.lyrics import Line, Lyrics, Syllable

_STAMP = re.compile(r"\[(\d+):(\d+(?:[.:]\d+)?)\]")
_TAG = re.compile(r"^\s*\[([A-Za-z#]+):([^\]]*)\]\s*$")
_WORD_TAG = re.compile(r"<(\d+):(\d+(?:[.:]\d+)?)>")


def _seconds(minutes: str, rest: str) -> float:
    return int(minutes) * 60 + float(rest.replace(":", "."))


def parse_lrc_tags(text: str) -> Dict[str, str]:
    """ID tags such as ti, ar, al, la, offset (keys lower-cased)."""
    tags: Dict[str, str] = {}
    for raw in text.splitlines():
        m = _TAG.match(raw)
        if m:
            tags.setdefault(m.group(1).lower(), m.group(2).strip())
    return tags


def _offset_seconds(tags: Dict[str, str]) -> float:
    try:
        return int(float(tags.get("offset", "0").replace(",", "."))) / 1000.0
    except ValueError:
        return 0.0


def _build_syllables(body: str, start: float, end: Optional[float],
                     offset: float) -> List[Syllable]:
    parts = _WORD_TAG.split(body)
    # split gives: lead, min, sec, text, min, sec, text, ...
    pieces: List[Tuple[float, str]] = []
    lead = parts[0]
    if lead.strip():
        pieces.append((start, lead))
    for i in range(1, len(parts), 3):
        t = max(0.0, _seconds(parts[i], parts[i + 1]) - offset)
        pieces.append((t, parts[i + 2]))
    syllables: List[Syllable] = []
    for i, (t, txt) in enumerate(pieces):
        if not txt.strip():
            # A bare trailing tag only closes the previous word.
            if syllables and syllables[-1].end is None:
                syllables[-1].end = t
            continue
        syllables.append(Syllable(text=txt, start=t))
    for i, syl in enumerate(syllables):
        if syl.end is None:
            nxt = syllables[i + 1].start if i + 1 < len(syllables) else end
            syl.end = nxt
    if syllables:
        syllables[-1].text = syllables[-1].text.rstrip()
    return syllables


def parse_lrc(text: str, language: Optional[str] = None,
              source: Optional[str] = None,
              apply_offset: bool = True) -> Lyrics:
    """Parse LRC text. Line end is the start of the next stamped line."""
    tags = parse_lrc_tags(text)
    offset = _offset_seconds(tags) if apply_offset else 0.0
    entries: List[Tuple[float, str]] = []
    for raw in text.splitlines():
        if _TAG.match(raw):
            continue
        stamps = list(_STAMP.finditer(raw))
        if not stamps:
            continue
        # Stamps are only the run at the start of the line.
        pos = 0
        lead: List[re.Match] = []
        for m in stamps:
            if raw[pos:m.start()].strip():
                break
            lead.append(m)
            pos = m.end()
        body = raw[pos:].strip("\r\n")
        for m in lead:
            t = max(0.0, _seconds(m.group(1), m.group(2)) - offset)
            entries.append((t, body))
    entries.sort(key=lambda e: e[0])  # stable: keeps file order on ties

    lines: List[Line] = []
    for i, (t, body) in enumerate(entries):
        plain = _WORD_TAG.sub("", body).strip()
        if not plain:
            continue
        end = entries[i + 1][0] if i + 1 < len(entries) else None
        syllables: List[Syllable] = []
        if _WORD_TAG.search(body):
            syllables = _build_syllables(body.strip(), t, end, offset)
        lines.append(Line(text=plain, start=t, end=end, syllables=syllables))
    if language is None:
        language = tags.get("la") or None
    return Lyrics(lines=lines, language=language, source=source)


def read_lrc(path, language: Optional[str] = None) -> Lyrics:
    data = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return parse_lrc(text, language=language, source=str(path))


# ---------------------------------------------------------------- offset

def shift_lyrics(lyrics: Lyrics, delta: float) -> Lyrics:
    """Return a copy with every time moved by delta seconds (clamped at 0)."""
    def mv(v):
        return None if v is None else max(0.0, v + delta)
    lines = []
    for ln in lyrics.lines:
        syls = [replace(s, start=mv(s.start), end=mv(s.end))
                for s in ln.syllables]
        lines.append(Line(text=ln.text, start=mv(ln.start), end=mv(ln.end),
                          syllables=syls))
    return replace(lyrics, lines=lines)


def _norm(word: str) -> str:
    word = unicodedata.normalize("NFKD", word.casefold())
    return "".join(c for c in word if c.isalnum())


def _word_fields(w: Any) -> Tuple[str, float]:
    if isinstance(w, dict):
        return str(w.get("word", w.get("text", ""))), float(w["start"])
    if hasattr(w, "start"):
        return str(getattr(w, "word", getattr(w, "text", ""))), float(w.start)
    return str(w[0]), float(w[1])


def _first_token(line: Line) -> str:
    for tok in line.text.split():
        n = _norm(tok)
        if n:
            return n
    return ""


@dataclass
class OffsetResult:
    lyrics: Lyrics
    shift: float
    applied: bool
    score_before: int
    score_after: int


def _score(lines: List[Line], tokens: List[str],
           words: Dict[str, List[float]], shift: float, tol: float) -> int:
    n = 0
    for ln, tok in zip(lines, tokens):
        if tok and any(abs(t - (ln.start + shift)) <= tol
                       for t in words.get(tok, ())):
            n += 1
    return n


def estimate_offset(lyrics: Lyrics, whisper_words: Iterable[Any],
                    max_shift: float = 30.0, tolerance: float = 0.5
                    ) -> Tuple[float, int, int]:
    """Best global shift (seconds to add to the LRC), with match counts.

    Returns (shift, score_at_zero, score_at_shift). Pure.
    """
    words: Dict[str, List[float]] = {}
    for w in whisper_words:
        text, start = _word_fields(w)
        n = _norm(text)
        if n:
            words.setdefault(n, []).append(start)
    lines = [ln for ln in lyrics.lines if ln.start is not None]
    tokens = [_first_token(ln) for ln in lines]
    before = _score(lines, tokens, words, 0.0, tolerance)
    deltas: List[float] = []
    for ln, tok in zip(lines, tokens):
        for t in words.get(tok, ()):
            d = t - ln.start
            if abs(d) <= max_shift:
                deltas.append(d)
    best, best_votes = 0.0, -1
    for d in sorted(deltas, key=abs):
        close = [x for x in deltas if abs(x - d) <= tolerance / 2]
        # one vote per line is approximated by the delta count; refine below
        if len(close) > best_votes:
            best_votes, best = len(close), sorted(close)[len(close) // 2]
    after = _score(lines, tokens, words, best, tolerance) if deltas else before
    return best, before, after


def correct_offset(lyrics: Lyrics, whisper_words: Iterable[Any],
                   max_shift: float = 30.0, tolerance: float = 0.5
                   ) -> OffsetResult:
    """Shift the LRC by the estimated global offset only if matching improves.

    whisper_words: (word, start[, end]) tuples, dicts with word/start, or
    objects with .word/.start. Input lyrics are never mutated.
    """
    words = list(whisper_words)
    shift, before, after = estimate_offset(lyrics, words, max_shift, tolerance)
    if after > before and abs(shift) > 1e-9:
        return OffsetResult(shift_lyrics(lyrics, shift), shift, True,
                            before, after)
    return OffsetResult(lyrics, 0.0, False, before, before)
