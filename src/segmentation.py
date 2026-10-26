"""I5: split words into lines and stanzas so lyrics never become one blob.

Pure functions, no audio or models. Evidence, strongest first:

1. pauses between word timings (``line_pause`` starts a new line,
   ``stanza_pause`` starts a new stanza);
2. strong punctuation (. ! ? ...) and, as a soft hint, commas;
3. repeated chorus lines: a phrase that already appears as a line elsewhere
   is split out of a longer run (and also serves as a preferred break);
4. a hard limit of ``max_syllables`` per line (default 12): oversize runs are
   split at the best break (largest pause, punctuation, repeat edges,
   balanced lengths) by dynamic programming.

``check_segmentation`` is the verifiable quality criterion: no line above the
limit (a single word longer than the limit cannot be split and is exempt) and
no pause of ``line_pause`` or more inside a line.

Syllable counting is a vowel-group heuristic for en/pt/es (see
``count_syllables``); it is an estimate, not a hyphenation dictionary.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

SUPPORTED = ("en", "pt", "es")
STRONG_PUNCT = ".!?…"
SOFT_PUNCT = ",;:"
TRAILING_QUOTES = "\"'”’)]»"


@dataclass(frozen=True)
class SegmentationConfig:
    max_syllables: int = 12
    line_pause: float = 0.6      # seconds: gap that forces a line break
    stanza_pause: float = 1.8    # seconds: gap that forces a stanza break
    min_line_syllables: int = 3  # punctuation breaks need this many first


@dataclass
class Word:
    text: str
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass
class Segmentation:
    lines: List[List[Word]]
    stanza_starts: Set[int]  # indexes of lines that open a new stanza (not 0)

    def line_texts(self) -> List[str]:
        return [" ".join(w.text for w in line) for line in self.lines]


# ---------------------------------------------------------------- syllables

_STRONG = set("aeoáéóàâêôãõ")
_WEAK = set("iuüíúy")
_ACCENTED_WEAK = set("íú")
_NASAL = set("ãõ")


def _letters(word: str) -> str:
    return "".join(c for c in word.lower() if c.isalpha())


def count_syllables(word: str, language: str) -> int:
    """Vowel-group heuristic, at least 1 for any word with a letter.

    en: vowel groups (aeiouy), minus a silent final ``e`` / ``es`` / ``ed``.
    es/pt: strong vowels (a e o) next to each other are a hiatus (two
    syllables), weak vowels (i u) join their neighbour as a diphthong unless
    accented (í ú); ``qu``/``gu`` before e/i has a silent u; ``ão``/``ãe``
    are one syllable.
    """
    if language not in SUPPORTED:
        raise ValueError(f"Unsupported language '{language}'")
    w = _letters(word)
    if not w:
        return 0
    if language == "en":
        return _count_en(w)
    w = re.sub(r"([qg])u(?=[eiéíê])", r"\1", w)
    count = 0
    prev = ""
    for c in w:
        is_v = c in _STRONG or c in _WEAK
        if not is_v:
            prev = ""
            continue
        if not prev:
            count += 1
        else:
            joins = (c in _WEAK and c not in _ACCENTED_WEAK) or \
                    (prev in _WEAK and prev not in _ACCENTED_WEAK)
            if prev in _NASAL and c in "oei":
                joins = True
            if not joins:
                count += 1
        prev = c
    return max(count, 1)


def _count_en(w: str) -> int:
    w = unicodedata.normalize("NFKD", w)
    w = "".join(c for c in w if not unicodedata.combining(c))
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if n > 1:
        silent_e = re.search(r"[^aeiouyl]e$", w)
        silent_ed = re.search(r"[^aeiouytd]ed$", w)
        silent_es = re.search(r"[^aeiouysxzh]es$", w)
        if silent_e or silent_ed or silent_es:
            n -= 1
    return max(n, 1)


def line_syllables(words: Sequence[Word], language: str) -> int:
    return sum(count_syllables(w.text, language) for w in words)


# ----------------------------------------------------------------- helpers

def _norm(word: str) -> str:
    return _letters(word)


def _trailing_punct(text: str) -> str:
    t = text.rstrip(TRAILING_QUOTES)
    return t[-1] if t else ""


def _gap(a: Word, b: Word) -> Optional[float]:
    if a.end is None or b.start is None:
        return None
    return max(0.0, b.start - a.end)


def _split_at_gaps(words: List[Word], cfg: SegmentationConfig,
                   language: str) -> List[List[Word]]:
    """Strong breaks: long pauses and sentence punctuation."""
    chunks: List[List[Word]] = []
    cur: List[Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        if i + 1 == len(words):
            break
        gap = _gap(w, words[i + 1])
        brk = gap is not None and gap >= cfg.line_pause
        if not brk and _trailing_punct(w.text) in STRONG_PUNCT:
            brk = line_syllables(cur, language) >= cfg.min_line_syllables
        if brk:
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def _repeated_phrases(chunks: List[List[Word]], language: str,
                      cfg: SegmentationConfig) -> Set[Tuple[str, ...]]:
    """Word sequences (>= 3 words, min..max syllables) that occur at least
    twice, inside chunks. Candidates are ranked by occurrences x length, so
    a chorus repeated 3 times beats a longer phrase that repeats twice by
    accident; occurrences already covered by a kept phrase do not count.
    Finds a chorus even when the text is one blob."""
    toks = [[_norm(w.text) for w in ch] for ch in chunks]
    syl = [[count_syllables(w.text, language) for w in ch] for ch in chunks]
    occ: Dict[Tuple[str, ...], List[Tuple[int, int]]] = {}
    longest = max((len(t) for t in toks), default=0)
    for n in range(3, min(longest, cfg.max_syllables) + 1):
        for ci, t in enumerate(toks):
            for i in range(len(t) - n + 1):
                key = tuple(t[i:i + n])
                if all(key) and cfg.min_line_syllables <= sum(
                        syl[ci][i:i + n]) <= cfg.max_syllables:
                    occ.setdefault(key, []).append((ci, i))

    def free(key, places, covered):
        used, kept = set(), []
        for ci, i in places:  # non-overlapping, uncovered occurrences
            span = {(ci, j) for j in range(i, i + len(key))}
            if span & used or span & covered:
                continue
            kept.append(span)
            used |= span
        return kept

    ranked = sorted(((k, p) for k, p in occ.items() if len(p) >= 2),
                    key=lambda kp: (-len(kp[0]) * len(free(kp[0], kp[1],
                                                           set())),
                                    -len(kp[0]), kp[0]))
    covered: Set[Tuple[int, int]] = set()
    found: Set[Tuple[str, ...]] = set()
    for key, places in ranked:
        kept = free(key, places, covered)
        if len(kept) >= 2:
            found.add(key)
            for span in kept:
                covered |= span
    return found


def _repeat_cuts(chunk: List[Word], phrases: Set[Tuple[str, ...]]) -> Set[int]:
    """Indexes (cut before word i) at the edges of known repeated phrases."""
    toks = [_norm(w.text) for w in chunk]
    cuts: Set[int] = set()
    i = 0
    while i < len(chunk):
        hit = 0
        for ph in sorted(phrases, key=len, reverse=True):
            n = len(ph)
            if n < len(chunk) and tuple(toks[i:i + n]) == ph:
                hit = n
                break
        if hit:
            cuts.add(i)
            cuts.add(i + hit)
            i += hit
        else:
            i += 1
    cuts.discard(0)
    cuts.discard(len(chunk))
    return cuts


def _split_oversize(chunk: List[Word], language: str, cfg: SegmentationConfig,
                    hints: Set[int]) -> List[List[Word]]:
    """Optimal partition into lines of <= max syllables (DP over cuts)."""
    n = len(chunk)
    syl = [count_syllables(w.text, language) for w in chunk]
    total = sum(syl)
    if total <= cfg.max_syllables:
        return [chunk]
    pieces = -(-total // cfg.max_syllables)
    target = total / pieces

    def bonus(j: int) -> float:  # cutting after word j-1 (before j)
        if j >= n:
            return 0.0
        b = 0.0
        g = _gap(chunk[j - 1], chunk[j])
        if g:
            b += min(g, 1.5) * 4
        p = _trailing_punct(chunk[j - 1].text)
        if p in STRONG_PUNCT:
            b += 3
        elif p in SOFT_PUNCT:
            b += 1.5
        if j in hints:
            b += 3
        return b

    INF = float("inf")
    best = [INF] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        s = 0
        for i in range(j - 1, -1, -1):
            s += syl[i]
            if s > cfg.max_syllables and i < j - 1:
                break
            if best[i] == INF:
                continue
            cost = best[i] + 1.0 + 0.1 * (s - target) ** 2 - bonus(j)
            if cost < best[j]:
                best[j], back[j] = cost, i
    out: List[List[Word]] = []
    j = n
    while j > 0:
        out.append(chunk[back[j]:j])
        j = back[j]
    return out[::-1]


# --------------------------------------------------------------------- API

def segment_words(words: Sequence[Word], language: str,
                  cfg: SegmentationConfig = SegmentationConfig(),
                  paragraph_breaks: Optional[Set[int]] = None
                  ) -> Segmentation:
    """Split words into lines and stanzas.

    ``paragraph_breaks`` holds word indexes that start a new stanza when the
    source text had blank lines (untimed input).
    """
    if language not in SUPPORTED:
        raise ValueError(f"Unsupported language '{language}'")
    words = list(words)
    if not words:
        return Segmentation([], set())
    pbreaks = paragraph_breaks or set()
    # Hard stanza boundaries first, so chunks never cross them.
    blocks: List[List[Word]] = [[]]
    for i, w in enumerate(words):
        if i > 0 and (i in pbreaks or (
                (_gap(words[i - 1], w) or 0.0) >= cfg.stanza_pause)):
            blocks.append([])
        blocks[-1].append(w)
    chunks_per_block = [_split_at_gaps(b, cfg, language) for b in blocks]
    all_chunks = [c for cs in chunks_per_block for c in cs]
    phrases = _repeated_phrases(all_chunks, language, cfg)

    lines: List[List[Word]] = []
    stanza_starts: Set[int] = set()
    for bi, chunks in enumerate(chunks_per_block):
        if bi > 0:
            stanza_starts.add(len(lines))
        for ch in chunks:
            hints = _repeat_cuts(ch, phrases)
            if hints:
                cuts = [0] + sorted(hints) + [len(ch)]
                parts = [ch[a:b] for a, b in zip(cuts, cuts[1:])]
            else:
                parts = [ch]
            for part in parts:
                lines.extend(_split_oversize(part, language, cfg, set()))
    return Segmentation(lines, stanza_starts)


def segment_text(text: str, language: str,
                 cfg: SegmentationConfig = SegmentationConfig()
                 ) -> Segmentation:
    """Untimed text: keep good lines, split blobs by punctuation, repeats and
    length. Blank lines in the text become stanza breaks."""
    words: List[Word] = []
    pbreaks: Set[int] = set()
    forced: Set[int] = set()  # newlines the author wrote are line breaks
    pending_blank = False
    for raw in text.splitlines():
        toks = [t for t in raw.split() if any(c.isalnum() for c in t)]
        if not toks:
            pending_blank = bool(words)
            continue
        if words:
            forced.add(len(words))
            if pending_blank:
                pbreaks.add(len(words))
        pending_blank = False
        words.extend(Word(t) for t in toks)
    # Author line breaks are honoured by segmenting each source line apart,
    # but repeat detection still looks at the whole text.
    bounds = [0] + sorted(forced) + [len(words)]
    src_lines = [words[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]
    chunks: List[List[Word]] = []
    for sl in src_lines:
        chunks.extend(_split_at_gaps(sl, cfg, language))
    phrases = _repeated_phrases(chunks, language, cfg)
    lines: List[List[Word]] = []
    stanza_starts: Set[int] = set()
    pos = 0
    for ch in chunks:
        if pos in pbreaks and lines:
            stanza_starts.add(len(lines))
        hints = _repeat_cuts(ch, phrases)
        cuts = [0] + sorted(hints) + [len(ch)]
        for a, b in zip(cuts, cuts[1:]):
            lines.extend(_split_oversize(ch[a:b], language, cfg, set()))
        pos += len(ch)
    return Segmentation(lines, stanza_starts)


def check_segmentation(seg: Segmentation, language: str,
                       cfg: SegmentationConfig = SegmentationConfig()
                       ) -> List[str]:
    """Quality criterion. Returns violations; empty list means it passes."""
    problems: List[str] = []
    for i, line in enumerate(seg.lines):
        n = line_syllables(line, language)
        if n > cfg.max_syllables and len(line) > 1:
            problems.append(f"line {i}: {n} syllables > {cfg.max_syllables}")
        for a, b in zip(line, line[1:]):
            g = _gap(a, b)
            if g is not None and g >= cfg.line_pause:
                problems.append(
                    f"line {i}: pause of {g:.2f}s inside the line")
    return problems


def line_needs_split(words: Sequence[Word], language: str,
                     cfg: SegmentationConfig = SegmentationConfig()) -> bool:
    return bool(check_segmentation(Segmentation([list(words)], set()),
                                   language, cfg))
