"""D4: reconcile downloaded lyrics with the Whisper transcription.

Both word sequences are aligned (difflib, on language-folded tokens). Whisper
words with no lyric counterpart are matched against two kinds of candidate:

* **displaced verse**: lyric lines that found no match elsewhere. They are
  moved to where Whisper heard them.
* **missing chorus**: lyric lines that already matched earlier in the audio.
  They are copied to the gap (the previous chorus) and confirmed by CTC on
  the audio window through an injectable ``confirmer``; if it does not
  match (or no confirmer is given) the issue is flagged.

Lyric lines Whisper never heard stay in place as **lost passage**. Confidence
is the fraction of lyric words matched after the fixes; below 80% the song
status is ``review``. Pure logic: the CTC confirmer is injected.

``confirmer(audio_path, words, language, window) -> float`` returns a match
score in 0..1 (e.g. share of words CTC places with good posterior).
"""
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from src.core.text_norm import fold_for_compare
from src.model.lyrics import Line, Lyrics
from src.stages.word_alignment import normalize_language

REVIEW_THRESHOLD = 0.80
Confirmer = Callable[[Path, List[str], str, Tuple[float, float]], float]
# (text, start, end) in seconds
WhisperWord = Tuple[str, float, float]

MISSING_CHORUS = "missing_chorus"
DISPLACED_VERSE = "displaced_verse"
LOST_PASSAGE = "lost_passage"


@dataclass
class Issue:
    kind: str
    lines: List[int]                 # original lyric line indices involved
    text: str
    window: Optional[Tuple[float, float]] = None   # audio window (seconds)
    inserted_after: Optional[int] = None           # original line index
    confirmed: Optional[bool] = None               # CTC result (chorus)
    flagged: bool = False
    note: str = ""


@dataclass
class Decision:
    name: str
    rule: str
    outcome: str


@dataclass
class ReconcileReport:
    lyrics: Lyrics
    language: str
    status: str                      # "ok" | "review"
    confidence_before: float
    confidence: float
    issues: List[Issue] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)

    def issues_of(self, kind: str) -> List[Issue]:
        return [i for i in self.issues if i.kind == kind]


def _tokens(lines: Sequence[str], language: str):
    toks, owner = [], []
    for li, text in enumerate(lines):
        for w in fold_for_compare(text, language).split():
            toks.append(w)
            owner.append(li)
    return toks, owner


def _match(a: List[str], b: List[str]):
    sm = SequenceMatcher(None, a, b, autojunk=False)
    pairs = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            pairs[blk.b + k] = blk.a + k      # whisper idx -> lyric idx
    return pairs


def _line_matched(owner, n_lines, matched_lyric):
    total = [0] * n_lines
    hit = [0] * n_lines
    for ti, li in enumerate(owner):
        total[li] += 1
        hit[li] += ti in matched_lyric
    return [hit[i] / total[i] if total[i] else 1.0 for i in range(n_lines)]


def _runs(flags: Sequence[bool]):
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    return out


def _ratio(a, b):
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0


def confidence_of(lines: Sequence[str], whisper_tokens: List[str],
                  language: str) -> float:
    toks, _ = _tokens(lines, language)
    if not toks:
        return 0.0
    return len(_match(toks, whisper_tokens)) / len(toks)


def reconcile_lyrics(
    lyrics: Lyrics,
    whisper_words: Sequence[WhisperWord],
    audio_path: Optional[Path] = None,
    language: Optional[str] = None,
    confirmer: Optional[Confirmer] = None,
    min_gap_words: int = 5,
    similarity: float = 0.6,
    confirm_threshold: float = 0.5,
    max_sec_per_word: float = 2.0,
) -> ReconcileReport:
    lang = normalize_language(language or lyrics.language)
    texts = [l.text for l in lyrics.lines]
    expanded = []                    # one entry per folded token
    for text, st, en in whisper_words:
        for tok in fold_for_compare(text, lang).split():
            expanded.append((tok, st, en))
    wtoks = [e[0] for e in expanded]
    whisper_words = expanded
    ltoks, owner = _tokens(texts, lang)
    pairs = _match(ltoks, wtoks)
    before = len(pairs) / len(ltoks) if ltoks else 0.0
    matched_lyric = set(pairs.values())
    frac = _line_matched(owner, len(texts), matched_lyric)
    unmatched_line = [f < 0.5 for f in frac]
    first_w = {}                     # lyric line -> first whisper idx matched
    for w, t in sorted(pairs.items()):
        first_w.setdefault(owner[t], w)

    def tokens_of(a, b):
        return [t for t, o in zip(ltoks, owner) if a <= o < b]

    gaps = _runs([i not in pairs for i in range(len(wtoks))])
    issues: List[Issue] = []
    removed = set()
    inserts = []                     # (after_line, seq, [text], origin)
    used_spans = []
    for gs, ge in gaps:
        if ge - gs < min_gap_words:
            continue
        gtok = wtoks[gs:ge]
        window = (whisper_words[gs][1], whisper_words[ge - 1][2])
        prev = [owner[t] for w, t in pairs.items() if w < gs]
        after = max(prev) if prev else -1
        best = (0.0, None)
        # displaced: unmatched lyric spans
        for a, b in _runs(unmatched_line):
            if any(x in removed for x in range(a, b)):
                continue
            r = _ratio(tokens_of(a, b), gtok)
            if r > best[0]:
                best = (r, ("displaced", a, b))
        # missing chorus: earlier matched spans
        for a in range(0, after + 1):
            for b in range(a + 1, min(after + 2, a + 13)):
                if any(unmatched_line[a:b]):
                    break
                if first_w.get(a, 1 << 30) >= gs:
                    continue
                r = _ratio(tokens_of(a, b), gtok)
                if r > best[0] + 1e-9:
                    best = (r, ("chorus", a, b))
        if best[1] is None or best[0] < similarity:
            continue
        kind, a, b = best[1]
        span = texts[a:b]
        if kind == "displaced":
            removed.update(range(a, b))
            inserts.append((after, len(inserts), span))
            issues.append(Issue(DISPLACED_VERSE, list(range(a, b)),
                                " / ".join(span), window, after,
                                note="moved to where Whisper heard it"))
        else:
            confirmed = None
            span_words = [w for t in span for w in t.split()]
            too_long = (window[1] - window[0]) > max_sec_per_word * max(
                len(span_words), 1)
            if too_long:
                # The gap is much longer than the chorus could take: it is
                # other material, and spreading the copy over it would give
                # the lines wrong times (I10, Avril).
                confirmed = False
            elif confirmer is not None and audio_path is not None:
                try:
                    score = float(confirmer(
                        audio_path, [w for t in span for w in t.split()],
                        lang, window))
                except Exception:
                    score = 0.0
                confirmed = score >= confirm_threshold
            # I10: only a CTC-confirmed chorus is inserted (with the window
            # as its time); otherwise it is flagged and left out, because a
            # copy with a wrong time hurts the sync more than a missing one.
            if confirmed:
                inserts.append((after, len(inserts), span))
            issues.append(Issue(
                MISSING_CHORUS, list(range(a, b)), " / ".join(span), window,
                after, confirmed, flagged=not confirmed,
                note=("copied from previous chorus" if confirmed else
                      "not inserted: audio window too long for the chorus"
                      if too_long else
                      "not inserted: CTC did not confirm the chorus"
                      if confirmed is False else
                      "not inserted: not confirmed (no CTC)")))

    for a, b in _runs(unmatched_line):
        if all(x in removed for x in range(a, b)):
            continue
        a2 = [x for x in range(a, b) if x not in removed]
        if len(tokens_of(a, b)) >= 1 and a2:
            issues.append(Issue(LOST_PASSAGE, a2, " / ".join(texts[x] for x in a2),
                                flagged=True,
                                note="Whisper did not hear these lines"))

    # Rebuild line order.
    new_lines: List[Line] = []
    by_after = {}
    for after, seq, span in inserts:
        by_after.setdefault(after, []).append(span)

    def add(text_lines, orig=None):
        for t in text_lines:
            new_lines.append(Line(text=t))

    for span in by_after.get(-1, []):
        add(span)
    for i, line in enumerate(lyrics.lines):
        if i not in removed:
            new_lines.append(line)
        for span in by_after.get(i, []):
            add(span)

    after_toks, _ = _tokens([l.text for l in new_lines], lang)
    conf = (len(_match(after_toks, wtoks)) / len(after_toks)
            if after_toks else 0.0)
    flagged = any(i.flagged and i.kind == MISSING_CHORUS for i in issues)
    status = "review" if conf < REVIEW_THRESHOLD or flagged else "ok"
    fixed = Lyrics(lines=new_lines, language=lang, source=lyrics.source)
    decisions = [
        Decision("missing_chorus",
                 "copy previous chorus, insert only if CTC confirms the audio window; else flag",
                 "; ".join(f"lines {i.lines}: "
                           f"{'confirmed' if i.confirmed else 'flagged'}"
                           for i in issues if i.kind == MISSING_CHORUS)
                 or "none found"),
        Decision("review_threshold", "below 80% matched words -> review",
                 f"{conf:.0%} matched -> {status}"),
        Decision("languages", "en, pt, es", lang),
    ]
    return ReconcileReport(fixed, lang, status, before, conf, issues, decisions)
