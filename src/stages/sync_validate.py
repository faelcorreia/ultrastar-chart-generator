"""D5: validate the sync of external lyrics (LRC, KAR, CDG) against the audio.

The audio is the truth. Every source line start is measured by two
independent estimators (Whisper words and CTC forced alignment on the
separated vocals), both injected:

``estimator(audio_path, lyrics, language) -> List[Optional[float]]``
returns one start time (seconds) per lyric line, or None when unknown.

Per line:

* ``ok``: source start within ``TOLERANCE`` (0.5 s) of the estimate.
* ``shifted``: both estimators agree with each other (within tolerance) and
  disagree with the source.
* ``uncertain``: the estimators disagree with each other (or one is
  missing). It is never corrected on its own and sends the song to review.

Auto-correction, simplest to finest: global offset, linear drift, realign
only the bad lines. The one that most improves the match on the lines where
the estimators agree wins; ties go to the simpler one.
"""
from dataclasses import dataclass, field
from itertools import combinations
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from typing import Callable, List, Optional, Sequence, Tuple

from src.core.text_norm import fold_for_compare
from src.model.lyrics import Line, Lyrics, Syllable
from src.stages.word_alignment import normalize_language

TOLERANCE = 0.5
# A global offset/drift needs several agreeing anchors; one or two noisy
# lines must not shift the whole song (found in the I4 real runs).
MIN_GLOBAL_ANCHORS = 3
# Whisper is often deaf to sung vocals (I4: 0-10% of lines on two English
# songs). Below this coverage its silence is not a disagreement, so CTC
# alone stands in for it and the report says so.
MIN_WHISPER_COVERAGE = 0.3
MIN_GLOBAL_MATCH = 0.5
# I9 (CTC-only judging): coarse tolerance for the consensus model, how far an
# estimate may sit from the model and still be trusted for its line, the share
# of anchors that must agree, and how a drift must beat a plain offset.
COARSE_TOL = 1.25
TRUST_TOL = 1.5
MIN_MODEL_SHARE = 0.3
DRIFT_GAIN = 1.15
MIN_DRIFT_SPAN = 20.0
DRIFT_RANGE = (0.8, 1.25)
OK, SHIFTED, UNCERTAIN = "ok", "shifted", "uncertain"
NONE, OFFSET, DRIFT, LINES = "none", "offset", "drift", "lines"
ROBUST = "robust"

Estimator = Callable[[Path, Lyrics, str], List[Optional[float]]]
WhisperWord = Tuple[str, float, float]


@dataclass
class LineCheck:
    index: int
    text: str
    source: Optional[float]
    whisper: Optional[float]
    ctc: Optional[float]
    status: str
    corrected: Optional[float] = None


@dataclass
class Correction:
    kind: str = NONE
    offset: float = 0.0
    scale: float = 1.0               # drift: new = scale * old + offset
    lines: List[int] = field(default_factory=list)
    match_before: float = 0.0
    match_after: float = 0.0


@dataclass
class SyncReport:
    lyrics: Lyrics                   # corrected
    language: str
    checks: List[LineCheck]
    correction: Correction
    pct_ok: float
    pct_shifted: float
    pct_uncertain: float
    pct_ok_after: float
    status: str                      # "ok" | "review"
    ctc_only: bool = False           # Whisper too deaf; CTC alone judged

    def lines_of(self, status: str) -> List[LineCheck]:
        return [c for c in self.checks if c.status == status]


# A window of Whisper words counts as the line's opening when its text is at
# least this similar (difflib ratio) to the line's first words. Whisper
# mishears sung words ("knocking on the heaven door"), so an exact match
# finds almost nothing there (I6).
FUZZY_CUTOFF = 0.72
FIRST_WORD_CUTOFF = 0.5
LOOKAHEAD = 3


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def estimate_from_whisper_words(lyrics: Lyrics, words: Sequence[WhisperWord],
                                language: str, probe: int = 3,
                                cutoff: float = FUZZY_CUTOFF,
                                ) -> List[Optional[float]]:
    """Whisper-based estimator helper: the start of the first run of Whisper
    words, in order, that approximately matches the first ``probe`` words of
    each line (``cutoff`` = 1.0 gives the old exact matcher)."""
    lang = normalize_language(language)
    toks = [fold_for_compare(w[0], lang) for w in words]
    out: List[Optional[float]] = []
    cursor = 0
    for line in lyrics.lines:
        lw = fold_for_compare(line.text, lang).split()[:probe]
        target = " ".join(lw)
        found, size = None, len(lw)
        # Window of the same length, or one word shorter/longer (Whisper
        # splits and merges words).
        sizes = sorted({max(1, len(lw) - 1), len(lw), len(lw) + 1})

        def best_at(i):
            # The window must start on a word like the line's first word, so
            # a window sliding over the end of the previous line is no match.
            if _similarity(lw[0], toks[i]) < FIRST_WORD_CUTOFF:
                return 0.0, 0
            top, top_n = 0.0, 0
            for n in sizes:
                if i + n <= len(toks):
                    sc = _similarity(target, " ".join(toks[i:i + n]))
                    if sc > top:
                        top, top_n = sc, n
            return top, top_n

        for i in range(cursor, len(toks)) if lw else ():
            best, n = best_at(i)
            if best >= cutoff:
                # Take the best start among the next few words.
                for j in range(i + 1, min(i + LOOKAHEAD, len(toks))):
                    sc, nj = best_at(j)
                    if sc > best + 1e-9:
                        i, best, n = j, sc, nj
                found, size = i, n
                break
        if found is None:
            out.append(None)
        else:
            out.append(words[found][1])
            cursor = found + size
    return out


def _classify(source, w, c, tol) -> Tuple[str, Optional[float]]:
    """Return (status, consensus estimate or None)."""
    if source is None or w is None or c is None:
        return UNCERTAIN, None
    if abs(w - c) > tol:
        return UNCERTAIN, None
    est = (w + c) / 2
    return (OK if abs(est - source) <= tol else SHIFTED), est


def _fit_line(pairs: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Least squares est = a*src + b with one trimming pass."""
    def lsq(p):
        n = len(p)
        mx = sum(x for x, _ in p) / n
        my = sum(y for _, y in p) / n
        sxx = sum((x - mx) ** 2 for x, _ in p)
        if sxx < 1e-9:
            return 1.0, my - mx
        a = sum((x - mx) * (y - my) for x, y in p) / sxx
        return a, my - a * mx
    a, b = lsq(pairs)
    if len(pairs) > 3:
        res = [abs(a * x + b - y) for x, y in pairs]
        cut = median(res) * 3 + 0.05
        kept = [p for p, r in zip(pairs, res) if r <= cut]
        if len(kept) >= 2:
            a, b = lsq(kept)
    return a, b


def fit_robust_model(anchors: List[Tuple[float, float]],
                     tol: float = COARSE_TOL) -> Optional[Tuple[float, float]]:
    """Consensus fit est = a*src + b that survives most estimates being
    garbage (RANSAC over offsets, then over line pairs for drift). Returns
    (a, b) when at least ``MIN_MODEL_SHARE`` of the anchors (and 3) agree
    within ``tol``; a drift only wins over a plain offset when it explains
    clearly more anchors."""
    need = max(MIN_GLOBAL_ANCHORS, MIN_MODEL_SHARE * len(anchors))
    best_n, best = 0, None
    for s0, e0 in anchors:
        inl = [(s, e) for s, e in anchors if abs(e - s - (e0 - s0)) <= tol]
        if len(inl) > best_n:
            best_n, best = len(inl), (1.0, median(e - s for s, e in inl))
    offset_n = best_n
    for (s1, e1), (s2, e2) in combinations(anchors, 2):
        if abs(s2 - s1) < MIN_DRIFT_SPAN:
            continue
        a = (e2 - e1) / (s2 - s1)
        if not DRIFT_RANGE[0] <= a <= DRIFT_RANGE[1]:
            continue
        b = e1 - a * s1
        inl = [(s, e) for s, e in anchors if abs(a * s + b - e) <= tol]
        if len(inl) > max(best_n, offset_n * DRIFT_GAIN):
            a, b = _fit_line(inl)
            best_n, best = len(inl), (a, b)
    if best is None or best_n < need:
        return None
    return best


def _match(anchors, fn) -> float:
    if not anchors:
        return 0.0
    good = sum(1 for src, est in anchors if abs(fn(src) - est) <= TOLERANCE)
    return good / len(anchors)


def _apply(lyrics: Lyrics, fn_for_line) -> Lyrics:
    lines = []
    for i, line in enumerate(lyrics.lines):
        fn = fn_for_line(i)
        if fn is None or line.start is None:
            lines.append(line)
            continue
        delta = fn(line.start) - line.start

        def sh(v):
            return None if v is None else v + delta
        lines.append(Line(
            text=line.text, start=sh(line.start), end=sh(line.end),
            syllables=[Syllable(s.text, sh(s.start), sh(s.end), s.pitch)
                       for s in line.syllables]))
    return Lyrics(lines=lines, language=lyrics.language, source=lyrics.source)


def validate_sync(
    audio_path, lyrics: Lyrics, language: Optional[str],
    whisper_estimator: Estimator, ctc_estimator: Estimator,
    tolerance: float = TOLERANCE,
) -> SyncReport:
    lang = normalize_language(language or lyrics.language)
    audio = Path(audio_path)
    n = len(lyrics.lines)
    ws = list(whisper_estimator(audio, lyrics, lang))
    cs = list(ctc_estimator(audio, lyrics, lang))
    ws += [None] * (n - len(ws))
    cs += [None] * (n - len(cs))

    heard = sum(1 for w in ws[:n] if w is not None)
    ctc_only = n > 0 and heard / n < MIN_WHISPER_COVERAGE
    checks, est = [], []
    for i, line in enumerate(lyrics.lines):
        w = cs[i] if (ctc_only and ws[i] is None) else ws[i]
        status, e = _classify(line.start, w, cs[i], tolerance)
        checks.append(LineCheck(i, line.text, line.start, ws[i], cs[i], status))
        est.append(e)

    anchors = [(lyrics.lines[i].start, e) for i, e in enumerate(est)
               if e is not None]
    before = _match(anchors, lambda s: s)
    best = Correction(NONE, match_before=before, match_after=before)
    best_fn = None
    best_lines = None

    # I9: CTC alone is noisy on some sung audio (quiet intro, soft onsets), so
    # replacing every line by its own estimate would wreck the source's good
    # relative timing. Fit one robust model first and keep it for the lines
    # the estimate does not support.
    robust = None
    if ctc_only and anchors:
        model = fit_robust_model(anchors)
        if model is not None:
            a, b = model
            fixed = {i: a * line.start + b
                     for i, line in enumerate(lyrics.lines)
                     if line.start is not None}
            for i, e in enumerate(est):
                if e is not None and i in fixed \
                        and abs(e - fixed[i]) <= TRUST_TOL:
                    fixed[i] = e
            robust = Correction(
                ROBUST, offset=b, scale=a, match_before=before,
                match_after=_match(
                    [(s, e) for s, e in anchors],
                    lambda s: a * s + b))
            best, best_fn, best_lines = robust, None, fixed

    if robust is None and anchors and before < 1.0:
        cands = []
        off = median(e - s for s, e in anchors)
        cands.append((Correction(OFFSET, offset=off),
                      lambda s, o=off: s + o))
        if len(anchors) >= 3:
            a, b = _fit_line(anchors)
            cands.append((Correction(DRIFT, scale=a, offset=b),
                          lambda s, a=a, b=b: a * s + b))
        for corr, fn in cands:
            corr.match_before = before
            corr.match_after = _match(anchors, fn)
            if (len(anchors) >= MIN_GLOBAL_ANCHORS
                    and corr.match_after >= MIN_GLOBAL_MATCH
                    and corr.match_after > best.match_after + 1e-9):
                best, best_fn = corr, fn
        bad = [i for i, c in enumerate(checks) if c.status == SHIFTED]
        if bad:
            fixed = sum(1 for s, e in anchors if abs(s - e) <= tolerance)
            fixed += sum(1 for i in bad if est[i] is not None)
            after = fixed / len(anchors)
            if after > best.match_after + 1e-9:
                best = Correction(LINES, lines=bad, match_before=before,
                                  match_after=after)
                best_fn, best_lines = None, bad

    if best.kind == ROBUST:
        best.lines = sorted(i for i, v in best_lines.items()
                            if est[i] is not None and v == est[i])
        corrected = _apply(lyrics, lambda i: (
            (lambda s, v=best_lines[i]: v) if i in best_lines else None))
        for i, v in best_lines.items():
            checks[i].corrected = v
    elif best.kind in (OFFSET, DRIFT):
        corrected = _apply(lyrics, lambda i: best_fn)
        for c in checks:
            if c.source is not None:
                c.corrected = best_fn(c.source)
    elif best.kind == LINES:
        def fix(i):
            if i in best_lines:
                return lambda s, e=est[i]: e
            return None
        corrected = _apply(lyrics, fix)
        for i in best_lines:
            checks[i].corrected = est[i]
    else:
        corrected = _apply(lyrics, lambda i: None)
    for c in checks:
        if c.corrected is None:
            c.corrected = c.source

    def pct(status):
        return 100.0 * sum(1 for c in checks if c.status == status) / n if n else 0.0

    ok_after = 0.0
    if n:
        ok_after = 100.0 * sum(
            1 for c, e in zip(checks, est)
            if e is not None and c.corrected is not None
            and abs(c.corrected - e) <= tolerance) / n
    return SyncReport(
        lyrics=corrected, language=lang, checks=checks, correction=best,
        pct_ok=pct(OK), pct_shifted=pct(SHIFTED), pct_uncertain=pct(UNCERTAIN),
        pct_ok_after=ok_after,
        status=("review" if ctc_only or any(
            c.status == UNCERTAIN for c in checks) else "ok"),
        ctc_only=ctc_only)
