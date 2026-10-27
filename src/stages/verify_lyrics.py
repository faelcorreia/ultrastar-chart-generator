"""I2: planner stages ``reconcile`` (D4) and ``sync_validate`` (D5).

Real estimators on the separated vocal (falls back to the full track):

* Whisper word times (``transcriber``, shared with the I3 fallback; one
  transcription per song is reused by both stages);
* CTC/MMS forced alignment of the whole lyric over the audio (``aligner``,
  the same callable the D2/D3 stages use), independent of the source times.
  CTC is also the confirmer D4 uses for a copied chorus.

Both stages leave their report in ``job.extra`` (``reconcile`` /
``sync_validate``, plain dicts) and flag ``job.extra["review"]`` per the D4
(<80% matched words, unconfirmed chorus) and D5 (uncertain lines) rules.
Lyrics that came from Whisper itself are not checked against Whisper.
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.model.lyrics import Line, Lyrics
from src.model.song_job import SongJob
from src.stages.reconcile import reconcile_lyrics
from src.stages.sync_validate import (
    UNCERTAIN, estimate_from_whisper_words, validate_sync)
from src.stages.whisper_lyrics import SOURCE as WHISPER_SOURCE
from src.stages.whisper_lyrics import whisper_transcriber
from src.stages.word_alignment import ctc_window_aligner, split_words

RECONCILE, SYNC_VALIDATE = "reconcile", "sync_validate"


def applies(job: SongJob) -> bool:
    """Lyrics from an external/user source can be checked against audio."""
    lyr = job.lyrics
    return (lyr is not None and bool(lyr.lines)
            and lyr.source != WHISPER_SOURCE
            and job.sources.get("lyrics") != WHISPER_SOURCE)


def flag_review(job: SongJob, reason: str) -> None:
    prev = job.extra.get("review")
    job.extra["review"] = f"{prev}; {reason}" if prev else reason


def analysis_audio(job: SongJob) -> Path:
    """The separated vocals when available, else the full mix."""
    if job.vocals_path is not None and Path(job.vocals_path).is_file():
        return Path(job.vocals_path)
    return Path(job.audio_path)


def make_confirmer(aligner) -> Callable:
    """D4 confirmer: CTC on the audio window; score = share of words placed
    inside the window with a plausible, ordered duration."""
    def confirm(audio_path, words, language, window) -> float:
        bounds = aligner(Path(audio_path), list(words), language, window)
        if not words or len(bounds) != len(words):
            return 0.0
        lo, hi = window
        good, prev_end = 0, lo - 0.5
        for s, e in bounds:
            if (e - s >= 0.04 and s >= lo - 0.5 and e <= hi + 0.5
                    and s >= prev_end - 0.05):
                good += 1
            prev_end = max(prev_end, e)
        return good / len(words)
    return confirm


def make_ctc_estimator(aligner) -> Callable:
    """D5 CTC estimator: align every word over the whole audio and return
    the first word start of each line (None for lines without words)."""
    def estimate(audio_path, lyrics: Lyrics, language: str):
        per_line = [split_words(l.text) for l in lyrics.lines]
        flat = [w for ws in per_line for w in ws]
        if not flat:
            return [None] * len(per_line)
        bounds = aligner(Path(audio_path), flat, language, None)
        if len(bounds) != len(flat):
            return [None] * len(per_line)
        out, pos = [], 0
        for ws in per_line:
            out.append(bounds[pos][0] if ws else None)
            pos += len(ws)
        return out
    return estimate


def _place_inserted(new: Lyrics, original: Lyrics, report) -> Lyrics:
    """Give lines inserted by D4 a start (and end) from the Whisper window,
    when the lyrics are synced, so later stages can window them."""
    if not original.is_synced:
        return new
    orig_ids = {id(l): i for i, l in enumerate(original.lines)}
    pending = {}
    for issue in report.issues:
        if issue.window and issue.inserted_after is not None \
                and issue.kind != "lost_passage":
            pending.setdefault(issue.inserted_after, []).append(issue)
    lines: List[Line] = []
    last, k = -1, 0
    for line in new.lines:
        if id(line) in orig_ids:
            last, k = orig_ids[id(line)], 0
            lines.append(line)
            continue
        issues = pending.get(last) or []
        # lines of consecutive issues follow each other; find the owner
        owner, offset = None, k
        for iss in issues:
            if offset < len(iss.lines):
                owner = iss
                break
            offset -= len(iss.lines)
        k += 1
        if owner is None:
            lines.append(line)
            continue
        a, b = owner.window
        n = len(owner.lines)
        step = (b - a) / n
        lines.append(Line(text=line.text, start=a + offset * step,
                          end=a + (offset + 1) * step,
                          stanza_break_before=line.stanza_break_before))
    return Lyrics(lines=lines, language=new.language, source=new.source)


def make_stages(transcriber=None, aligner=None) -> Dict[str, Callable]:
    transcriber = transcriber or whisper_transcriber
    aligner = aligner or ctc_window_aligner
    memo: Dict[tuple, list] = {}

    def whisper_words(job: SongJob, audio: Path, language: str):
        key = (str(audio), language)
        if key not in memo:
            words, _ = transcriber(audio, language)
            memo[key] = [(str(t), float(s), float(e)) for t, s, e in words]
        return memo[key]

    def reconcile(job: SongJob) -> None:
        lang = job.require_language()
        audio = analysis_audio(job)
        words = whisper_words(job, audio, lang)
        rep = reconcile_lyrics(job.lyrics, words, audio, lang,
                               confirmer=make_confirmer(aligner))
        job.lyrics = _place_inserted(rep.lyrics, job.lyrics, rep)
        job.extra[RECONCILE] = {
            "status": rep.status,
            "confidence_before": round(rep.confidence_before, 3),
            "confidence": round(rep.confidence, 3),
            "issues": [{"kind": i.kind, "lines": i.lines, "text": i.text,
                        "confirmed": i.confirmed, "flagged": i.flagged,
                        "note": i.note} for i in rep.issues],
        }
        if rep.status == "review":
            flagged = [i.kind for i in rep.issues if i.flagged]
            flag_review(job, (
                f"lyrics match Whisper on {rep.confidence:.0%} of words"
                + (f" ({', '.join(sorted(set(flagged)))})" if flagged else "")
                + "; review"))

    def sync_validate(job: SongJob) -> None:
        if not job.lyrics.is_synced:
            return
        lang = job.require_language()
        audio = analysis_audio(job)
        words = whisper_words(job, audio, lang)

        def whisper_est(a, lyrics, language):
            return estimate_from_whisper_words(lyrics, words, language)

        rep = validate_sync(audio, job.lyrics, lang, whisper_est,
                            make_ctc_estimator(aligner))
        job.lyrics = rep.lyrics
        job.extra[SYNC_VALIDATE] = {
            "status": rep.status,
            "ok_pct": round(rep.pct_ok, 1),
            "shifted_pct": round(rep.pct_shifted, 1),
            "uncertain_pct": round(rep.pct_uncertain, 1),
            "ok_after_pct": round(rep.pct_ok_after, 1),
            "correction": rep.correction.kind,
            "ctc_only": rep.ctc_only,
            "uncertain_lines": [
                {"line": c.index, "text": c.text, "source": c.source,
                 "whisper": c.whisper, "ctc": c.ctc}
                for c in rep.lines_of(UNCERTAIN)],
        }
        if rep.ctc_only:
            flag_review(job, (
                "Whisper heard too few lines to confirm the sync; timing "
                f"judged by CTC alone ({rep.pct_uncertain:.0f}% of lines "
                "still uncertain); review"))
        elif rep.status == "review":
            flag_review(job, (
                f"sync uncertain on {rep.pct_uncertain:.0f}% of lines "
                "(Whisper and CTC disagree); review"))

    return {RECONCILE: reconcile, SYNC_VALIDATE: sync_validate}
