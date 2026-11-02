"""E1: planner. From a SongJob, list the stages still needed and run them.

Every stage is a callable ``func(job) -> None`` that mutates the job; they are
injected through a ``{name: callable}`` dict so tests use fakes. Stage results
are cached per song (A5) as the job fields the stage changed, and a stage that
already ran is never listed again (``job.extra["done"]``).

A job with lyrics but no audio goes to the "awaiting audio" queue: the lyrics
(and language/year/album) are stored in the work folder, the result says
``awaiting_audio``, and the next run with audio adopts them without a new
lookup.

Public entry point for the batch CLI: ``run_job(job, output_dir, stages)``.
A stage can end the song with ``NeedsReview(reason)``, or flag it by setting
``job.extra["review"] = reason`` (the pipeline then finishes with "review").
"""
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.model.song_job import SongJob
from src.stages.cache import StageCache, fingerprint
from src.stages.verify_lyrics import applies as checkable

DONE = "done"
FAILED = "failed"
REVIEW = "review"
AWAITING_AUDIO = "awaiting_audio"

CDG = "cdg"
NORMALIZE, METADATA, LYRICS, LANGUAGE = "normalize", "metadata", "lyrics", "language"
VOCALS, ALIGN, ALIGN_WINDOW, ALIGN_PLAIN = "vocals", "align", "align_window", "align_plain"
PITCHES, BEATS, SYLLABLES, WRITE = "pitches", "beats", "syllables", "write"
RECONCILE, SYNC_VALIDATE = "reconcile", "sync_validate"

Stage = Callable[[SongJob], None]
_AWAITING_STAGE = "awaiting_audio"
_AWAITING_KEY = "v1"


class NeedsReview(Exception):
    """Raised by a stage to end the song with status "review"."""


@dataclass
class JobResult:
    status: str
    reason: str
    job: SongJob
    ran: List[str] = field(default_factory=list)
    cached: List[str] = field(default_factory=list)
    awaiting_audio: bool = False


def has_audio(job: SongJob) -> bool:
    return job.audio_path is not None and Path(job.audio_path).is_file()


def _done(job: SongJob) -> List[str]:
    return job.extra.setdefault(DONE, [])


def _align_name(job: SongJob) -> str:
    if job.lyrics is None:
        return ALIGN
    return ALIGN_WINDOW if job.lyrics.is_synced else ALIGN_PLAIN


def plan(job: SongJob) -> List[str]:
    """Stages still needed, in order, from what the job already knows.

    Without audio only the stages that need none are listed (metadata,
    lyrics). The align step is "align" while the lyrics are unknown, then
    "align_window" (synced) or "align_plain".
    """
    done = set(_done(job))
    audio = has_audio(job)
    steps: List[str] = []
    if job.cdg_path is not None and job.lyrics is None:
        # A CDG disc carries the lyric timing; OCR reads the words. This runs
        # before LYRICS so the online lookup/Whisper never overrides it.
        steps.append(CDG)
    if audio and Path(job.audio_path).suffix.lower() != ".mp3":
        steps.append(NORMALIZE)
    if not (job.artist and job.title and job.year and job.cover_path):
        steps.append(METADATA)
    if job.lyrics is None:
        if audio and job.vocals_path is None:
            # Whisper (when LRCLIB has nothing) works far better on the
            # separated vocals than on the full mix (I4 numbers).
            steps.append(VOCALS)
        steps.append(LYRICS)
    if audio:
        if not job.resolved_language:
            steps.append(LANGUAGE)
        if job.vocals_path is None and VOCALS not in steps:
            steps.append(VOCALS)
        if checkable(job) and job.sources.get("lyrics") != CDG:
            # I2: D4 then D5, before the lyrics are aligned. A CDG disc is
            # the authority on its own timing, so it is not re-checked
            # against Whisper/CTC.
            steps.append(RECONCILE)
            if job.lyrics.is_synced:
                steps.append(SYNC_VALIDATE)
        if job.lyrics is None or not job.lyrics.has_syllable_timing:
            steps.append(ALIGN)
        steps += [PITCHES, BEATS, SYLLABLES, WRITE]
    else:
        # language can still come from an explicit value on the lyrics
        pass
    out = []
    for s in steps:
        if s == ALIGN:
            s = _align_name(job)
        if ALIGN in s and ALIGN in done:
            continue
        if s not in done:
            out.append(s)
    return out


# -- cache of job deltas ---------------------------------------------------
_FIELDS = ("artist", "title", "year", "album", "language", "audio_path",
           "vocals_path", "cover_path", "cdg_path", "duration", "lyrics",
           "sources", "notes", "extra")


def _snapshot(job: SongJob) -> dict:
    return {f: copy.deepcopy(getattr(job, f)) for f in _FIELDS}


def _delta(job: SongJob, before: dict) -> dict:
    return {f: getattr(job, f) for f in _FIELDS
            if getattr(job, f) != before[f]}


def _apply(job: SongJob, delta: dict) -> None:
    for name, value in delta.items():
        if name in ("sources", "extra"):
            getattr(job, name).update(value)
        elif name == "notes":
            job.notes.extend(n for n in value if n not in job.notes)
        else:
            setattr(job, name, value)
    _done(job)


def _store_awaiting(cache: StageCache, job: SongJob) -> None:
    cache.save(_AWAITING_STAGE, _AWAITING_KEY, {
        "lyrics": job.lyrics, "language": job.language, "year": job.year,
        "album": job.album})


def _adopt_awaiting(cache: StageCache, job: SongJob) -> None:
    stored = cache.load(_AWAITING_STAGE, _AWAITING_KEY)
    if not stored:
        return
    if job.lyrics is None and stored.get("lyrics") is not None:
        job.lyrics = stored["lyrics"]
        job.sources.setdefault("lyrics", "awaiting_audio")
    for name in ("language", "year", "album"):
        if getattr(job, name) is None and stored.get(name) is not None:
            setattr(job, name, stored[name])


def run_job(job: SongJob, output_dir, stages: Dict[str, Stage],
            force: bool = False, song_name: Optional[str] = None
            ) -> JobResult:
    """Plan and execute the stages for `job`, using the per-stage cache.

    `stages` maps stage name to ``func(job)``; a missing implementation fails
    the song with a clear reason. Never raises for stage errors.
    """
    name = song_name or f"{job.artist} - {job.title}"
    cache = StageCache(output_dir, name, enabled=not force)
    _adopt_awaiting(cache, job)
    result = JobResult(FAILED, "", job)
    key = fingerprint(job.audio_path if has_audio(job) else None,
                      job.artist, job.title, job.language, job.lyrics)

    while True:
        pending = plan(job)
        if not pending:
            break
        stage = pending[0]
        func = stages.get(stage) or (
            stages.get(ALIGN) if stage.startswith(ALIGN) else None)
        if func is None:
            return _fail(result, f"no implementation for stage '{stage}'")
        if stage == LYRICS:
            run_key = fingerprint(stage, job.artist, job.title, job.album,
                                  job.duration)
            if not (job.artist and job.title):
                return _fail(result, "cannot fetch lyrics without artist and "
                                     "title")
        else:
            key = fingerprint(key, stage)
            run_key = key
        before = _snapshot(job)
        cached = cache.load(stage, run_key)
        try:
            if cached is not None:
                _apply(job, cached)
                result.cached.append(stage)
            else:
                func(job)
                cache.save(stage, run_key, _delta(job, before))
                result.ran.append(stage)
        except NeedsReview as exc:
            result.status, result.reason = REVIEW, str(exc) or stage
            return result
        except Exception as exc:  # noqa: BLE001 - reported, never raised
            return _fail(result, f"{stage}: {exc}")
        _done(job).append(stage)
        if stage == LYRICS and job.lyrics is None:
            return _fail(result, "lyrics not found" + (
                "" if has_audio(job) else " and no audio available"))

    if not has_audio(job):
        if job.lyrics is None:
            return _fail(result, "no audio and no lyrics")
        _store_awaiting(cache, job)
        result.status = AWAITING_AUDIO
        result.awaiting_audio = True
        result.reason = "lyrics stored; waiting for audio"
        return result
    review = job.extra.get("review")
    if review:
        result.status, result.reason = REVIEW, str(review)
    else:
        result.status, result.reason = DONE, "ok"
    return result


def _fail(result: JobResult, reason: str) -> JobResult:
    result.status, result.reason = FAILED, reason
    return result


# -- default stages for the lightweight steps ------------------------------
def default_stages(output_dir, lrclib=None, musicbrainz=None,
                   probe=None, separate=None, pitches=None,
                   beats=None, syllables=None, writer=None,
                   transcriber=None, aligner=None) -> Dict[str, Stage]:
    """Real implementations for every stage. `separate`, `pitches`,
    `beats`, `syllables` and `writer` replace the heavy analyzers and the
    UltraStar writer (tests). `transcriber` is the Whisper word transcriber
    used when LRCLIB has no lyrics (I3) and by reconcile/sync_validate (I2);
    `aligner` is the CTC window aligner they use (default: real MMS)."""
    from src.stages.audio_analysis import make_stages
    from src.stages.song_output import make_stages as make_song_output
    from src.core.language import resolve_language
    from src.readers.metadata import read_metadata
    from src.stages.audio_normalize import normalize_audio
    from src.stages.verify_lyrics import analysis_audio
    from src.stages.verify_lyrics import make_stages as make_verify
    from src.stages.word_alignment import ctc_window_aligner
    from src.stages.plain_alignment import align_plain_lyrics
    from src.stages.window_alignment import align_in_windows
    from src.stages.whisper_lyrics import (
        lyrics_from_whisper, whisper_transcriber)

    out = Path(output_dir)

    def normalize(job):
        job.audio_path = normalize_audio(
            job.audio_path, out / f"{job.artist} - {job.title}" / ".work")

    def metadata(job):
        if has_audio(job):
            read_metadata(job.audio_path, job, probe=probe)
        if musicbrainz is not None:
            musicbrainz.fill(job)

    def cdg(job):
        from src.readers.cdg import read_cdg
        from src.readers.tesseract_ocr import make_ocr
        language = job.language or "en"  # OCR code; None stays unknown
        result = read_cdg(job.cdg_path, ocr=make_ocr(language),
                          language=job.language)
        if not result.lyrics.lines:
            raise RuntimeError(
                f"no lyric lines found in {Path(job.cdg_path).name}")
        job.lyrics = result.lyrics
        job.sources["lyrics"] = "cdg"
        job.note(f"CDG: {len(result.lyrics.lines)} timed lines")

    def lyrics(job):
        if lrclib is None:
            raise RuntimeError("no LRCLIB client configured")
        found = lrclib.find(job.artist, job.title, job.album, job.duration)
        if found is not None:
            job.lyrics = found.to_lyrics(job.language)
            job.sources["lyrics"] = job.lyrics.source or "lrclib"
        elif has_audio(job):
            # I3: no lyrics online -> Whisper transcription, flagged review
            job.lyrics = lyrics_from_whisper(
                job, transcriber or whisper_transcriber)

    def language(job):
        job.language = resolve_language(
            job.resolved_language, job.audio_path, notes=job.notes,
            label=f"'{job.artist} - {job.title}'")

    def align_window(job):
        # I13: CTC on the full mix put words seconds off wherever the
        # instruments masked the voice; the vocals stem keeps them on the audio.
        voice = analysis_audio(job)
        mix = Path(job.audio_path)
        res = align_in_windows(job.lyrics, voice, job.require_language(),
                               aligner=aligner or ctc_window_aligner,
                               alt_audio_path=mix if voice != mix else None)
        job.lyrics = res.lyrics
        if res.failed_lines:
            msg = f"alignment failed on lines {res.failed_lines}"
            prev = job.extra.get("review")
            job.extra["review"] = f"{prev}; {msg}" if prev else msg

    def align_plain(job):
        job.lyrics = align_plain_lyrics(job.lyrics, analysis_audio(job),
                                        job.require_language(),
                                        aligner=aligner or ctc_window_aligner)

    stages = {CDG: cdg, NORMALIZE: normalize, METADATA: metadata,
              LYRICS: lyrics, LANGUAGE: language, ALIGN_WINDOW: align_window,
              ALIGN_PLAIN: align_plain}
    stages.update(make_verify(transcriber, aligner))
    stages.update(make_stages(output_dir, separate, pitches, beats))
    stages.update(make_song_output(output_dir, syllables, writer))
    return stages
