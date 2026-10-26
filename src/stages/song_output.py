"""I1b: planner stages syllables and write.

Same pattern as ``audio_analysis``: the legacy classes (MFA/CTC transcriber,
UltraStar formatter) are imported lazily and the functions that run them are
injectable. A runner has the signature ``runner(song, name, work_dir)`` for
syllables (returns the dict entries produced) and
``writer(song, name, out_dir, lyrics)`` for write (returns the folder).
"""
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.model.legacy import absorb, song_from_job
from src.model.song_job import SongJob
from src.stages.audio_analysis import _run_legacy, song_name, work_dir

SyllablesRunner = Callable[[Dict[str, Any], str, Path], Dict[str, Any]]
Writer = Callable[[Dict[str, Any], str, Path, Any], Any]


def default_syllables(song, name, work):
    from src.core.syllables_transcriber import SyllablesTranscriber
    return _run_legacy(SyllablesTranscriber, "transcribe_vocals",
                       ("syllables",), song, name, work)


def default_writer(song, name, out_dir, lyrics):
    from src.core.ultrastar_formatter import UltraStarFormatter
    fmt = UltraStarFormatter({name: song})
    fmt.output_dir = Path(out_dir)
    return fmt.format_ultrastar_song_folder(name, lyrics=lyrics)


def _syllables_stage(output_dir, runner: SyllablesRunner):
    def stage(job: SongJob) -> None:
        if job.vocals_path is None:
            raise ValueError("vocals track required but not available")
        if job.lyrics is None or not job.lyrics.lines:
            raise ValueError("lyrics required but not available")
        if job.lyrics.has_syllable_timing:
            # Already timed (word alignment): nothing to transcribe.
            job.extra["syllables"] = job.lyrics
            return
        work = work_dir(output_dir, job)
        work.mkdir(parents=True, exist_ok=True)
        text = work / f"{song_name(job)} - lyrics.txt"
        text.write_text(job.lyrics.text + "\n", encoding="utf-8")
        song = song_from_job(job)
        song["lyrics"] = text
        song["language"] = job.require_language()
        absorb(job, runner(song, song_name(job), work))
    return stage


def _write_stage(output_dir, writer: Writer):
    def stage(job: SongJob) -> None:
        song = song_from_job(job)
        for key, label in (("full_track", "audio"), ("vocals", "vocals"),
                           ("instrumental", "instrumental"),
                           ("cover", "cover"), ("pitches", "pitches"),
                           ("beats", "beats")):
            if key not in song:
                raise ValueError(f"{label} required but not available")
        syl = song.get("syllables")
        lyrics = None
        if syl is None:
            if job.lyrics is None or not job.lyrics.has_syllable_timing:
                raise ValueError("syllables required but not available")
            lyrics = job.lyrics
        elif not isinstance(syl, (str, Path)):
            lyrics = syl
            song.pop("syllables")
        song["language"] = job.resolved_language
        job.extra["output_folder"] = Path(writer(
            song, song_name(job), Path(output_dir), lyrics)
            or Path(output_dir) / song_name(job))
    return stage


def make_stages(output_dir, syllables: Optional[SyllablesRunner] = None,
                writer: Optional[Writer] = None) -> Dict[str, Callable]:
    """The syllables and write stages for the planner."""
    return {
        "syllables": _syllables_stage(output_dir, syllables or default_syllables),
        "write": _write_stage(output_dir, writer or default_writer),
    }
