"""I1a: planner stages vocals, pitches and beats.

Each stage adapts a legacy ``src.core`` analyzer (dict API) to
``func(job) -> None``. The analyzers are imported lazily (demucs, torch and
librosa are heavy) and the function that runs them is injectable, so tests
use fakes. A runner has the signature ``runner(song, name, work_dir)`` where
`song` is the legacy dict, `name` the "Artist - Title" label and `work_dir`
the per-song ``.work`` folder; it returns the dict entries it produced.
"""
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.model.legacy import absorb, song_from_job
from src.model.song_job import SongJob

Runner = Callable[[Dict[str, Any], str, Path], Dict[str, Any]]


def song_name(job: SongJob) -> str:
    return f"{job.artist} - {job.title}"


def work_dir(output_dir, job: SongJob) -> Path:
    return Path(output_dir) / song_name(job) / ".work"


def _run_legacy(cls, method: str, keys, song, name, work) -> Dict[str, Any]:
    """Run a legacy analyzer with its output folder pointed at `work`."""
    songs = {name: song}
    proc = cls(songs)
    proc.output_dir = work
    getattr(proc, method)(name)
    return {k: songs[name][k] for k in keys if k in songs[name]}


def default_separate(song, name, work):
    from src.core.vocals_separator import VocalsSeparator
    return _run_legacy(VocalsSeparator, "separate_vocals",
                       ("vocals", "instrumental"), song, name, work)


def default_pitches(song, name, work):
    from src.core.pitches_analyzer import PitchesAnalyzer
    return _run_legacy(PitchesAnalyzer, "analyze_pitches", ("pitches",),
                       song, name, work)


def default_beats(song, name, work):
    from src.core.beats_analyzer import BeatsAnalyzer
    return _run_legacy(BeatsAnalyzer, "analyze_beats", ("beats",),
                       song, name, work)


def _make(output_dir, runner: Runner, required: Callable[[dict], None]):
    def stage(job: SongJob) -> None:
        if job.audio_path is None:
            raise ValueError("no audio to analyze")
        work = work_dir(output_dir, job)
        work.mkdir(parents=True, exist_ok=True)
        song = song_from_job(job)
        required(song)
        absorb(job, runner(song, song_name(job), work))
    return stage


def _need(key: str, label: str):
    def check(song):
        if key not in song:
            raise ValueError(f"{label} required but not available")
    return check


def make_stages(output_dir, separate: Optional[Runner] = None,
                pitches: Optional[Runner] = None,
                beats: Optional[Runner] = None) -> Dict[str, Callable]:
    """The vocals, pitches and beats stages for the planner."""
    def beats_input(song):
        # Beats read the instrumental; without one (vocals supplied by the
        # user) fall back to the full track.
        song.setdefault("instrumental", song["full_track"])

    return {
        "vocals": _make(output_dir, separate or default_separate,
                        lambda s: None),
        "pitches": _make(output_dir, pitches or default_pitches,
                         _need("vocals", "vocals track")),
        "beats": _make(output_dir, beats or default_beats, beats_input),
    }
