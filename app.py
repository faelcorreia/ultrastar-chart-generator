import re
import sys
from pathlib import Path

import src.core.config as config
from src.core.language import LanguageError, resolve_language
from src.model.legacy import job_from_song
from src.model.lyrics import Lyrics
from src.planner import DONE, FAILED, default_stages, run_job


# Input files follow the "Artist - Song Title - <suffix>.<ext>" convention.
# Matching the suffix explicitly (instead of using rsplit on "-") also works
# when the artist or title contains hyphens.
_INPUT_PATTERNS = (
    (re.compile(r"^(?P<base>.+?)\s+-\s+full track$", re.IGNORECASE), "full_track"),
    (re.compile(r"^(?P<base>.+?)\s+-\s+cover$", re.IGNORECASE), "cover"),
    (
        re.compile(
            r"^(?P<base>.+?)\s+-\s+"
            r"(?:(?P<language>[A-Za-z]{2,5}(?:[-_][A-Za-z]{2,5})?)\s+)?"
            r"lyrics$",
            re.IGNORECASE,
        ),
        "lyrics",
    ),
)

_REQUIRED_FILES = ("full_track", "lyrics", "cover")


def _parse_input_file(path: Path) -> dict | None:
    """Extract the song identity, language and file role from a filename."""
    for pattern, role in _INPUT_PATTERNS:
        match = pattern.match(path.stem)
        if not match:
            continue

        base = match.group("base").strip()
        artist, separator, title = base.partition(" - ")
        if not separator or not artist.strip() or not title.strip():
            print(
                f"[WARNING] Ignoring '{path.name}': expected "
                "'Artist - Song Title - <suffix>'."
            )
            return None

        return {
            "artist_song": base,
            "artist": artist.strip(),
            "title": title.strip(),
            "role": role,
            "language": (match.groupdict().get("language") or "").lower() or None,
            "path": path,
        }

    return None


def _collect_songs(input_dir: Path) -> dict:
    songs: dict[str, dict] = {}

    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue

        parsed = _parse_input_file(path)
        if parsed is None:
            continue

        artist_song = parsed["artist_song"]
        song = songs.setdefault(artist_song, {
            "artist": parsed["artist"],
            "title": parsed["title"],
            "language": parsed["language"],
        })

        role = parsed["role"]
        if role in song:
            print(
                f"[WARNING] Duplicate {role} for '{artist_song}'; using "
                f"'{path.name}' and ignoring '{Path(song[role]).name}'."
            )
        song[role] = path

        if role == "lyrics" and parsed["language"]:
            song["language"] = parsed["language"]

    valid_songs = {}
    for artist_song, song in songs.items():
        missing = [role for role in _REQUIRED_FILES if role not in song]
        if missing:
            print(
                f"[ERROR] Skipping '{artist_song}': missing "
                f"{', '.join(missing)} file(s)."
            )
            continue
        valid_songs[artist_song] = song

    return valid_songs


def _main(argv=None, input_dir=None, output_dir=None, stages=None) -> int:
    """Run every collected song through the planner (K3).

    `argv` defaults to the command line, `input_dir`/`output_dir` to the
    configured folders and `stages` to ``default_stages(output_dir)``; the
    last three exist so tests can inject fakes. Exit code 1 when a song fails.
    """
    input_dir = Path(input_dir or config.INPUT_DIR)
    output_dir = Path(output_dir or config.OUTPUT_DIR)
    if not input_dir.is_dir():
        print(f"[ERROR] INPUT_DIR does not exist: {input_dir}")
        return 1

    songs = _collect_songs(input_dir)
    # Optional arguments restrict the run to the named "Artist - Title" songs.
    # --force recomputes every stage instead of resuming from .work caches.
    args = sys.argv[1:] if argv is None else list(argv)
    force = "--force" in args
    wanted = [arg for arg in args if arg != "--force"]
    if wanted:
        unknown = [name for name in wanted if name not in songs]
        if unknown:
            print(f"[ERROR] Not found in {input_dir}: {', '.join(unknown)}")
            return 1
        songs = {name: songs[name] for name in wanted}
    if not songs:
        print(
            f"[ERROR] No complete songs found in {input_dir}. Expected files "
            "such as 'Artist - Title - full track.mp3', "
            "'Artist - Title - en lyrics.txt' and 'Artist - Title - cover.jpg'."
        )
        return 1

    # Explicit language: list/filename, tags, then Whisper (flagged), else fail.
    jobs = {}
    for name, song in songs.items():
        notes: list[str] = []
        try:
            song["language"] = resolve_language(
                song.get("language"), song.get("full_track"),
                notes=notes, label=f"'{name}'",
            )
        except LanguageError as exc:
            print(f"[ERROR] {exc}")
            return 1
        job = job_from_song(song)
        job.notes.extend(notes)
        # The lyrics are plain text here: the planner aligns them to the audio.
        job.lyrics = Lyrics.from_text(
            song["lyrics"].read_text(encoding="utf-8"),
            language=job.language, source=str(song["lyrics"]))
        jobs[name] = job

    print(f"Found {len(jobs)} song(s):")
    for name, job in jobs.items():
        print(f"  - {name} [{job.language}] ({job.artist} / {job.title})")
        for note in job.notes:
            print(f"    [NOTE] {note}")

    if stages is None:
        stages = default_stages(output_dir)

    failed = []
    for name, job in jobs.items():
        print(f"\nAnalyzing {name}")
        result = run_job(job, output_dir, stages, force=force, song_name=name)
        if result.ran:
            print(f"   ran: {', '.join(result.ran)}")
        if result.cached:
            print(f"   cached: {', '.join(result.cached)}")
        print(f"=> {result.status}: {result.reason}")
        if result.status == DONE:
            print(f"   {job.extra.get('output_folder', output_dir / name)}")
        elif result.status == FAILED:
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} song(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
