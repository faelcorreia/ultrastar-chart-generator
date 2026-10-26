import re
import sys
from pathlib import Path

import src.core.config as config
from src.core.vocals_separator import VocalsSeparator
from src.core.ultrastar_formatter import UltraStarFormatter
from src.core.beats_analyzer import BeatsAnalyzer
from src.core.pitches_analyzer import PitchesAnalyzer
from src.core.syllables_transcriber import SyllablesTranscriber
from src.core.language import LanguageError, resolve_language
from src.model.legacy import absorb, job_from_song, song_from_job


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


def _main() -> int:
    input_dir = Path(config.INPUT_DIR)
    if not input_dir.is_dir():
        print(f"[ERROR] INPUT_DIR does not exist: {input_dir}")
        return 1

    songs = _collect_songs(input_dir)
    # Optional arguments restrict the run to the named "Artist - Title" songs.
    wanted = sys.argv[1:]
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
    notes_by_song: dict[str, list[str]] = {}
    for name in list(songs):
        song = songs[name]
        notes_by_song[name] = []
        try:
            song["language"] = resolve_language(
                song.get("language"), song.get("full_track"),
                notes=notes_by_song[name], label=f"'{name}'",
            )
        except LanguageError as exc:
            print(f"[ERROR] {exc}")
            return 1

    # Every song goes through a SongJob; the stages still get the dict view.
    jobs = {name: job_from_song(song) for name, song in songs.items()}
    for name, job in jobs.items():
        job.notes.extend(notes_by_song[name])
    songs = {name: song_from_job(job) for name, job in jobs.items()}

    print(f"Found {len(jobs)} song(s):")
    for artist_song, job in jobs.items():
        print(
            f"  - {artist_song} [{job.language}] "
            f"({job.artist} / {job.title})"
        )
        for note in job.notes:
            print(f"    [NOTE] {note}")

    vocals = VocalsSeparator(songs)
    pitches = PitchesAnalyzer(songs)
    beats = BeatsAnalyzer(songs)
    syllables = SyllablesTranscriber(songs)
    ultrastar = UltraStarFormatter(songs)

    for artist_song, job in jobs.items():
        print(f"\nAnalyzing {artist_song}")

        print("=> Running Demucs to extract vocals from the original audio")
        vocals.separate_vocals(artist_song=artist_song)
        absorb(job, songs[artist_song])

        print("=> Running librosa to extract vocal pitches")
        pitches.analyze_pitches(artist_song=artist_song)
        absorb(job, songs[artist_song])

        print("=> Running librosa to extract BPM from the original audio")
        beats.analyze_beats(artist_song=artist_song)
        absorb(job, songs[artist_song])

        print("=> Running MFA to timestamp syllables")
        syllables.transcribe_vocals(artist_song)
        absorb(job, songs[artist_song])

        print("=> Creating UltraStar song folder")
        ultrastar.format_ultrastar_song_folder(artist_song=artist_song)

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
