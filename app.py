import re
from pathlib import Path

import src.core.config as config
from src.core.vocals_separator import VocalsSeparator
from src.core.ultrastar_formatter import UltraStarFormatter
from src.core.beats_analyzer import BeatsAnalyzer
from src.core.pitches_analyzer import PitchesAnalyzer
from src.core.syllables_transcriber import SyllablesTranscriber


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
            "language": (match.groupdict().get("language") or "pt").lower(),
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

        if role == "lyrics":
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
    if not songs:
        print(
            f"[ERROR] No complete songs found in {input_dir}. Expected files "
            "such as 'Artist - Title - full track.mp3', "
            "'Artist - Title - en lyrics.txt' and 'Artist - Title - cover.jpg'."
        )
        return 1

    print(f"Found {len(songs)} song(s):")
    for artist_song, song in songs.items():
        print(
            f"  - {artist_song} [{song['language']}] "
            f"({song['artist']} / {song['title']})"
        )

    vocals = VocalsSeparator(songs)
    pitches = PitchesAnalyzer(songs)
    beats = BeatsAnalyzer(songs)
    syllables = SyllablesTranscriber(songs)
    ultrastar = UltraStarFormatter(songs)

    for artist_song in songs:
        print(f"\nAnalyzing {artist_song}")

        print("=> Running Demucs to extract vocals from the original audio")
        vocals.separate_vocals(artist_song=artist_song)

        print("=> Running librosa to extract vocal pitches")
        pitches.analyze_pitches(artist_song=artist_song)

        print("=> Running librosa to extract BPM from the original audio")
        beats.analyze_beats(artist_song=artist_song)

        print("=> Running MFA to timestamp syllables")
        syllables.transcribe_vocals(artist_song)

        print("=> Creating UltraStar song folder")
        ultrastar.format_ultrastar_song_folder(artist_song=artist_song)

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
