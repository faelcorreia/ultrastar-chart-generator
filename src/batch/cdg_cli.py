"""Convert a folder of CDG+audio pairs into UltraStar songs.

    python -m src.batch.cdg_cli --cdg-dir DIR --output-dir OUT [--language en]

Each `Artist - Title.cdg` is paired with the audio file of the same stem
(`.mp3`, `.mp2`, ...). The CDG supplies the lyric text and its timing; the
normal pipeline separates vocals, measures pitch/BPM and writes the folder.
`--language` is both the Tesseract language for the OCR and the song language
(omit it to let the language stage resolve from the audio).
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

from src.batch.locate import DEFAULT_LIBRARY, split_song_name, strip_label
from src.batch.report import BatchReport, REPORT_NAME
from src.batch.runner import run_batch
from src.batch.songlist import ListEntry, song_key
from src.planner import JobResult


def entries_from_cdg_folder(folder, language: Optional[str] = None,
                            limit: Optional[int] = None) -> List[ListEntry]:
    """One entry per .cdg file, named from `Artist - Title [label].cdg`."""
    entries: List[ListEntry] = []
    seen = set()
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".cdg":
            continue
        artist, title = split_song_name(path)
        if not artist or not title:
            print(f"[WARNING] ignoring '{path.name}': expected "
                  "'Artist - Title.cdg'", file=sys.stderr)
            continue
        entry = ListEntry(artist, strip_label(title), None, language)
        key = (entry.artist.casefold(), entry.title.casefold())
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)
        if limit and len(entries) >= limit:
            break
    return entries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.batch.cdg_cli")
    p.add_argument("--cdg-dir", required=True,
                   help="folder with CDG+audio pairs named 'Artist - Title.*'")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--language", default=None,
                   help="Tesseract/song language (e.g. en, pt, es)")
    p.add_argument("--library", default=DEFAULT_LIBRARY,
                   help="songs already here are skipped")
    p.add_argument("--force-existing", action="store_true",
                   help="process songs even if already in the library")
    p.add_argument("--force", action="store_true",
                   help="ignore the per-stage cache")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N songs (useful for a first test)")
    p.add_argument("--report", default=None,
                   help=f"report/state JSON (default: OUTPUT_DIR/{REPORT_NAME})")
    return p


def main(argv: Optional[List[str]] = None) -> List[JobResult]:
    args = build_parser().parse_args(argv)
    entries = entries_from_cdg_folder(args.cdg_dir, args.language, args.limit)
    if not entries:
        print(f"[ERROR] no 'Artist - Title.cdg' found in {args.cdg_dir}",
              file=sys.stderr)
        return []

    from src.planner import default_stages
    from src.providers.lrclib import LrclibClient
    from src.providers.musicbrainz import MusicBrainzClient

    cache = Path(args.output_dir) / ".cache"
    stages = default_stages(
        args.output_dir,
        lrclib=LrclibClient(cache_dir=cache / "lrclib"),
        musicbrainz=MusicBrainzClient(
            "ultrastar-song-creator/0.1 (CDG CLI)",
            cover_dir=cache / "covers"))

    def show(entry, res):
        print(f"[{res.status}] {entry.artist} - {entry.title}: {res.reason}",
              file=sys.stderr)

    report = BatchReport(args.report or Path(args.output_dir) / REPORT_NAME)
    results = run_batch(entries, args.cdg_dir, args.output_dir, stages,
                        library=args.library,
                        force_existing=args.force_existing, force=args.force,
                        on_result=show, report=report)
    print(report.summary([song_key(e.artist, e.title) for e in entries]),
          file=sys.stderr)
    return results


if __name__ == "__main__":
    main()
