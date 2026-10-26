"""F1: batch CLI. `python -m src.batch.cli lista.csv --audio-dir DIR`."""
import argparse
import sys
from typing import Dict, List, Optional

from src.batch.locate import DEFAULT_LIBRARY
from src.batch.runner import run_batch
from src.batch.songlist import parse_song_list
from src.planner import JobResult, Stage


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.batch.cli")
    p.add_argument("list_file", help="UTF-8 file: artista;título;ano[;idioma]")
    p.add_argument("--audio-dir", required=True,
                   help="folder with audio named 'Artista - Título.ext'")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--library", default=DEFAULT_LIBRARY,
                   help="songs already here are skipped")
    p.add_argument("--force-existing", action="store_true",
                   help="process songs even if already in the library")
    p.add_argument("--force", action="store_true",
                   help="ignore the per-stage cache")
    return p


def main(argv: Optional[List[str]] = None,
         stages: Optional[Dict[str, Stage]] = None) -> List[JobResult]:
    args = build_parser().parse_args(argv)
    entries = parse_song_list(args.list_file)
    if stages is None:
        from src.planner import default_stages
        stages = default_stages(args.output_dir)

    def show(entry, res):
        print(f"[{res.status}] {entry.artist} - {entry.title}: {res.reason}",
              file=sys.stderr)

    return run_batch(entries, args.audio_dir, args.output_dir, stages,
                     library=args.library, force_existing=args.force_existing,
                     force=args.force, on_result=show)


if __name__ == "__main__":
    main()
