"""F1: batch CLI. `python -m src.batch.cli lista.csv --audio-dir DIR`."""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.batch.locate import DEFAULT_LIBRARY
from src.batch.report import REPORT_NAME, BatchReport
from src.batch.runner import run_batch
from src.batch.songlist import parse_song_list, song_key
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
    p.add_argument("--report", default=None,
                   help=f"report/state JSON (default: OUTPUT_DIR/{REPORT_NAME})")
    p.add_argument("--retry-all", action="store_true",
                   help="reprocess songs even if done/skipped in the report")
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

    report = BatchReport(args.report
                         or Path(args.output_dir) / REPORT_NAME)
    results = run_batch(entries, args.audio_dir, args.output_dir, stages,
                        library=args.library,
                        force_existing=args.force_existing, force=args.force,
                        on_result=show, report=report,
                        retry_all=args.retry_all)
    print(report.summary([song_key(e.artist, e.title) for e in entries]),
          file=sys.stderr)
    return results


if __name__ == "__main__":
    main()
