#!/usr/bin/env python
"""Compare Whisper models (small/medium/large) by WER and time.

Report only: does not change the pipeline default and downloads models only
when you run it (faster-whisper fetches them on first use).

Manifest JSON: [{"language": "en", "audio": "a.mp3", "reference": "a.txt"}, ...]

    .venv/bin/python scripts/compare_whisper_models.py manifest.json \
        --models small medium large-v3 --out docs/whisper-results.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.text_norm import fold_for_compare  # noqa: E402


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def transcribe(model, audio: str, language: str) -> str:
    segments, _ = model.transcribe(audio, language=language)
    return " ".join(s.text.strip() for s in segments)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--models", nargs="+",
                    default=["small", "medium", "large-v3"])
    ap.add_argument("--out", help="write the Markdown table here")
    args = ap.parse_args(argv)

    items = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    base = Path(args.manifest).parent
    from faster_whisper import WhisperModel

    rows = []
    for name in args.models:
        t0 = time.time()
        model = WhisperModel(name, device="cpu", compute_type="int8")
        load = time.time() - t0
        for it in items:
            lang = it["language"]
            ref = (base / it["reference"]).read_text(encoding="utf-8")
            t0 = time.time()
            hyp = transcribe(model, str(base / it["audio"]), lang)
            secs = time.time() - t0
            score = wer(fold_for_compare(ref, lang),
                        fold_for_compare(hyp, lang))
            rows.append((name, lang, Path(it["audio"]).name, score, secs, load))
            print(f"{name} {lang} WER={score:.3f} time={secs:.1f}s")

    lines = ["| model | lang | audio | WER | time (s) | load (s) |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {m} | {l} | {a} | {w:.3f} | {t:.1f} | {ld:.1f} |"
              for m, l, a, w, t, ld in rows]
    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
