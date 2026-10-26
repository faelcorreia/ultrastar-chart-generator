"""Instrumental video with on-screen lyrics via OCR (G4).

Frames are sampled with ffmpeg and read by an OCR engine; both are
injectable. Line start times come from detecting text changes between
consecutive samples; a line ends when the text changes or disappears.
"""
import re
import subprocess
import tempfile
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.model.lyrics import Line, Lyrics

# frame extractor: (video, fps, out_dir) -> [(time_seconds, image_path)]
FrameExtractor = Callable[[str, float, str], List[Tuple[float, str]]]
# OCR: image_path -> recognized text (may be multi-line / empty)
OcrFn = Callable[[str], str]


class VideoOcrError(RuntimeError):
    pass


def ffmpeg_extract_frames(video, fps: float, out_dir) -> List[Tuple[float, str]]:
    out_dir = Path(out_dir)
    pattern = str(out_dir / "f%06d.png")
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps}",
             pattern], capture_output=True, text=True)
    except FileNotFoundError:
        raise VideoOcrError("ffmpeg not found in PATH")
    if r.returncode:
        raise VideoOcrError(f"ffmpeg failed: {r.stderr.strip()}")
    files = sorted(out_dir.glob("f*.png"))
    return [(i / fps, str(f)) for i, f in enumerate(files)]


def pytesseract_ocr(image_path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise VideoOcrError("pytesseract/Pillow not installed")
    return pytesseract.image_to_string(Image.open(image_path))


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", " ", t).strip()


def _clean(text: str) -> str:
    return " ".join(text.split())


def _same(a: str, b: str, threshold: float) -> bool:
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


def lyrics_from_video(video, language: Optional[str] = None, fps: float = 2.0,
                      extractor: Optional[FrameExtractor] = None,
                      ocr: Optional[OcrFn] = None, min_chars: int = 2,
                      similarity: float = 0.85,
                      min_duration: float = 0.0) -> Lyrics:
    """OCR sampled frames into a synced `Lyrics` (one line per text block)."""
    extractor = extractor or ffmpeg_extract_frames
    ocr = ocr or pytesseract_ocr
    with tempfile.TemporaryDirectory() as tmp:
        frames = extractor(str(video), fps, tmp)
        samples = [(t, _clean(ocr(p))) for t, p in frames]
    return lyrics_from_samples(samples, fps, language, min_chars,
                               similarity, min_duration)


def lyrics_from_samples(samples: List[Tuple[float, str]], fps: float = 2.0,
                        language: Optional[str] = None, min_chars: int = 2,
                        similarity: float = 0.85,
                        min_duration: float = 0.0) -> Lyrics:
    """Group consecutive similar OCR texts; keep the most frequent variant."""
    step = 1.0 / fps if fps else 0.0
    groups: List[dict] = []
    cur = None
    for t, text in samples:
        if len(_norm(text)) < min_chars:
            if cur:
                cur["end"] = cur["last"] + step
                groups.append(cur)
                cur = None
            continue
        if cur and _same(cur["ref"], text, similarity):
            cur["last"] = t
            cur["votes"][text] = cur["votes"].get(text, 0) + 1
            continue
        if cur:
            cur["end"] = t
            groups.append(cur)
        cur = {"start": t, "last": t, "ref": text, "votes": {text: 1}}
    if cur:
        cur["end"] = cur["last"] + step
        groups.append(cur)
    lines = []
    for g in groups:
        if g["end"] - g["start"] < min_duration:
            continue
        best = max(g["votes"].items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        lines.append(Line(text=best, start=g["start"], end=g["end"]))
    return Lyrics(lines=lines, language=language, source="video_ocr")
