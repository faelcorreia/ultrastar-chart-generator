"""CDG reader (G3): highlight commands give timing, OCR (injectable) the text.

A CD+G stream is 24-byte packets at 300 packets/s. The parser replays the
screen (memory preset, tile block, tile block XOR) and records "highlight"
events: XOR tile blocks, or normal tile blocks that redraw already inked
tiles with the same shape in other colours. Per page (started by a memory
preset) the text lines are the horizontal ink bands of the screen as it was
just before the first highlight. Highlight progress along x inside a band
gives the time of each word.

OCR is an injectable callable `ocr(image) -> str` that gets an HxWx3 uint8
RGB numpy array of one line band. Without it, lines have empty text and only
timing. Limitations: scrolls are ignored; a page is delimited by memory
presets only.
"""
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.model.lyrics import Line, Lyrics, Syllable

PACKET = 24
PACKETS_PER_SECOND = 300.0
WIDTH, HEIGHT = 300, 216
TILE_W, TILE_H = 6, 12

Ocr = Callable[[np.ndarray], str]


@dataclass
class Highlight:
    time: float
    row: int  # tile row
    col: int  # tile column


@dataclass
class CdgPage:
    start: float
    snapshot: np.ndarray  # indexed screen just before first highlight
    palette: np.ndarray  # 16x3 uint8
    highlights: List[Highlight] = field(default_factory=list)


@dataclass
class CdgResult:
    lyrics: Lyrics
    pages: List[CdgPage] = field(default_factory=list)
    audio_path: Optional[Path] = None


def iter_packets(data: bytes):
    """Yield (time, instruction, 16 data bytes) for CDG command packets."""
    for i in range(len(data) // PACKET):
        pk = data[i * PACKET:(i + 1) * PACKET]
        if pk[0] & 0x3F == 9:
            yield i / PACKETS_PER_SECOND, pk[1] & 0x3F, pk[4:20]


class _Screen:
    def __init__(self):
        self.px = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        self.clut = np.zeros((16, 3), dtype=np.uint8)
        self.page: Optional[CdgPage] = None
        self.pages: List[CdgPage] = []
        self.dirty = False

    def close_page(self):
        if self.page is not None and self.page.highlights:
            self.pages.append(self.page)
        self.page = None

    def preset(self, t: float, color: int):
        if self.dirty or self.page is not None:
            self.close_page()
        self.px[:] = color
        self.dirty = False

    def load_clut(self, data: bytes, high: bool):
        for i in range(8):
            v = ((data[2 * i] & 0x3F) << 6) | (data[2 * i + 1] & 0x3F)
            r, g, b = (v >> 8) & 15, (v >> 4) & 15, v & 15
            self.clut[i + (8 if high else 0)] = (r * 17, g * 17, b * 17)

    def tile(self, t: float, data: bytes, xor: bool):
        c0, c1 = data[0] & 15, data[1] & 15
        row, col = data[2] & 31, data[3] & 63
        y, x = row * TILE_H, col * TILE_W
        if y + TILE_H > HEIGHT or x + TILE_W > WIDTH:
            return
        bits = np.array([[(data[4 + r] >> (5 - c)) & 1 for c in range(6)]
                         for r in range(12)], dtype=np.uint8)
        new = np.where(bits == 1, c1, c0).astype(np.uint8)
        old = self.px[y:y + TILE_H, x:x + TILE_W]
        bg = self._bg()
        highlight = xor
        if not xor:
            same_shape = np.array_equal(old != bg, new != bg)
            highlight = (same_shape and (new != bg).any()
                         and not np.array_equal(old, new))
        if highlight:
            if self.page is None:
                self.page = CdgPage(t, self.px.copy(), self.clut.copy())
            elif not self.page.highlights:
                self.page.snapshot = self.px.copy()
                self.page.palette = self.clut.copy()
            self.page.highlights.append(Highlight(t, row, col))
        else:
            self.dirty = True
        self.px[y:y + TILE_H, x:x + TILE_W] = (old ^ new) if xor else new

    def _bg(self) -> int:
        return int(np.bincount(self.px.ravel(), minlength=16).argmax())


def parse_cdg_pages(data: bytes) -> List[CdgPage]:
    scr = _Screen()
    for t, instr, d in iter_packets(data):
        if instr == 1:
            scr.preset(t, d[0] & 15)
        elif instr == 30:
            scr.load_clut(d, False)
        elif instr == 31:
            scr.load_clut(d, True)
        elif instr in (6, 38):
            scr.tile(t, d, instr == 38)
    scr.close_page()
    return scr.pages


def _bands(ink: np.ndarray, gap: int = 3) -> List[Tuple[int, int]]:
    rows = np.where(ink.any(axis=1))[0]
    bands: List[Tuple[int, int]] = []
    for r in rows:
        if bands and r - bands[-1][1] <= gap + 1:
            bands[-1] = (bands[-1][0], int(r))
        else:
            bands.append((int(r), int(r)))
    return bands


def _segments(cols: np.ndarray, word_gap: int) -> List[Tuple[int, int]]:
    xs = np.where(cols)[0]
    segs: List[Tuple[int, int]] = []
    for x in xs:
        if segs and x - segs[-1][1] <= word_gap:
            segs[-1] = (segs[-1][0], int(x))
        else:
            segs.append((int(x), int(x)))
    return segs


def _page_lines(page: CdgPage, ocr: Optional[Ocr], word_gap: int
                ) -> List[Line]:
    snap = page.snapshot
    bg = int(np.bincount(snap.ravel(), minlength=16).argmax())
    ink = snap != bg
    out: List[Line] = []
    for y0, y1 in _bands(ink):
        events = sorted(
            (h for h in page.highlights
             if h.row * TILE_H <= y1 and (h.row + 1) * TILE_H > y0),
            key=lambda h: h.time)
        if not events:
            continue
        band_ink = ink[y0:y1 + 1]
        xs = np.where(band_ink.any(axis=0))[0]
        xmin, xmax = int(xs[0]), int(xs[-1]) + 1
        timeline: List[Tuple[float, int]] = []
        reach = 0
        for h in events:
            reach = max(reach, (h.col + 1) * TILE_W)
            timeline.append((h.time, reach))

        def when(x: int, last: bool) -> float:
            for t, r in timeline:
                if (r >= x if last else r > x):
                    return t
            return timeline[-1][0]

        text = ""
        if ocr is not None:
            pad = 2
            ya, yb = max(0, y0 - pad), min(HEIGHT, y1 + 1 + pad)
            img = page.palette[snap[ya:yb]]
            text = ocr(np.ascontiguousarray(img)).strip()
        words = text.split()
        spans: List[Tuple[int, int]] = []
        if words:
            segs = _segments(band_ink.any(axis=0), word_gap)
            if len(segs) == len(words):
                spans = [(a, b + 1) for a, b in segs]
            else:
                total = sum(len(w) for w in words)
                x, width = xmin, xmax - xmin
                for w in words:
                    nx = x + width * len(w) / total
                    spans.append((int(round(x)), int(round(nx))))
                    x = nx
        syls: List[Syllable] = []
        prev_end = events[0].time
        for i, (w, (a, b)) in enumerate(zip(words, spans)):
            s = max(when(a, False), prev_end)
            e = max(when(b, True), s)
            syls.append(Syllable(text=w + (" " if i < len(words) - 1 else ""),
                                 start=s, end=e))
            prev_end = e
        out.append(Line(text=text, start=events[0].time,
                        end=events[-1].time, syllables=syls))
    return out


def parse_cdg(data: bytes, ocr: Optional[Ocr] = None,
              language: Optional[str] = None, source: Optional[str] = None,
              word_gap: int = 7) -> CdgResult:
    pages = parse_cdg_pages(data)
    lines: List[Line] = []
    for page in pages:
        lines.extend(_page_lines(page, ocr, word_gap))
    return CdgResult(Lyrics(lines=lines, language=language, source=source),
                     pages)


def read_cdg(path, ocr: Optional[Ocr] = None,
             language: Optional[str] = None) -> CdgResult:
    p = Path(path)
    return parse_cdg(p.read_bytes(), ocr, language, str(p))


def read_cdg_zip(zip_path, out_dir=None, ocr: Optional[Ocr] = None,
                 language: Optional[str] = None) -> CdgResult:
    """Read a zip with MP3+CDG. The MP3 is extracted to out_dir (a temp dir
    when None) and returned as audio_path. Member names are never used as
    paths beyond their base name."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [i for i in zf.infolist() if not i.is_dir()]

        def pick(ext: str, stem: Optional[str] = None):
            found = [m for m in members
                     if PurePosixPath(m.filename.replace("\\", "/"))
                     .suffix.lower() == ext]
            if stem is not None:
                same = [m for m in found if PurePosixPath(
                    m.filename.replace("\\", "/")).stem.lower() == stem]
                found = same or found
            return found[0] if found else None

        cdg = pick(".cdg")
        if cdg is None:
            raise ValueError("No .cdg file in zip")
        stem = PurePosixPath(cdg.filename.replace("\\", "/")).stem.lower()
        mp3 = pick(".mp3", stem)
        audio = None
        if mp3 is not None:
            dest = Path(out_dir) if out_dir else Path(tempfile.mkdtemp())
            dest.mkdir(parents=True, exist_ok=True)
            name = PurePosixPath(mp3.filename.replace("\\", "/")).name
            audio = dest / name
            audio.write_bytes(zf.read(mp3))
        result = parse_cdg(zf.read(cdg), ocr, language, str(zip_path))
    result.audio_path = audio
    return result
