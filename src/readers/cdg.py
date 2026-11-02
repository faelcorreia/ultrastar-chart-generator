"""CDG reader (G3): highlight commands give timing, OCR (injectable) the text.

A CD+G stream is 24-byte packets at 300 packets/s. The reader replays the
screen (memory preset, tile block, tile block XOR) and treats a tile write as
a *highlight* when it repaints a tile that already carries ink with different
pixels. That is what real discs do: the lyric line is drawn first, then the
singer marks each word by repainting its tiles in another colour. Treating
every XOR write as a highlight (the old reader) fails on discs that draw the
text itself with XOR, which is most of them.

Highlight events are grouped into lyric lines: a new line starts when the
write lands on a row outside the current line's rows and its column goes back
to the left, or after a long pause. That handles both the one-line-per-page
discs (Katy Perry sample) and the four-line screen with an animated dash
countdown (HM sample), where pages are not delimited by memory presets.

The line text is OCR'd from the screen just before the line's first highlight,
cropped to the tall (letter-like) ink so the dashes/bullets of the countdown
never reach the OCR. Word x positions come from that same ink; the highlight
timeline gives each word's start and end. Without OCR the lines carry timing
only, as before.

OCR is an injectable callable ``ocr(image) -> str`` that gets an HxWx3 uint8
RGB numpy array of one line. Scrolls (instructions 20/24) are still not
replayed; a scroll-based disc yields no usable lines instead of wrong ones.
"""
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.model.lyrics import Line, Lyrics, Syllable

PACKET = 24
PACKETS_PER_SECOND = 300.0
WIDTH, HEIGHT = 300, 216
TILE_W, TILE_H = 6, 12

OCR_SCALE = 2  # CDG text is ~12 px tall; OCR does much better at 2x
MAX_LINE_ROWS = 6  # a text line is at most ~48 px tall; more is a graphic
MIN_LINE_EVENTS = 3
MIN_LINE_SECONDS = 0.12
LONG_PAUSE = 4.0  # seconds without a highlight before a new line may start

Ocr = Callable[[np.ndarray], str]


@dataclass
class Highlight:
    time: float
    row: int  # tile row
    col: int  # tile column


@dataclass
class CdgPage:
    """One detected lyric line (the name is kept for compatibility)."""

    start: float
    snapshot: np.ndarray  # indexed screen just before the first highlight
    palette: np.ndarray  # 16x3 uint8
    highlights: List[Highlight] = field(default_factory=list)
    end: Optional[float] = None
    rows: Tuple[int, int] = (0, 0)


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


def _clut_value(data: bytes, i: int) -> int:
    return ((data[2 * i] & 0x3F) << 6) | (data[2 * i + 1] & 0x3F)


def _clut_color(v: int) -> Tuple[int, int, int]:
    return ((v >> 8) & 15) * 17, ((v >> 4) & 15) * 17, (v & 15) * 17


class _Screen:
    """Replays the CDG screen and records repaint (highlight) events."""

    def __init__(self):
        self.px = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        self.clut = np.zeros((16, 3), dtype=np.uint8)
        # Pixel count per colour index, kept incrementally so ``bg`` is O(1).
        self.counts = np.zeros(16, dtype=np.int64)
        self.counts[0] = HEIGHT * WIDTH
        self.events: List[Highlight] = []

    def bg(self) -> int:
        return int(self.counts.argmax())

    def _write(self, y: int, x: int, region: np.ndarray) -> None:
        old = self.px[y:y + TILE_H, x:x + TILE_W]
        self.counts -= np.bincount(old.ravel(), minlength=16)
        self.counts += np.bincount(region.ravel(), minlength=16)
        self.px[y:y + TILE_H, x:x + TILE_W] = region

    def preset(self, color: int) -> None:
        self.px[:] = color
        self.counts[:] = 0
        self.counts[color] = HEIGHT * WIDTH

    def load_clut(self, data: bytes, high: bool) -> None:
        base = 8 if high else 0
        for i in range(8):
            self.clut[base + i] = _clut_color(_clut_value(data, i))

    def tile(self, t: float, data: bytes, xor: bool) -> None:
        c0, c1 = data[0] & 15, data[1] & 15
        row, col = data[2] & 31, data[3] & 63
        y, x = row * TILE_H, col * TILE_W
        if y + TILE_H > HEIGHT or x + TILE_W > WIDTH:
            return
        bits = np.array([[(data[4 + r] >> (5 - c)) & 1 for c in range(6)]
                         for r in range(12)], dtype=np.uint8)
        tile_px = np.where(bits == 1, c1, c0).astype(np.uint8)
        old = self.px[y:y + TILE_H, x:x + TILE_W].copy()
        new = old ^ tile_px if xor else tile_px
        bg = self.bg()
        if ((old != bg).any() and (new != bg).any()
                and not np.array_equal(old, new)):
            self.events.append(Highlight(t, row, col))
        self._write(y, x, new)


def _replay(data: bytes, snapshots_at: Sequence[float] = (),
            stop_after: Optional[float] = None
            ) -> Tuple[_Screen, Dict[float, Tuple[np.ndarray, np.ndarray]]]:
    """Replay the stream, saving the screen just before each requested time.

    Times are exact packet timestamps of the first highlight of a line. The
    snapshot is taken before the packet at that time is applied, so it shows
    the line as it was drawn, before the singer's colour sweep reached it.
    """
    scr = _Screen()
    targets = sorted(set(snapshots_at))
    snaps: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    ti = 0
    for t, instr, d in iter_packets(data):
        while ti < len(targets) and t >= targets[ti]:
            snaps[targets[ti]] = (scr.px.copy(), scr.clut.copy())
            ti += 1
        if stop_after is not None and ti >= len(targets) and t > stop_after:
            break
        if instr == 1:
            scr.preset(d[0] & 15)
        elif instr == 30:
            scr.load_clut(d, False)
        elif instr == 31:
            scr.load_clut(d, True)
        elif instr in (6, 38):
            scr.tile(t, d, instr == 38)
    while ti < len(targets):
        snaps[targets[ti]] = (scr.px.copy(), scr.clut.copy())
        ti += 1
    return scr, snaps


def _cluster(events: List[Highlight]) -> List[dict]:
    """Group highlight events into lyric lines.

    A line stays open while writes fall on its rows. A write on another row
    starts a new line when its column goes back to the left (the sweep
    restarts), or when a long pause separates the two.
    """
    lines: List[dict] = []
    cur: Optional[dict] = None
    for ev in sorted(events, key=lambda h: (h.time, h.row, h.col)):
        if cur is None:
            cur = {"events": [ev], "rows": {ev.row}, "max_col": ev.col,
                   "start": ev.time, "last": ev.time}
            continue
        new_line = (ev.row not in cur["rows"] and ev.col < cur["max_col"]) or (
            ev.time - cur["last"] > LONG_PAUSE and ev.col < cur["max_col"])
        if new_line:
            lines.append(cur)
            cur = {"events": [ev], "rows": {ev.row}, "max_col": ev.col,
                   "start": ev.time, "last": ev.time}
        else:
            cur["events"].append(ev)
            if ev.row not in cur["rows"] and ev.col >= cur["max_col"]:
                cur["rows"].add(ev.row)  # a taller glyph extends the band
            cur["max_col"] = max(cur["max_col"], ev.col)
            cur["last"] = ev.time
    if cur is not None:
        lines.append(cur)
    return lines


def _tall_segments(mask: np.ndarray, word_gap: int) -> List[Tuple[int, int]]:
    """Column spans that carry letter-like ink (dashes and bullets are thin)."""
    tall = np.zeros(mask.shape[1], dtype=bool)
    for x in range(mask.shape[1]):
        col = np.where(mask[:, x])[0]
        if len(col) and col.max() - col.min() >= 6:
            tall[x] = True
    xs = np.where(tall)[0]
    segs: List[Tuple[int, int]] = []
    for x in xs:
        if segs and x - segs[-1][1] <= word_gap:
            segs[-1] = (segs[-1][0], int(x))
        else:
            segs.append((int(x), int(x)))
    return segs


def _has_alnum(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _clean_words(text: str, segs: List[Tuple[int, int]]
                 ) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Drop countdown dashes/bullets at the edges, with their ink spans.

    A dash token usually leaves no tall-ink segment, so a segment is only
    removed when there are more segments than words left.
    """
    words = text.split()
    while words and not _has_alnum(words[0]):
        words.pop(0)
        if len(segs) > len(words):
            segs = segs[1:]
    while words and not _has_alnum(words[-1]):
        words.pop()
        if len(segs) > len(words):
            segs = segs[:-1]
    return words, segs


def _when(timeline: List[Tuple[float, int]], x: int, last: bool) -> float:
    for t, reach in timeline:
        if reach >= x if last else reach > x:
            return t
    return timeline[-1][0]


def _spans(words: List[str], segs: List[Tuple[int, int]],
           ink: np.ndarray) -> List[Tuple[int, int]]:
    if len(segs) == len(words) and words:
        return [(a, b + 1) for a, b in segs]
    xs = np.where(ink.any(axis=0))[0]
    if not len(xs):
        return []
    x, xmax = int(xs[0]), int(xs[-1]) + 1
    total = sum(len(w) for w in words) or 1
    width = xmax - x
    out = []
    for w in words:
        nx = x + width * len(w) / total
        out.append((int(round(x)), int(round(nx))))
        x = nx
    return out


def _line_from_cluster(page: CdgPage, ocr: Optional[Ocr], word_gap: int
                       ) -> Optional[Line]:
    evs = sorted(page.highlights, key=lambda h: h.time)
    r0, r1 = page.rows
    y0, y1 = r0 * TILE_H, (r1 + 1) * TILE_H
    snap = page.snapshot
    bg = int(np.bincount(snap.ravel(), minlength=16).argmax())
    ink = snap != bg
    band = ink[y0:y1]
    text = ""
    words: List[str] = []
    segs: List[Tuple[int, int]] = []
    if ocr is not None:
        segs = _tall_segments(band, word_gap)
        if segs:
            xa = max(0, segs[0][0] - 2)
            xb = min(WIDTH, segs[-1][1] + 3)
        else:
            xa, xb = 0, WIDTH
        img = page.palette[snap[y0:y1, xa:xb]]
        if OCR_SCALE > 1:
            img = np.repeat(np.repeat(img, OCR_SCALE, axis=0),
                            OCR_SCALE, axis=1)
        text = (ocr(np.ascontiguousarray(img)) or "").strip()
        words, segs = _clean_words(text, segs)
        if not words:
            return None
    start = evs[0].time
    end = evs[-1].time
    if ocr is None:
        return Line(text="", start=start, end=end)
    timeline: List[Tuple[float, int]] = []
    reach = 0
    for ev in evs:
        reach = max(reach, (ev.col + 1) * TILE_W)
        timeline.append((ev.time, reach))
    spans = _spans(words, segs, band)
    if not spans:
        return None
    syls: List[Syllable] = []
    prev_end = start
    for i, (w, (a, b)) in enumerate(zip(words, spans)):
        s = max(_when(timeline, a, False), prev_end)
        e = max(_when(timeline, b, True), s)
        syls.append(Syllable(text=w + (" " if i < len(words) - 1 else ""),
                             start=s, end=e))
        prev_end = e
    return Line(text=" ".join(words), start=syls[0].start, end=syls[-1].end,
                syllables=syls)


def _pages(data: bytes, ocr: Optional[Ocr], word_gap: int
           ) -> List[Tuple[CdgPage, Line]]:
    first, _ = _replay(data)
    clusters = _cluster(first.events)
    clusters = [c for c in clusters
                if len(c["events"]) >= MIN_LINE_EVENTS
                and c["last"] - c["start"] >= MIN_LINE_SECONDS]
    # A graphic animation touches many rows at once; a lyric line does not.
    clusters = [c for c in clusters
                if max(c["rows"]) - min(c["rows"]) < MAX_LINE_ROWS]
    # The snapshot must be taken before the first highlight of each line.
    targets = [c["start"] - 1e-6 for c in clusters]
    _, snaps = _replay(data, targets, stop_after=targets[-1] if targets else 0)
    found: List[Tuple[CdgPage, Line]] = []
    for c, target in zip(clusters, targets):
        px, clut = snaps[target]
        rows = (min(c["rows"]), max(c["rows"]))
        page = CdgPage(start=c["start"], snapshot=px, palette=clut,
                       highlights=list(c["events"]), end=c["last"], rows=rows)
        line = _line_from_cluster(page, ocr, word_gap)
        if line is not None:
            found.append((page, line))
    return found


def parse_cdg_pages(data: bytes, ocr: Optional[Ocr] = None,
                    word_gap: int = 7) -> List[CdgPage]:
    """Detected lyric lines with their snapshot and highlight events.

    With an OCR callable, lines whose crop yields no word at all are dropped;
    without it every timed line is returned.
    """
    return [page for page, _ in _pages(data, ocr, word_gap)]


def parse_cdg(data: bytes, ocr: Optional[Ocr] = None,
              language: Optional[str] = None, source: Optional[str] = None,
              word_gap: int = 7) -> CdgResult:
    found = _pages(data, ocr, word_gap)
    return CdgResult(Lyrics(lines=[line for _, line in found],
                            language=language, source=source),
                     [page for page, _ in found])


def read_cdg(path, ocr: Optional[Ocr] = None,
             language: Optional[str] = None) -> CdgResult:
    p = Path(path)
    return parse_cdg(p.read_bytes(), ocr, language, str(p))


def read_cdg_zip(zip_path, out_dir=None, ocr: Optional[Ocr] = None,
                 language: Optional[str] = None) -> CdgResult:
    """Read a zip with MP3/MP2+CDG. The audio is extracted to out_dir (a temp
    dir when None) and returned as audio_path. Member names are never used as
    paths beyond their base name."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [i for i in zf.infolist() if not i.is_dir()]

        def pick(exts: Tuple[str, ...], stem: Optional[str] = None):
            found = [m for m in members
                     if PurePosixPath(m.filename.replace("\\", "/"))
                     .suffix.lower() in exts]
            if stem is not None:
                same = [m for m in found if PurePosixPath(
                    m.filename.replace("\\", "/")).stem.lower() == stem]
                found = same or found
            return found[0] if found else None

        cdg = pick((".cdg",))
        if cdg is None:
            raise ValueError("No .cdg file in zip")
        stem = PurePosixPath(cdg.filename.replace("\\", "/")).stem.lower()
        audio_member = pick((".mp3", ".mp2"), stem)
        audio = None
        if audio_member is not None:
            dest = Path(out_dir) if out_dir else Path(tempfile.mkdtemp())
            dest.mkdir(parents=True, exist_ok=True)
            name = PurePosixPath(
                audio_member.filename.replace("\\", "/")).name
            audio = dest / name
            audio.write_bytes(zf.read(audio_member))
        result = parse_cdg(zf.read(cdg), ocr, language, str(zip_path))
    result.audio_path = audio
    return result
