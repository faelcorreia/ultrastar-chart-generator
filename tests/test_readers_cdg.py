import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from src.readers.cdg import parse_cdg, read_cdg_zip


def pkt(instr, data=b""):
    data = bytes(data).ljust(16, b"\0")
    return bytes([9, instr]) + b"\0\0" + data + b"\0\0\0\0"


EMPTY = bytes(24)


def tile(row, col, c0=0, c1=1, mask=0x3F, xor=False):
    d = bytes([c0, c1, row, col]) + bytes([mask] * 12)
    return pkt(38 if xor else 6, d)


def clut():
    cols = [(0, 0, 0), (15, 15, 15), (15, 0, 0)] + [(0, 0, 0)] * 5
    d = b""
    for r, g, b in cols:
        v = (r << 8) | (g << 4) | b
        d += bytes([(v >> 6) & 0x3F, v & 0x3F])
    return pkt(30, d)


def build_cdg(lines):
    """lines: list of (tile_row, [(first_col, ntiles, highlight_second)])
    Words are 3 tiles wide with 2 blank tiles between them."""
    packets = [pkt(1, bytes([0])), clut()]
    for row, words in lines:
        for col, n, _ in words:
            for k in range(n):
                packets.append(tile(row, col + k))
    # idle until 2 s
    while len(packets) < 600:
        packets.append(EMPTY)
    events = []
    for row, words in lines:
        for col, n, sec in words:
            for k in range(n):
                events.append((sec + k * 0.1, row, col + k))
    events.sort()
    for sec, row, col in events:
        idx = int(sec * 300)
        while len(packets) < idx:
            packets.append(EMPTY)
        packets.append(tile(row, col, c0=0, c1=2, xor=True))
    return b"".join(packets)


# two lines, 3 words each: columns 2-4, 7-9, 12-14
LINES = [
    (4, [(2, 3, 2.0), (7, 3, 3.0), (12, 3, 4.0)]),
    (9, [(2, 3, 6.0), (7, 3, 7.0), (12, 3, 8.0)]),
]


class CdgTest(unittest.TestCase):
    def fake_ocr(self, texts):
        calls = []

        def ocr(img):
            self.assertEqual(img.ndim, 3)
            self.assertEqual(img.shape[2], 3)
            calls.append(img.shape)
            return texts[len(calls) - 1]
        return ocr

    def test_timing_without_ocr(self):
        res = parse_cdg(build_cdg(LINES))
        lines = res.lyrics.lines
        self.assertEqual(len(lines), 2)
        self.assertAlmostEqual(lines[0].start, 2.0, places=1)
        self.assertAlmostEqual(lines[1].start, 6.0, places=1)
        self.assertEqual(lines[0].text, "")
        self.assertTrue(res.lyrics.is_synced)

    def test_words_with_ocr_accents(self):
        ocr = self.fake_ocr(["Corazón niño don't", "¿Qué pasó, señor?"])
        res = parse_cdg(build_cdg(LINES), ocr=ocr, language="es")
        l0, l1 = res.lyrics.lines
        self.assertEqual(l0.text, "Corazón niño don't")
        self.assertEqual([s.text.strip() for s in l0.syllables],
                         ["Corazón", "niño", "don't"])
        self.assertAlmostEqual(l0.syllables[0].start, 2.0, places=1)
        self.assertAlmostEqual(l0.syllables[1].start, 3.0, places=1)
        self.assertAlmostEqual(l0.syllables[2].start, 4.0, places=1)
        self.assertLessEqual(l0.syllables[0].end, l0.syllables[1].start + 1e-9)
        self.assertEqual(res.lyrics.language, "es")
        # 4 words but 3 ink segments: proportional fallback still timed
        self.assertEqual(len(l1.syllables), 3)
        self.assertTrue(all(s.is_timed for s in l1.syllables))
        self.assertGreaterEqual(l1.syllables[0].start, 6.0 - 1e-6)

    def test_ocr_word_count_mismatch_falls_back(self):
        ocr = self.fake_ocr(["Coração partido de amor", "x y z"])
        res = parse_cdg(build_cdg(LINES), ocr=ocr)
        syl = res.lyrics.lines[0].syllables
        self.assertEqual(len(syl), 4)
        starts = [s.start for s in syl]
        self.assertEqual(starts, sorted(starts))

    def test_zip_mp3_cdg(self):
        with tempfile.TemporaryDirectory() as tmp:
            zp = Path(tmp) / "song.zip"
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("../evil/Canción.mp3", b"ID3fake")
                zf.writestr("Canción.cdg", build_cdg(LINES))
            out = Path(tmp) / "out"
            res = read_cdg_zip(zp, out_dir=out)
            self.assertEqual(res.audio_path, out / "Canción.mp3")
            self.assertEqual(res.audio_path.read_bytes(), b"ID3fake")
            self.assertEqual(len(res.lyrics.lines), 2)

    def test_zip_without_cdg(self):
        with tempfile.TemporaryDirectory() as tmp:
            zp = Path(tmp) / "x.zip"
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("a.mp3", b"x")
            with self.assertRaises(ValueError):
                read_cdg_zip(zp)

    def test_zip_mp2_cdg(self):
        with tempfile.TemporaryDirectory() as tmp:
            zp = Path(tmp) / "song.zip"
            with zipfile.ZipFile(zp, "w") as zf:
                zf.writestr("Song.mp2", b"mp2fake")
                zf.writestr("Song.cdg", build_cdg(LINES))
            out = Path(tmp) / "out"
            res = read_cdg_zip(zp, out_dir=out)
            self.assertEqual(res.audio_path, out / "Song.mp2")
            self.assertEqual(res.audio_path.read_bytes(), b"mp2fake")

    def test_text_drawn_with_xor_is_read(self):
        # Real discs (HM, KaraokeMP3CDG) draw the line itself with XOR and
        # repaint it to highlight; the old reader saw the draw as highlight
        # and snapshotted a blank screen.
        ocr = self.fake_ocr(["Hello big world", "Second line here"])
        res = parse_cdg(build_xor_cdg(LINES), ocr=ocr)
        self.assertEqual([line.text for line in res.lyrics.lines],
                         ["Hello big world", "Second line here"])
        first = res.lyrics.lines[0].syllables
        self.assertAlmostEqual(first[0].start, 2.0, places=1)
        self.assertAlmostEqual(first[1].start, 3.0, places=1)

    def test_countdown_dashes_do_not_shift_first_word(self):
        ocr = self.fake_ocr(["--- Word"])
        res = parse_cdg(build_dash_cdg(), ocr=ocr)
        self.assertEqual(len(res.lyrics.lines), 1)
        line = res.lyrics.lines[0]
        self.assertEqual(line.text, "Word")
        self.assertAlmostEqual(line.start, 2.0, places=1)


def build_xor_cdg(lines):
    """Like build_cdg, but the text itself is drawn with XOR tiles."""
    packets = [pkt(1, bytes([0])), clut()]
    for row, words in lines:
        for col, n, _ in words:
            for k in range(n):
                packets.append(tile(row, col + k, xor=True))
    while len(packets) < 600:
        packets.append(EMPTY)
    events = []
    for row, words in lines:
        for col, n, sec in words:
            for k in range(n):
                events.append((sec + k * 0.1, row, col + k))
    for sec, row, col in sorted(events):
        idx = int(sec * 300)
        while len(packets) < idx:
            packets.append(EMPTY)
        packets.append(tile(row, col, c0=0, c1=2, xor=True))
    return b"".join(packets)


def tile_rows(row, col, rows, c0=0, c1=1):
    mask = bytes([0x3F if r in rows else 0 for r in range(12)])
    return pkt(6, bytes([c0, c1, row, col]) + mask)


def build_dash_cdg():
    """A thin dash countdown before a real word; only the word gets timing."""
    packets = [pkt(1, bytes([0])), clut()]
    for col in range(2, 5):
        packets.append(tile_rows(2, col, {5, 6}))
    for col in range(8, 11):
        packets.append(tile_rows(1, col, set(range(1, 11))))
    while len(packets) < 600:
        packets.append(EMPTY)

    def put(sec, packet):
        idx = int(sec * 300)
        while len(packets) < idx:
            packets.append(EMPTY)
        packets.append(packet)

    for k, col in enumerate(range(2, 5)):
        put(1.0 + k * 0.1, tile(2, col, c0=0, c1=2, xor=True))
    for k, col in enumerate(range(8, 11)):
        put(2.0 + k * 0.1, tile(1, col, c0=0, c1=2, xor=True))
    return b"".join(packets)


if __name__ == "__main__":
    unittest.main()
