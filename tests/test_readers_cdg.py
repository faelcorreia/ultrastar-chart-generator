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


if __name__ == "__main__":
    unittest.main()
