import unittest
from pathlib import Path

from src.model.lyrics import Lyrics
from src.stages.plain_alignment import align_plain_lyrics

TEXTS = {
    "en": ("[00:10.00]Don't stop believin'\n[00:14.00]Hold on to that feelin'\n",
           ["Don't stop believin'", "Hold on to that feelin'"]),
    "pt": ("[00:05.00]Coração partido\n[00:09.00]Não vou chorar\n",
           ["Coração partido", "Não vou chorar"]),
    "es": ("[00:01.00]El niño canta mañana\n[00:04.00]¿Qué pasó, señor?\n",
           ["El niño canta mañana", "¿Qué pasó, señor?"]),
}


class FakeAligner:
    """Spreads words evenly in the window (or in 0..len for no window)."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, audio, words, language, window):
        self.calls.append((list(words), language, window))
        if self.fail_on and words[0] == self.fail_on:
            raise RuntimeError("boom")
        s, e = window if window else (0.0, float(len(words)))
        step = (e - s) / len(words)
        return [(s + i * step, s + (i + 1) * step) for i in range(len(words))]


def fake_syl(word, language):
    return [word[:2], word[2:]] if len(word) > 3 else [word]


class PlainTest(unittest.TestCase):
    def test_languages(self):
        for lang, (_, lines) in TEXTS.items():
            with self.subTest(lang):
                lyr = Lyrics.from_text("\n".join(lines) + "\n\n", lang)
                fake = FakeAligner()
                out = align_plain_lyrics(lyr, Path("a.mp3"), aligner=fake,
                                         syllabifier=fake_syl)
                self.assertEqual(len(fake.calls), 1)
                self.assertIsNone(fake.calls[0][2])
                self.assertTrue(out.has_syllable_timing)
                self.assertLess(out.lines[0].end, out.lines[1].start + 1e-6)
                self.assertEqual(out.language, lang)

    def test_blob_is_segmented_by_pauses(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        bounds = [(i * 0.3, i * 0.3 + 0.25) for i in range(12)]
        bounds[6] = (bounds[6][0] + 2.5, bounds[6][1] + 2.5)
        bounds[7:] = [(a + 2.5, b + 2.5) for a, b in bounds[7:]]
        out = align_plain_lyrics(Lyrics.from_text(text, "en"), Path("a.mp3"),
                                 aligner=lambda *a: bounds,
                                 syllabifier=fake_syl)
        self.assertEqual(len(out.lines), 2)
        self.assertTrue(out.lines[1].stanza_break_before)
        self.assertFalse(out.lines[0].stanza_break_before)

    def test_language_required(self):
        with self.assertRaises(ValueError):
            align_plain_lyrics(Lyrics.from_text("hola mundo"), Path("a.mp3"),
                               aligner=FakeAligner())

    def test_bad_count(self):
        with self.assertRaises(RuntimeError):
            align_plain_lyrics(Lyrics.from_text("a b", "en"), Path("a.mp3"),
                               aligner=lambda *a: [(0, 1)])


if __name__ == "__main__":
    unittest.main()
