"""I5 segmentation tests. All lyrics here are short invented texts."""
import unittest

from src.segmentation import (
    SegmentationConfig, Word, check_segmentation, count_syllables,
    line_needs_split, segment_text, segment_words,
)


def timed(text, step=0.3, gaps=None):
    """Words of ``text`` spaced by ``step``; ``gaps`` maps word index -> extra
    pause (seconds) before that word."""
    gaps = gaps or {}
    t, out = 0.0, []
    for i, w in enumerate(text.split()):
        t += gaps.get(i, 0.0)
        out.append(Word(w, t, t + 0.25))
        t += step
    return out


BLOB = {
    "en": ("the river runs along the quiet hill and the morning sings to me "
           "we walk beside the water while the silver evening falls"),
    "pt": ("o rio corre devagar pela colina antiga e a manhã canta pra mim "
           "nós andamos junto da água enquanto a tarde prateada cai"),
    "es": ("el río corre despacio por la colina antigua y la mañana canta "
           "para mí caminamos junto al agua mientras la tarde plateada cae"),
}
CHORUS = {
    "en": "oh my little lantern shine\nthe road is long tonight\n"
          "oh my little lantern shine\nthe fields are cold and wide\n"
          "oh my little lantern shine",
    "pt": "ai minha lanterna pequena brilha\na estrada é longa hoje\n"
          "ai minha lanterna pequena brilha\no campo é frio e largo\n"
          "ai minha lanterna pequena brilha",
    "es": "ay mi linterna pequeña brilla\nel camino es largo hoy\n"
          "ay mi linterna pequeña brilla\nel campo es frío y ancho\n"
          "ay mi linterna pequeña brilla",
}


class SyllableTest(unittest.TestCase):
    def test_counts(self):
        cases = [("en", "river", 2), ("en", "make", 1), ("en", "a", 1),
                 ("pt", "canção", 2), ("pt", "água", 2), ("pt", "poesia", 3),
                 ("es", "aire", 2), ("es", "mañana", 3), ("es", "país", 2),
                 ("es", "que", 1)]
        for lang, word, n in cases:
            self.assertEqual(count_syllables(word, lang), n, word)

    def test_unsupported_language(self):
        with self.assertRaises(ValueError):
            count_syllables("x", "fr")


class BlobTest(unittest.TestCase):
    def test_blob_with_pauses_splits_on_pauses(self):
        for lang, text in BLOB.items():
            with self.subTest(lang=lang):
                n = len(text.split())
                words = timed(text, gaps={n // 3: 0.9, 2 * n // 3: 0.9})
                seg = segment_words(words, lang)
                self.assertGreaterEqual(len(seg.lines), 3)
                self.assertEqual(check_segmentation(seg, lang), [])
                # the long pauses are line boundaries
                starts = {line[0].text for line in seg.lines}
                self.assertIn(words[n // 3].text, starts)
                self.assertIn(words[2 * n // 3].text, starts)

    def test_blob_without_pauses_respects_limit(self):
        cfg = SegmentationConfig(max_syllables=12)
        for lang, text in BLOB.items():
            with self.subTest(lang=lang):
                self.assertTrue(line_needs_split(timed(text), lang, cfg))
                seg = segment_words(timed(text), lang, cfg)
                self.assertGreater(len(seg.lines), 1)
                self.assertEqual(check_segmentation(seg, lang, cfg), [])
                flat = [w.text for line in seg.lines for w in line]
                self.assertEqual(flat, text.split())

    def test_configurable_limit(self):
        cfg = SegmentationConfig(max_syllables=6)
        seg = segment_words(timed(BLOB["en"]), "en", cfg)
        self.assertEqual(check_segmentation(seg, "en", cfg), [])
        default = segment_words(timed(BLOB["en"]), "en")
        self.assertGreater(len(seg.lines), len(default.lines))

    def test_stanza_break_on_long_pause(self):
        words = timed("the river runs along the hill and the morning sings "
                      "we walk beside the water in the evening light",
                      gaps={10: 2.5})
        seg = segment_words(words, "en")
        self.assertEqual(len(seg.stanza_starts), 1)
        first = min(seg.stanza_starts)
        self.assertEqual(seg.lines[first][0].text, "we")

    def test_configurable_pause_threshold(self):
        words = timed("one two three four five six seven eight",
                      gaps={4: 0.8})
        self.assertTrue(line_needs_split(words, "en"))
        lax = SegmentationConfig(line_pause=1.0)
        self.assertFalse(line_needs_split(words, "en", lax))

    def test_punctuation_breaks_untimed(self):
        for lang, text in BLOB.items():
            with self.subTest(lang=lang):
                parts = text.split()
                parts[6] += "."
                parts[13] += "!"
                seg = segment_text(" ".join(parts), lang)
                self.assertEqual(check_segmentation(seg, lang), [])
                texts = seg.line_texts()
                self.assertTrue(any(t.endswith(parts[6]) for t in texts))
                self.assertTrue(any(t.endswith(parts[13]) for t in texts))


class ChorusTest(unittest.TestCase):
    def test_repeated_chorus_in_blob_becomes_own_lines(self):
        for lang, text in CHORUS.items():
            with self.subTest(lang=lang):
                blob = text.replace("\n", " ")
                seg = segment_text(blob, lang)
                self.assertEqual(check_segmentation(seg, lang), [])
                chorus = text.split("\n")[0]
                self.assertEqual(seg.line_texts().count(chorus), 3)

    def test_timed_chorus_without_pauses(self):
        for lang, text in CHORUS.items():
            with self.subTest(lang=lang):
                seg = segment_words(timed(text.replace("\n", " ")), lang)
                self.assertEqual(check_segmentation(seg, lang), [])
                self.assertEqual(seg.line_texts().count(
                    text.split("\n")[0]), 3)

    def test_author_lines_and_blank_lines_kept(self):
        text = "the road is long\nthe night is cold\n\nthe fields are wide"
        seg = segment_text(text, "en")
        self.assertEqual(seg.line_texts(), [
            "the road is long", "the night is cold", "the fields are wide"])
        self.assertEqual(seg.stanza_starts, {2})


if __name__ == "__main__":
    unittest.main()
