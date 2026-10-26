import unittest

from src.model.lyrics import Lyrics
from src.readers.lrc import (correct_offset, estimate_offset, parse_lrc,
                             parse_lrc_tags, shift_lyrics)

EN = """[ti:Don't Stop]
[ar:Test]
[la:en]
[offset:500]
[00:10.00]Don't stop believin'
[00:14.50]Hold on to that feelin'
"""

PT = """[00:05.00]Coração partido
[00:09.00][00:30.50]Não vou chorar, amor
[00:12.00]
[00:13.00]Saudade é o que sinto
"""

ES = """[00:01.20]El niño canta "mañana"
[00:04.00]¿Qué pasó, señor?
"""

ENH = "[00:10.00]<00:10.00>Hel<00:10.40>lo <00:10.80>wor<00:11.20>ld<00:11.60>\n[00:12.00]next line\n"


class LrcParseTest(unittest.TestCase):
    def test_tags_and_offset(self):
        tags = parse_lrc_tags(EN)
        self.assertEqual(tags["ti"], "Don't Stop")
        self.assertEqual(tags["offset"], "500")
        lyr = parse_lrc(EN)
        self.assertEqual(lyr.language, "en")
        # positive offset: lyrics appear 0.5 s earlier
        self.assertAlmostEqual(lyr.lines[0].start, 9.5)
        self.assertAlmostEqual(lyr.lines[0].end, 14.0)
        self.assertEqual(lyr.lines[0].text, "Don't stop believin'")
        self.assertIsNone(lyr.lines[1].end)
        self.assertEqual(parse_lrc(EN, apply_offset=False).lines[0].start, 10.0)

    def test_negative_offset_clamps(self):
        lyr = parse_lrc("[offset:-250]\n[00:01.00]a b\n")
        self.assertAlmostEqual(lyr.lines[0].start, 1.25)
        lyr = parse_lrc("[offset:2000]\n[00:01.00]a b\n")
        self.assertEqual(lyr.lines[0].start, 0.0)

    def test_multiple_stamps_accents_and_blank(self):
        lyr = parse_lrc(PT)
        texts = [(l.text, l.start) for l in lyr.lines]
        self.assertEqual(texts, [
            ("Coração partido", 5.0),
            ("Não vou chorar, amor", 9.0),
            ("Saudade é o que sinto", 13.0),
            ("Não vou chorar, amor", 30.5),
        ])
        self.assertEqual(lyr.lines[1].end, 12.0)  # blank line closes it
        self.assertTrue(lyr.is_synced)

    def test_spanish(self):
        lyr = parse_lrc(ES, language="es")
        self.assertEqual(lyr.language, "es")
        self.assertEqual(lyr.lines[0].text, 'El niño canta "mañana"')
        self.assertEqual(lyr.lines[1].text, "¿Qué pasó, señor?")
        self.assertEqual(lyr.lines[0].end, 4.0)

    def test_enhanced_words(self):
        lyr = parse_lrc(ENH)
        ln = lyr.lines[0]
        self.assertEqual(ln.text, "Hello world")
        self.assertEqual([s.text for s in ln.syllables],
                         ["Hel", "lo ", "wor", "ld"])
        self.assertAlmostEqual(ln.syllables[1].start, 10.4)
        self.assertAlmostEqual(ln.syllables[1].end, 10.8)
        self.assertAlmostEqual(ln.syllables[3].end, 11.6)  # closing tag
        self.assertTrue(all(s.is_timed for s in ln.syllables))
        self.assertEqual(lyr.lines[1].syllables, [])

    def test_ms_colon_format_and_crlf(self):
        lyr = parse_lrc("[01:02:50]x y\r\n[01:05.5]z\r\n")
        self.assertAlmostEqual(lyr.lines[0].start, 62.5)
        self.assertEqual(lyr.lines[1].text, "z")


class OffsetTest(unittest.TestCase):
    def setUp(self):
        self.lyr = parse_lrc("[00:10.00]Don't stop\n[00:15.00]Hold on\n"
                             "[00:20.00]Niño bonito\n[00:25.00]Corazón\n")

    def words(self, shift):
        return [("don't", 10 + shift, 10.4 + shift),
                ("stop", 10.5 + shift, 11 + shift),
                ("hold", 15 + shift, 15.4 + shift),
                ("nino", 20 + shift, 20.4 + shift),
                ("corazon", 25 + shift, 25.9 + shift)]

    def test_applies_when_improves(self):
        res = correct_offset(self.lyr, self.words(2.0))
        self.assertTrue(res.applied)
        self.assertAlmostEqual(res.shift, 2.0, places=1)
        self.assertEqual(res.score_before, 0)
        self.assertEqual(res.score_after, 4)
        self.assertAlmostEqual(res.lyrics.lines[0].start, 12.0, places=1)
        self.assertEqual(self.lyr.lines[0].start, 10.0)  # input untouched

    def test_not_applied_when_already_aligned(self):
        res = correct_offset(self.lyr, self.words(0.0))
        self.assertFalse(res.applied)
        self.assertIs(res.lyrics, self.lyr)

    def test_not_applied_without_matches(self):
        res = correct_offset(self.lyr, [("zzz", 3.0, 4.0)])
        self.assertFalse(res.applied)
        self.assertEqual(res.shift, 0.0)

    def test_dict_words_and_shift(self):
        words = [{"word": "Hold", "start": 13.0}, {"word": "Niño", "start": 18.0},
                 {"word": "Corazón,", "start": 23.0}]
        shift, before, after = estimate_offset(self.lyr, words)
        self.assertAlmostEqual(shift, -2.0, places=1)
        self.assertGreater(after, before)
        moved = shift_lyrics(parse_lrc(ENH), 1.0)
        self.assertAlmostEqual(moved.lines[0].syllables[0].start, 11.0)


if __name__ == "__main__":
    unittest.main()
