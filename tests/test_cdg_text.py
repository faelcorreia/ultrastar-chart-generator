import unittest

from src.model.lyrics import Line, Lyrics, Syllable
from src.readers.cdg_text import SOURCE_FOREIGN, as_timing_guess, clean_ocr_text


class CdgTextTest(unittest.TestCase):
    def test_pipe_becomes_i(self):
        self.assertEqual(clean_ocr_text("AM  |  THE ONLY ONE"),
                         "AM I THE ONLY ONE")

    def test_timing_guess_drops_syllables_and_noise_lines(self):
        lyr = Lyrics(lines=[
            Line(text="HELLO  WORLD", start=2.0, end=3.0,
                 syllables=[Syllable("HELLO", 2.0, 2.5),
                            Syllable("WORLD", 2.5, 3.0)]),
            Line(text="1", start=4.0),
            Line(text="BYE", start=5.0),
        ], language="en", source="cdg")
        out = as_timing_guess(lyr)
        self.assertEqual([l.text for l in out.lines], ["HELLO WORLD", "BYE"])
        self.assertEqual([l.start for l in out.lines], [2.0, 5.0])
        self.assertTrue(all(not l.syllables for l in out.lines))
        self.assertFalse(out.has_syllable_timing)
        self.assertTrue(out.is_synced)
        self.assertEqual(out.source, SOURCE_FOREIGN)


if __name__ == "__main__":
    unittest.main()
