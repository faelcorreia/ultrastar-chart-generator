"""K2b: window_alignment é adaptador do almighty-listener. listen falso (almighty_fake) em en/pt/es."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from almighty_fake import FakeAlmighty
from src.model.lyrics import Line, Lyrics
from src.readers.lrc import parse_lrc
from src.stages import almighty_runners as ar
from src.stages.window_alignment import align_in_windows, line_windows

TEXTS = {
    "en": ("[00:10.00]Don't stop believin'\n[00:14.00]Hold on to that feelin'\n",
           ["Don't stop believin'", "Hold on to that feelin'"]),
    "pt": ("[00:05.00]Coração partido\n[00:09.00]Não vou chorar\n",
           ["Coração partido", "Não vou chorar"]),
    "es": ("[00:01.00]El niño canta mañana\n[00:04.00]¿Qué pasó, señor?\n",
           ["El niño canta mañana", "¿Qué pasó, señor?"]),
}


class WindowAdapterTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.audio = self.tmp / "A - T - full track.mp3"
        self.work = self.tmp / "A - T" / ".work"

    def tearDown(self):
        self._tmp.cleanup()

    def align(self, lyr, listen, **kw):
        return align_in_windows(lyr, self.audio, work=self.work, listen=listen, **kw)

    def test_languages_window_respected_and_text_kept(self):
        for lang, (lrc, lines) in TEXTS.items():
            with self.subTest(lang):
                lyr = parse_lrc(lrc, language=lang)
                fake = FakeAlmighty(self.tmp)
                res = self.align(lyr, fake, audio_duration=20.0)
                self.assertEqual(res.failed_lines, [])
                self.assertTrue(res.lyrics.has_syllable_timing)
                self.assertEqual(res.lyrics.language, lang)
                for i, line in enumerate(res.lyrics.lines):
                    lo, hi = line_windows(lyr, 20.0)[i]
                    self.assertGreaterEqual(line.syllables[0].start, lo)
                    self.assertLessEqual(line.syllables[-1].end, hi + 1e-9)
                    joined = "".join(s.text for s in line.syllables)
                    self.assertEqual(joined, lines[i])

    def test_listen_gets_full_track_and_lines_with_windows(self):
        lyr = parse_lrc(TEXTS["en"][0], language="en")
        fake = FakeAlmighty(self.tmp)
        self.align(lyr, fake, audio_duration=18.0)
        call = fake.calls[0]
        self.assertEqual(call["path"], self.audio)
        self.assertEqual(call["want"], ["transcript", "alignment"])
        self.assertEqual(call["work_dir"], self.work / ".almighty")
        self.assertEqual(call["cache_dir"], self.tmp / ".almighty-cache")
        opt = call["options"]["transcript"]
        self.assertEqual(opt["language"], "en")
        self.assertTrue(opt["alt_audio"])
        self.assertEqual(opt["lines"], [
            {"text": "Don't stop believin'", "window": [10.0, 14.0]},
            {"text": "Hold on to that feelin'", "window": [14.0, 18.0]},
        ])

    def test_absolute_offset_comes_back_unchanged(self):
        lyr = parse_lrc(TEXTS["en"][0], language="en")
        res = self.align(lyr, FakeAlmighty(self.tmp), audio_duration=18.0)
        second = res.lyrics.lines[1]
        self.assertAlmostEqual(second.start, 14.0)
        self.assertLessEqual(second.end, 18.0 + 1e-9)
        self.assertAlmostEqual(res.lyrics.lines[0].start, 10.0)

    def test_failed_line_keeps_even_timing_in_window(self):
        lyr = parse_lrc(TEXTS["pt"][0], language="pt")
        res = self.align(lyr, FakeAlmighty(self.tmp, fail_on=("Coração",)))
        self.assertEqual(res.failed_lines, [0])
        line = res.lyrics.lines[0]
        self.assertTrue(all(s.is_timed for s in line.syllables))
        self.assertAlmostEqual(line.start, 5.0)
        self.assertLessEqual(line.end, 9.0 + 1e-9)
        self.assertEqual(res.lyrics.lines[1].syllables[0].start, 9.0)

    def test_failed_index_refers_to_lyrics_line_when_unsent_lines_precede(self):
        lyr = Lyrics(lines=[
            Line(text="♪", start=8.0),
            Line(text="Coração partido", start=10.0),
            Line(text="Não vou chorar", start=14.0),
        ], language="pt")
        res = self.align(lyr, FakeAlmighty(self.tmp, fail_on=("Coração",)))
        self.assertEqual(res.failed_lines, [1])
        self.assertEqual(res.lyrics.lines[0].text, "♪")
        self.assertEqual(res.lyrics.lines[0].syllables, [])
        self.assertEqual(res.lyrics.lines[0].start, 8.0)

    def test_alt_audio_flag_is_passed_through(self):
        lyr = parse_lrc(TEXTS["es"][0], language="es")
        off = FakeAlmighty(self.tmp)
        self.align(lyr, off, alt_audio=False)
        self.assertFalse(off.calls[0]["options"]["transcript"]["alt_audio"])
        on = FakeAlmighty(self.tmp)
        self.align(lyr, on)
        self.assertTrue(on.calls[0]["options"]["transcript"]["alt_audio"])

    def test_only_unsent_lines_skip_almighty(self):
        lyr = Lyrics(lines=[Line(text="♪", start=1.0), Line(text="-", start=3.0)], language="en")
        fake = FakeAlmighty(self.tmp)
        res = self.align(lyr, fake)
        self.assertEqual(fake.calls, [])
        self.assertEqual(res.failed_lines, [])
        self.assertEqual([l.start for l in res.lyrics.lines], [1.0, 3.0])

    def test_transcript_without_lines_is_a_contract_error(self):
        lyr = parse_lrc(TEXTS["en"][0], language="en")
        with self.assertRaisesRegex(ar.AlmightyError, "fora do contrato"):
            self.align(lyr, FakeAlmighty(self.tmp, mode="v1"))

    def test_word_count_mismatch_is_a_contract_error(self):
        lyr = parse_lrc(TEXTS["en"][0], language="en")
        with self.assertRaisesRegex(ar.AlmightyError, "fora do contrato"):
            self.align(lyr, FakeAlmighty(self.tmp, mode="short"))

    def test_almighty_failure_names_the_song(self):
        lyr = parse_lrc(TEXTS["en"][0], language="en")
        with self.assertRaisesRegex(ar.AlmightyError, "A - T"):
            self.align(lyr, FakeAlmighty(self.tmp, error=RuntimeError("sem stems")))

    def test_language_and_sync_are_required(self):
        lyr = parse_lrc(TEXTS["en"][0])
        with self.assertRaises(ValueError):
            self.align(lyr, FakeAlmighty(self.tmp))
        with self.assertRaises(ValueError):
            align_in_windows(lyr, self.audio, "fr", work=self.work, listen=FakeAlmighty(self.tmp))
        with self.assertRaises(ValueError):
            self.align(Lyrics.from_text("a b", "en"), FakeAlmighty(self.tmp))

    def test_default_listen_is_used_without_injection(self):
        lyr = parse_lrc(TEXTS["en"][0], language="en")
        fake = FakeAlmighty(self.tmp)
        with mock.patch.object(ar, "_default_listen", return_value=fake):
            align_in_windows(lyr, self.audio, work=self.work)
        self.assertEqual(len(fake.calls), 1)


if __name__ == "__main__":
    unittest.main()
