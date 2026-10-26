import unittest
from pathlib import Path

from src.model import Line, Lyrics, SongJob, Syllable

SAMPLES = {
    "en": "Don't stop believin'\nI'm gonna be alright",
    "pt": "Não vou ficar\nCoração, ação e canção",
    "es": "El niño y la señal\n¿Qué será, mañana?",
}


class SyllableTest(unittest.TestCase):
    def test_timing_in_seconds(self):
        syllable = Syllable("be", start=1.5, end=2.0)

        self.assertTrue(syllable.is_timed)
        self.assertAlmostEqual(syllable.duration, 0.5)

    def test_untimed(self):
        syllable = Syllable("be")

        self.assertFalse(syllable.is_timed)
        self.assertIsNone(syllable.duration)


class LineTest(unittest.TestCase):
    def test_derives_text_and_times_from_syllables(self):
        line = Line(syllables=[
            Syllable("ca", 1.0, 1.4), Syllable("ña", 1.4, 2.0),
        ])

        self.assertEqual(line.text, "caña")
        self.assertEqual(line.start, 1.0)
        self.assertEqual(line.end, 2.0)

    def test_explicit_values_are_kept(self):
        line = Line(text="Hi", start=3.0, end=4.0,
                    syllables=[Syllable("Hi", 3.1, 3.9)])

        self.assertEqual((line.start, line.end), (3.0, 4.0))


class LyricsTest(unittest.TestCase):
    def test_language_not_defaulted(self):
        self.assertIsNone(Lyrics().language)
        self.assertIsNone(Lyrics.from_text("hello").language)

    def test_from_text_per_language(self):
        for language, text in SAMPLES.items():
            with self.subTest(language=language):
                lyrics = Lyrics.from_text(text, language=language)

                self.assertEqual(lyrics.language, language)
                self.assertEqual(len(lyrics.lines), 2)
                self.assertEqual(lyrics.text, text)
                self.assertFalse(lyrics.is_synced)

    def test_from_text_skips_blank_lines(self):
        lyrics = Lyrics.from_text("a\n\n  \nb\n")

        self.assertEqual([line.text for line in lyrics.lines], ["a", "b"])

    def test_english_apostrophes_preserved(self):
        lyrics = Lyrics.from_text(SAMPLES["en"], language="en")

        self.assertEqual(lyrics.lines[0].text, "Don't stop believin'")
        self.assertEqual(lyrics.lines[1].text, "I'm gonna be alright")

    def test_synced_and_syllable_timing(self):
        lyrics = Lyrics(lines=[
            Line(syllables=[Syllable("Co", 0.0, 0.5),
                            Syllable("ra", 0.5, 1.0),
                            Syllable("ção", 1.0, 1.8)]),
        ], language="pt")

        self.assertTrue(lyrics.is_synced)
        self.assertTrue(lyrics.has_syllable_timing)
        self.assertEqual(lyrics.text, "Coração")

    def test_line_timed_only(self):
        lyrics = Lyrics(lines=[Line("¿Qué será?", 1.0, 3.0)], language="es")

        self.assertTrue(lyrics.is_synced)
        self.assertFalse(lyrics.has_syllable_timing)

    def test_empty_is_not_synced(self):
        self.assertFalse(Lyrics().is_synced)
        self.assertFalse(Lyrics().has_syllable_timing)

    def test_lines_not_shared_between_instances(self):
        first, second = Lyrics(), Lyrics()
        first.lines.append(Line("x"))

        self.assertEqual(second.lines, [])


class SongJobTest(unittest.TestCase):
    def test_everything_unknown_by_default(self):
        job = SongJob()

        self.assertIsNone(job.language)
        self.assertIsNone(job.resolved_language)
        self.assertEqual(
            job.missing(),
            ["artist", "title", "language", "audio", "lyrics", "cover",
             "year"],
        )

    def test_missing_shrinks_as_job_accumulates(self):
        job = SongJob(artist="Los Lobos", title="La Bamba", year=1987,
                      audio_path="in/la_bamba.mp3")
        job.lyrics = Lyrics.from_text("Para bailar", language="es")

        self.assertIsInstance(job.audio_path, Path)
        self.assertEqual(job.missing(), ["cover"])
        self.assertEqual(job.resolved_language, "es")

    def test_job_language_wins_over_lyrics_language(self):
        job = SongJob(language="pt",
                      lyrics=Lyrics.from_text("hi", language="en"))

        self.assertEqual(job.resolved_language, "pt")

    def test_require_language_fails_clearly(self):
        job = SongJob(artist="Beyoncé", title="Halo")

        with self.assertRaises(ValueError) as ctx:
            job.require_language()
        self.assertIn("Beyoncé - Halo", str(ctx.exception))

    def test_require_language_per_language(self):
        for language in SAMPLES:
            with self.subTest(language=language):
                job = SongJob(language=language)

                self.assertEqual(job.require_language(), language)

    def test_accents_in_metadata(self):
        job = SongJob(artist="Caetano Veloso", title="Coração Vagabundo")
        job.sources["title"] = "filename"
        job.note("title from file name")

        self.assertEqual(job.title, "Coração Vagabundo")
        self.assertEqual(job.sources, {"title": "filename"})
        self.assertEqual(job.notes, ["title from file name"])

    def test_mutable_defaults_not_shared(self):
        first, second = SongJob(), SongJob()
        first.note("x")
        first.sources["a"] = "b"

        self.assertEqual(second.notes, [])
        self.assertEqual(second.sources, {})


if __name__ == "__main__":
    unittest.main()
