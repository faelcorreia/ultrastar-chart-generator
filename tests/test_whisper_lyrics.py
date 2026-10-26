import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from src.batch.report import BatchReport  # noqa: E402
from src.core.language import LanguageError  # noqa: E402
from src.model.song_job import SongJob  # noqa: E402
from src.planner import REVIEW, default_stages, run_job  # noqa: E402
from src.stages.whisper_lyrics import lyrics_from_whisper  # noqa: E402
from tests.test_audio_analysis import SONGS, fake_runners  # noqa: E402
from tests.test_song_output import fake_syllables, fake_writer  # noqa: E402

# (word, start, end): two phrases separated by a long pause, then a stanza gap
WORDS = {
    "en": ["Let it be", "whisper words of wisdom", "let it be"],
    "pt": ["Força estranha", "vi o menino correndo", "força estranha"],
    "es": ["Sueño mañana", "canción de niño aquí", "sueño mañana"],
}


def timed(lang):
    out, t = [], 0.0
    for phrase in WORDS[lang]:
        for w in phrase.split():
            out.append((w, t, t + 0.4))
            t += 0.5
        t += 2.5  # pause: line and stanza break
    return out


class FakeLrclib:
    def __init__(self):
        self.calls = 0

    def find(self, *args):
        self.calls += 1
        return None


class WhisperFallbackTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def job(self, lang, language=None):
        artist, title = SONGS[lang]
        audio = self.tmp / f"{lang}.mp3"
        audio.write_bytes(b"a")
        cover = self.tmp / f"{lang}.jpg"
        cover.write_bytes(b"c")
        return SongJob(artist=artist, title=title, language=language,
                       audio_path=audio, year=2000, cover_path=cover)

    def test_lrclib_miss_goes_through_whisper_to_review(self):
        for lang in SONGS:
            out = self.tmp / "out" / lang
            seen, wc = [], []

            def transcriber(path, language, lang=lang):
                seen.append(language)
                return timed(lang), lang

            lrclib = FakeLrclib()
            sep, pit, bea = fake_runners([])
            stages = default_stages(
                out, lrclib=lrclib, separate=sep, pitches=pit, beats=bea,
                syllables=fake_syllables([]), writer=fake_writer(wc),
                transcriber=transcriber)
            stages["align_window"] = lambda j: None  # heavy CTC aligner
            stages["reconcile"] = lambda j: None
            report = BatchReport(out / "report.json")
            res = run_job(self.job(lang, language=lang), out, stages)
            self.assertEqual(res.status, REVIEW, (lang, res.reason))
            self.assertIn("Whisper", res.reason)
            self.assertIn("LRCLIB", res.reason)
            self.assertIn(f"language {lang}", res.reason)
            self.assertEqual(lrclib.calls, 1)
            self.assertEqual(seen, [lang])
            self.assertEqual(res.ran[-1], "write")
            self.assertEqual(len(wc), 1)  # the song was written
            lyr = res.job.lyrics
            self.assertEqual(lyr.source, "whisper")
            self.assertEqual(lyr.language, lang)
            self.assertEqual([l.text for l in lyr.lines], WORDS[lang])
            self.assertTrue(lyr.is_synced)
            self.assertEqual([l.stanza_break_before for l in lyr.lines],
                             [False, True, True])
            self.assertEqual(res.job.sources["lyrics"], "whisper")
            report.record("k", res.status, res.reason, res.job.artist,
                          res.job.title, res.job.language)
            self.assertEqual(
                BatchReport(out / "report.json").get("k")["status"], REVIEW)

    def test_language_detected_by_whisper_is_noted(self):
        for lang in SONGS:
            job = self.job(lang)
            lyr = lyrics_from_whisper(
                job, lambda p, l, lang=lang: (timed(lang), lang),
                tag_reader=lambda p: None)
            self.assertEqual(lyr.language, lang)
            self.assertIn("DETECTED by Whisper", job.notes[0])
            self.assertIn("DETECTED", job.extra["review"])

    def test_language_from_tags_is_not_detected(self):
        job = self.job("pt")
        seen = []

        def transcriber(path, language):
            seen.append(language)
            return timed("pt"), "en"
        lyr = lyrics_from_whisper(job, transcriber,
                                  tag_reader=lambda p: "pt")
        self.assertEqual((lyr.language, seen), ("pt", ["pt"]))
        self.assertIn("file tags", job.extra["review"])

    def test_no_silent_default_language(self):
        job = self.job("en")
        with self.assertRaises(LanguageError):
            lyrics_from_whisper(job, lambda p, l: (timed("en"), None),
                                tag_reader=lambda p: None)
        with self.assertRaises(LanguageError):
            lyrics_from_whisper(job, lambda p, l: (timed("en"), "fr"),
                                tag_reader=lambda p: None)
        with self.assertRaises(LanguageError):
            lyrics_from_whisper(self.job("en", language="fr"),
                                lambda p, l: (timed("en"), "en"))

    def test_nothing_transcribed_fails(self):
        stages = default_stages(self.tmp / "o", lrclib=FakeLrclib(),
                                transcriber=lambda p, l: ([], l))
        res = run_job(self.job("es", language="es"), self.tmp / "o", stages)
        self.assertEqual(res.status, "failed")
        self.assertIn("no words", res.reason)

    def test_lrclib_hit_does_not_use_whisper(self):
        from src.model.lyrics import Lyrics

        class Hit:
            def to_lyrics(self, language):
                return Lyrics.from_text("hola", language=language,
                                        source="lrclib:1")

        class Lrclib:
            def find(self, *a):
                return Hit()

        def boom(p, l):
            raise AssertionError("Whisper must not run")
        job = self.job("es", language="es")
        stages = default_stages(self.tmp / "o", lrclib=Lrclib(),
                                transcriber=boom)
        stages["lyrics"](job)
        self.assertEqual(job.lyrics.source, "lrclib:1")
        self.assertNotIn("review", job.extra)


if __name__ == "__main__":
    unittest.main()
