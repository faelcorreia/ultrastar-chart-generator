import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from src.model.lyrics import Line, Lyrics, Syllable  # noqa: E402
from src.model.song_job import SongJob  # noqa: E402
from src.planner import (  # noqa: E402
    AWAITING_AUDIO, DONE, FAILED, REVIEW, NeedsReview, plan, run_job)

SONGS = {
    "en": ("The Beatles", "Let It Be", "When I find myself"),
    "pt": ("Caetano Veloso", "Força Estranha", "Eu vi o menino correndo"),
    "es": ("Shakira", "Sueño Mañana", "Ojalá que llueva café"),
}


class Fakes:
    """Fake stages that record calls."""

    def __init__(self, tmp, lyrics=None, review=False):
        self.tmp, self.calls, self.lyrics = tmp, [], lyrics
        self.review = review

    def stages(self):
        def mk(name, fn=None):
            def stage(job):
                self.calls.append(name)
                if fn:
                    fn(job)
            return stage

        def lyrics(job):
            job.lyrics = self.lyrics

        def language(job):
            job.language = job.resolved_language or "en"

        def vocals(job):
            p = self.tmp / "vocals.wav"
            p.write_bytes(b"v")
            job.vocals_path = p

        def align(job):
            job.lyrics = Lyrics(lines=[Line(syllables=[
                Syllable("a", 0.0, 1.0)])],
                language=job.resolved_language)
            if self.review:
                job.extra["review"] = "low confidence"

        def metadata(job):
            job.year = 2000
            job.cover_path = None

        return {
            "normalize": mk("normalize", lambda j: setattr(
                j, "audio_path", self._mp3(j))),
            "metadata": mk("metadata", metadata),
            "lyrics": mk("lyrics", lyrics),
            "language": mk("language", language),
            "vocals": mk("vocals", vocals),
            "reconcile": mk("reconcile"),
            "sync_validate": mk("sync_validate"),
            "align_window": mk("align_window", align),
            "align_plain": mk("align_plain", align),
            "pitches": mk("pitches"), "beats": mk("beats"),
            "syllables": mk("syllables"), "write": mk("write"),
        }

    def _mp3(self, job):
        p = Path(job.audio_path).with_suffix(".mp3")
        p.write_bytes(b"mp3")
        return p


class PlannerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def audio(self, name="a.mp3"):
        p = self.tmp / name
        p.write_bytes(b"audio")
        return p

    def test_plan_lists_only_missing_stages(self):
        for lang, (artist, title, line) in SONGS.items():
            job = SongJob(artist=artist, title=title, language=lang,
                          audio_path=self.audio(f"{lang}.flac"),
                          lyrics=Lyrics.from_text(line, language=lang))
            self.assertEqual(plan(job), [
                "normalize", "metadata", "vocals", "reconcile", "align_plain", "pitches",
                "beats", "syllables", "write"])
            job.lyrics = Lyrics(lines=[Line(text=line, start=1.0)])
            self.assertIn("align_window", plan(job))
            job.audio_path = self.audio(f"{lang}.mp3")
            self.assertNotIn("normalize", plan(job))

    def test_plan_without_audio_has_only_audio_free_stages(self):
        job = SongJob(artist="A", title="T")
        self.assertEqual(plan(job), ["metadata", "lyrics"])

    def test_full_run_done_then_cache_skips(self):
        for lang, (artist, title, line) in SONGS.items():
            out = self.tmp / lang
            lyr = Lyrics(lines=[Line(text=line, start=1.0)])

            def job():
                return SongJob(artist=artist, title=title,
                               audio_path=self.audio(f"{lang}.flac"))
            fakes = Fakes(self.tmp, lyrics=lyr)
            res = run_job(job(), out, fakes.stages())
            self.assertEqual((res.status, res.reason), (DONE, "ok"), lang)
            self.assertEqual(fakes.calls, [
                "normalize", "metadata", "lyrics", "language", "vocals",
                "reconcile", "sync_validate", "align_window", "pitches", "beats", "syllables", "write"])
            self.assertEqual(res.job.language, "en")
            fakes2 = Fakes(self.tmp, lyrics=lyr)
            res2 = run_job(job(), out, fakes2.stages())
            self.assertEqual(res2.status, DONE)
            self.assertEqual(fakes2.calls, [], lang)
            self.assertEqual(len(res2.cached), 12)

    def test_plain_lyrics_use_plain_alignment(self):
        fakes = Fakes(self.tmp, lyrics=Lyrics.from_text("hola mundo"))
        job = SongJob(artist="S", title="X", language="es",
                      audio_path=self.audio())
        res = run_job(job, self.tmp / "o", fakes.stages())
        self.assertEqual(res.status, DONE)
        self.assertIn("align_plain", fakes.calls)
        self.assertNotIn("align_window", fakes.calls)

    def test_awaiting_audio_then_resume(self):
        for lang, (artist, title, line) in SONGS.items():
            out = self.tmp / lang
            lyr = Lyrics.from_text(line, language=lang, source="lrclib:1")
            fakes = Fakes(self.tmp, lyrics=lyr)
            res = run_job(SongJob(artist=artist, title=title), out,
                          fakes.stages())
            self.assertEqual(res.status, AWAITING_AUDIO, lang)
            self.assertTrue(res.awaiting_audio)
            self.assertEqual(fakes.calls, ["metadata", "lyrics"])
            # audio appears: lyrics are adopted, no new lookup
            fakes2 = Fakes(self.tmp, lyrics=None)
            job = SongJob(artist=artist, title=title,
                          audio_path=self.audio(f"{lang}.mp3"))
            res2 = run_job(job, out, fakes2.stages())
            self.assertEqual(res2.status, DONE, res2.reason)
            self.assertNotIn("lyrics", fakes2.calls)
            self.assertEqual(res2.job.lyrics.language, lang)

    def test_no_audio_and_no_lyrics_fails(self):
        fakes = Fakes(self.tmp, lyrics=None)
        res = run_job(SongJob(artist="A", title="T"), self.tmp / "o",
                      fakes.stages())
        self.assertEqual(res.status, FAILED)
        self.assertIn("lyrics not found", res.reason)

    def test_missing_artist_title_fails_clearly(self):
        res = run_job(SongJob(), self.tmp / "o", Fakes(self.tmp).stages(),
                      song_name="x")
        self.assertEqual(res.status, FAILED)
        self.assertIn("artist and title", res.reason)

    def test_stage_error_and_review(self):
        stages = Fakes(self.tmp, lyrics=Lyrics.from_text("hi")).stages()

        def boom(job):
            raise ValueError("Language not resolved")
        stages["language"] = boom
        job = SongJob(artist="A", title="T", audio_path=self.audio())
        res = run_job(job, self.tmp / "o1", stages)
        self.assertEqual(res.status, FAILED)
        self.assertEqual(res.reason, "language: Language not resolved")

        fakes = Fakes(self.tmp, lyrics=Lyrics.from_text("hi"), review=True)
        job = SongJob(artist="A", title="T", language="en",
                      audio_path=self.audio())
        res = run_job(job, self.tmp / "o2", fakes.stages())
        self.assertEqual((res.status, res.reason), (REVIEW, "low confidence"))

        stages = fakes.stages()

        def rev(job):
            raise NeedsReview("chorus missing")
        stages["vocals"] = rev
        job = SongJob(artist="A", title="T", language="en",
                      audio_path=self.audio())
        res = run_job(job, self.tmp / "o3", stages)
        self.assertEqual((res.status, res.reason), (REVIEW, "chorus missing"))

    def test_missing_implementation_fails(self):
        job = SongJob(artist="A", title="T", language="pt",
                      lyrics=Lyrics.from_text("oi"), year=1, cover_path=None,
                      audio_path=self.audio())
        res = run_job(job, self.tmp / "o", {})
        self.assertEqual(res.status, FAILED)
        self.assertIn("no implementation", res.reason)


if __name__ == "__main__":
    unittest.main()
