import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from src.model.lyrics import Line, Lyrics  # noqa: E402
from src.model.song_job import SongJob  # noqa: E402
from src.planner import REVIEW, default_stages, plan, run_job  # noqa: E402
from tests.test_audio_analysis import SONGS, fake_runners  # noqa: E402
from tests.test_song_output import fake_syllables, fake_writer  # noqa: E402

LINES = {
    "en": ["let it be tonight", "whisper words of wisdom",
           "there will be an answer", "speaking words of wisdom"],
    "pt": ["força estranha no meu ser", "vi o menino correndo",
           "nada mais me faz sofrer", "canção de um novo dia"],
    "es": ["sueño con el mañana", "canción de niño aquí",
           "ojalá que llueva café", "despierta con el sol"],
}
SOURCE_OFFSET = 4.0     # the downloaded LRC is 4 s late


def true_starts():
    return [10.0 * i for i in range(4)]


def whisper_words(lang):
    out = []
    for text, start in zip(LINES[lang], true_starts()):
        for k, w in enumerate(text.split()):
            out.append((w, start + 0.5 * k, start + 0.5 * k + 0.4))
    return out


def fake_transcriber(lang, calls):
    def transcribe(path, language):
        calls.append(Path(path).name)
        return whisper_words(lang), lang
    return transcribe


def fake_aligner(lang, wrong=False):
    """CTC fake: places every lyric word at its true time."""
    def align(audio, words, language, window):
        starts = {}
        flat = [w for t in LINES[lang] for w in t.split()]
        times = [s + 0.5 * k for t, s in zip(LINES[lang], true_starts())
                 for k in range(len(t.split()))]
        if window is not None:
            lo, hi = window
            return [(lo + 0.1 * i, lo + 0.1 * i + 0.3)
                    for i in range(len(words))]
        n = len(words)
        if wrong:
            times = [t + 30 for t in times]
        if n != len(flat):
            times = [times[0] + 0.5 * i for i in range(n)]
        return [(t, t + 0.4) for t in times[:n]]
    return align


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def job(self, lang, lines, ext="mp3", synced=True):
        artist, title = SONGS[lang]
        audio = self.tmp / f"{lang}.{ext}"
        audio.write_bytes(b"a")
        cover = self.tmp / f"{lang}.jpg"
        cover.write_bytes(b"c")
        starts = [t + SOURCE_OFFSET for t in true_starts()]
        job = SongJob(artist=artist, title=title, language=lang,
                      audio_path=audio, year=2000, cover_path=cover)
        if synced:
            job.lyrics = Lyrics(language=lang, source="lrclib", lines=[
                Line(text=t, start=s) for t, s in zip(lines, starts)])
        else:
            job.lyrics = Lyrics.from_text("\n".join(lines), language=lang)
        return job

    def stages(self, lang, wrong=False, calls=None):
        out = self.tmp / "out" / lang
        sep, pit, bea = fake_runners([])
        stages = default_stages(
            out, separate=sep, pitches=pit, beats=bea,
            syllables=fake_syllables([]), writer=fake_writer([]),
            transcriber=fake_transcriber(lang, calls if calls is not None
                                         else []),
            aligner=fake_aligner(lang, wrong))
        stages["align_window"] = lambda j: None
        stages["align_plain"] = lambda j: None
        stages["normalize"] = lambda j: setattr(
            j, "audio_path", Path(j.audio_path).with_suffix(".mp3"))
        return out, stages

    def test_plan_order_and_applicability(self):
        for lang in LINES:
            job = self.job(lang, LINES[lang])
            self.assertEqual(plan(job), ["vocals", "reconcile",
                                         "sync_validate", "align_window",
                                         "pitches", "beats", "syllables",
                                         "write"])
            job = self.job(lang, LINES[lang], ext="flac", synced=False)
            self.assertEqual(plan(job)[:3], ["normalize", "vocals",
                                             "reconcile"])
            self.assertNotIn("sync_validate", plan(job))
            job.sources["lyrics"] = "whisper"
            self.assertNotIn("reconcile", plan(job))

    def test_offset_lrc_is_corrected_and_reported(self):
        for lang in LINES:
            calls = []
            out, stages = self.stages(lang, calls=calls)
            res = run_job(self.job(lang, LINES[lang]), out, stages)
            self.assertEqual(res.status, "done", (lang, res.reason))
            self.assertEqual(res.ran[:3], ["vocals", "reconcile",
                                           "sync_validate"])
            rep = res.job.extra["sync_validate"]
            self.assertEqual(rep["correction"], "offset")
            self.assertEqual(rep["ok_after_pct"], 100.0)
            self.assertEqual([l.start for l in res.job.lyrics.lines],
                             true_starts())
            self.assertEqual(res.job.extra["reconcile"]["status"], "ok")
            self.assertEqual(len(calls), 1)   # one Whisper run, two stages

    def test_estimators_disagree_goes_to_review(self):
        for lang in LINES:
            out, stages = self.stages(lang, wrong=True)
            res = run_job(self.job(lang, LINES[lang]), out, stages)
            self.assertEqual(res.status, REVIEW, (lang, res.reason))
            self.assertIn("uncertain", res.reason)
            rep = res.job.extra["sync_validate"]
            self.assertEqual(rep["uncertain_pct"], 100.0)
            self.assertIn("whisper", rep["uncertain_lines"][0])

    def test_low_match_goes_to_review(self):
        for lang in LINES:
            out, stages = self.stages(lang)
            other = ["zzz qqq www eee"] * 4
            res = run_job(self.job(lang, other, synced=False), out, stages)
            self.assertEqual(res.status, REVIEW, (lang, res.reason))
            self.assertIn("match Whisper", res.reason)

    def test_missing_chorus_confirmed_by_ctc_and_placed(self):
        lang = "en"
        chorus = ["la la la la", "na na na na"]
        words = []
        t = 0.0
        for text in chorus + ["verse one goes here now"] + chorus:
            for w in text.split():
                words.append((w, t, t + 0.4))
                t += 0.5
            t += 3
        lyr = Lyrics(language=lang, lines=[
            Line(text=chorus[0], start=0.0), Line(text=chorus[1], start=3.0),
            Line(text="verse one goes here now", start=9.0)])
        job = SongJob(artist="A", title="T", language=lang, lyrics=lyr,
                      audio_path=self.tmp / "a.mp3")
        job.audio_path.write_bytes(b"a")
        from src.stages.verify_lyrics import make_stages

        def aligner(audio, w, language, window):
            lo, _ = window
            return [(lo + 0.5 * i, lo + 0.5 * i + 0.4) for i in range(len(w))]
        st = make_stages(lambda p, l: (words, lang), aligner)
        st["reconcile"](job)
        texts = [l.text for l in job.lyrics.lines]
        self.assertEqual(texts[-2:], chorus)
        self.assertTrue(all(l.start is not None for l in job.lyrics.lines))
        issue = job.extra["reconcile"]["issues"][0]
        self.assertEqual(issue["kind"], "missing_chorus")
        self.assertTrue(issue["confirmed"])
        self.assertEqual(job.extra["reconcile"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
