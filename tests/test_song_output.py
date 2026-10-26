import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from src.model.lyrics import Line, Lyrics, Syllable  # noqa: E402
from src.model.song_job import SongJob  # noqa: E402
from src.planner import default_stages, run_job  # noqa: E402
from src.stages import song_output as so  # noqa: E402
from tests.test_audio_analysis import SONGS, fake_runners  # noqa: E402

TEXT = {"en": "Let it be\nWhisper words of wisdom",
        "pt": "Força estranha\nNão há nada além",
        "es": "Sueño mañana\nCanción de niño"}


def fake_syllables(calls):
    def runner(song, name, work):
        calls.append((name, song["lyrics"], song["language"], work))
        p = work / f"{name} - syllables.json"
        p.write_text("[]")
        return {"syllables": p}
    return runner


def fake_writer(calls):
    def writer(song, name, out_dir, lyrics):
        calls.append((name, dict(song), out_dir, lyrics))
        folder = out_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{name}.txt").write_text("#TITLE\n")
        return folder
    return writer


class SongOutputTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def job(self, lang, plain=True):
        artist, title = SONGS[lang]
        audio = self.tmp / f"{lang}.mp3"
        audio.write_bytes(b"a")
        cover = self.tmp / f"{lang}.jpg"
        cover.write_bytes(b"c")
        job = SongJob(artist=artist, title=title, language=lang,
                      audio_path=audio, year=2000, cover_path=cover)
        if plain:
            job.lyrics = Lyrics.from_text(TEXT[lang], language=lang)
        else:
            job.lyrics = Lyrics(language=lang, lines=[Line(syllables=[
                Syllable(w, i, i + 1)
                for i, w in enumerate(TEXT[lang].split()[:2])])])
        return job

    def test_registry_complete_and_lazy(self):
        stages = default_stages(self.tmp)
        for name in ("syllables", "write"):
            self.assertIn(name, stages)
        code = ("import sys; from src.planner import default_stages; "
                "default_stages('x'); "
                "assert 'torch' not in sys.modules; "
                "assert 'whisper' not in sys.modules")
        root = Path(__file__).resolve().parent.parent
        res = subprocess.run([sys.executable, "-c", code], cwd=root,
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_full_plan_in_three_languages(self):
        for lang, (artist, title) in SONGS.items():
            out = self.tmp / "out" / lang
            sc, wc, ac = [], [], []
            sep, pit, bea = fake_runners(ac)
            stages = default_stages(out, separate=sep, pitches=pit,
                                    beats=bea, syllables=fake_syllables(sc),
                                    writer=fake_writer(wc))
            stages["align_plain"] = lambda j: None  # heavy CTC aligner
            res = run_job(self.job(lang), out, stages)
            self.assertEqual(res.status, "done", (lang, res.reason))
            self.assertEqual(res.ran[-2:], ["syllables", "write"])
            name = f"{artist} - {title}"
            work = out / name / ".work"
            self.assertEqual(sc[0][3], work)
            self.assertEqual(sc[0][1].read_text(encoding="utf-8").strip(),
                             TEXT[lang])
            self.assertEqual(sc[0][2], lang)
            self.assertEqual(wc[0][0], name)
            self.assertEqual(wc[0][2], out)
            self.assertTrue((out / name / f"{name}.txt").is_file())
            self.assertEqual(wc[0][1]["syllables"], work / f"{name} - syllables.json")
            res2 = run_job(self.job(lang), out, stages)
            self.assertEqual(res2.status, "done")
            self.assertIn("syllables", res2.cached)
            self.assertEqual(len(sc), 1)

    def test_timed_lyrics_skip_transcriber_and_go_to_writer(self):
        for lang in SONGS:
            sc, wc = [], []
            stages = so.make_stages(self.tmp, fake_syllables(sc),
                                    fake_writer(wc))
            job = self.job(lang, plain=False)
            job.vocals_path = self.tmp / "v.mp3"
            job.extra.update(instrumental=self.tmp / "i.mp3",
                             pitches=self.tmp / "p.json",
                             beats=self.tmp / "b.json")
            stages["syllables"](job)
            self.assertEqual(sc, [])
            stages["write"](job)
            self.assertIs(wc[0][3], job.lyrics)
            self.assertNotIn("syllables", wc[0][1])
            self.assertEqual(wc[0][1]["language"], lang)

    def test_missing_inputs_fail_clearly(self):
        stages = so.make_stages(self.tmp, fake_syllables([]), fake_writer([]))
        job = self.job("en")
        with self.assertRaises(ValueError) as ctx:
            stages["syllables"](job)
        self.assertIn("vocals", str(ctx.exception))
        job.vocals_path = self.tmp / "v.mp3"
        job.lyrics = None
        with self.assertRaises(ValueError):
            stages["syllables"](job)
        with self.assertRaises(ValueError) as ctx:
            stages["write"](self.job("pt"))
        self.assertIn("required", str(ctx.exception))

    def test_default_writer_uses_given_output_dir(self):
        import src.writer.formatter as fm
        seen = {}

        class Fmt(fm.UltraStarFormatter):
            def format_ultrastar_song_folder(self, name, lyrics=None):
                seen["dir"] = self.output_dir
                return self.output_dir / name
        orig = fm.UltraStarFormatter
        import src.core.ultrastar_formatter as shim
        shim.UltraStarFormatter = Fmt
        try:
            res = so.default_writer({}, "A - B", self.tmp, None)
        finally:
            shim.UltraStarFormatter = orig
        self.assertEqual(seen["dir"], self.tmp)
        self.assertEqual(res, self.tmp / "A - B")


if __name__ == "__main__":
    unittest.main()
