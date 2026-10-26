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
from src.planner import default_stages  # noqa: E402
from src.stages import audio_analysis as aa  # noqa: E402

SONGS = {
    "en": ("The Beatles", "Let It Be"),
    "pt": ("Caetano Veloso", "Força Estranha"),
    "es": ("Shakira", "Sueño Mañana"),
}


def fake_runners(calls):
    def separate(song, name, work):
        calls.append(("vocals", name, work))
        v, i = work / f"{name} - vocals.mp3", work / f"{name} - inst.mp3"
        v.write_bytes(b"v")
        i.write_bytes(b"i")
        return {"vocals": v, "instrumental": i}

    def pitches(song, name, work):
        calls.append(("pitches", name, song["vocals"]))
        p = work / f"{name} - pitches.json"
        p.write_text("{}")
        return {"pitches": p}

    def beats(song, name, work):
        calls.append(("beats", name, song["instrumental"]))
        p = work / f"{name} - beats.json"
        p.write_text("{}")
        return {"beats": p}
    return separate, pitches, beats


class AudioAnalysisTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def job(self, lang, artist, title):
        audio = self.tmp / f"{lang}.mp3"
        audio.write_bytes(b"a")
        return SongJob(artist=artist, title=title, language=lang,
                       audio_path=audio)

    def test_registry_has_stages_without_heavy_imports(self):
        stages = default_stages(self.tmp)
        for name in ("vocals", "pitches", "beats"):
            self.assertIn(name, stages)
        # other tests may have imported them, so check in a clean process
        code = ("import sys; from src.planner import default_stages; "
                "default_stages('x'); "
                "assert 'demucs' not in sys.modules; "
                "assert 'librosa' not in sys.modules")
        root = Path(__file__).resolve().parent.parent
        res = subprocess.run([sys.executable, "-c", code], cwd=root,
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_stages_fill_job_in_three_languages(self):
        for lang, (artist, title) in SONGS.items():
            calls = []
            stages = aa.make_stages(self.tmp / lang, *fake_runners(calls))
            job = self.job(lang, artist, title)
            for name in ("vocals", "pitches", "beats"):
                stages[name](job)
            work = self.tmp / lang / f"{artist} - {title}" / ".work"
            self.assertEqual(job.vocals_path.parent, work, lang)
            self.assertTrue(job.extra["pitches"].is_file(), lang)
            self.assertTrue(job.extra["beats"].is_file(), lang)
            self.assertEqual([c[1] for c in calls],
                             [f"{artist} - {title}"] * 3)
            self.assertEqual(calls[2][2], job.extra["instrumental"])

    def test_beats_fall_back_to_full_track(self):
        calls = []
        stages = aa.make_stages(self.tmp, *fake_runners(calls))
        job = self.job("es", *SONGS["es"])
        job.vocals_path = self.tmp / "v.mp3"
        stages["beats"](job)
        self.assertEqual(calls[0][2], job.audio_path)

    def test_pitches_without_vocals_fails_clearly(self):
        stages = aa.make_stages(self.tmp, *fake_runners([]))
        job = self.job("pt", *SONGS["pt"])
        with self.assertRaises(ValueError) as ctx:
            stages["pitches"](job)
        self.assertIn("vocals", str(ctx.exception))

    def test_no_audio_fails(self):
        stages = aa.make_stages(self.tmp, *fake_runners([]))
        with self.assertRaises(ValueError):
            stages["vocals"](SongJob(artist="A", title="T"))

    def test_legacy_adapter_points_output_to_work_dir(self):
        class Fake:
            def __init__(self, songs):
                self.songs = songs

            def run(self, name):
                p = self.output_dir / f"{name}.json"
                p.write_text("{}")
                self.songs[name]["beats"] = p

        for lang, (artist, title) in SONGS.items():
            name = f"{artist} - {title}"
            out = aa._run_legacy(Fake, "run", ("beats", "nope"), {}, name,
                                 self.tmp)
            self.assertEqual(out, {"beats": self.tmp / f"{name}.json"})

    def test_planner_runs_real_wiring_with_injected_runners(self):
        from src.planner import run_job
        for lang, (artist, title) in SONGS.items():
            calls = []
            sep, pit, bea = fake_runners(calls)
            stages = default_stages(self.tmp / lang, separate=sep,
                                    pitches=pit, beats=bea)
            def build():
                job = self.job(lang, artist, title)
                job.lyrics = Lyrics(lines=[Line(syllables=[
                    Syllable("a", 0.0, 1.0)])], language=lang)
                job.year, job.cover_path = 2000, self.tmp / "c.jpg"
                return job
            stages["syllables"] = stages["write"] = lambda j: None
            stages["reconcile"] = lambda j: None
            res = run_job(build(), self.tmp / lang, stages)
            self.assertEqual(res.status, "done", (lang, res.reason))
            self.assertEqual([c[0] for c in calls],
                             ["vocals", "pitches", "beats"])
            res2 = run_job(build(), self.tmp / lang, stages)
            self.assertEqual(res2.status, "done")
            self.assertIn("vocals", res2.cached, lang)
            self.assertEqual(len(calls), 3, lang)


if __name__ == "__main__":
    unittest.main()
