import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from src.batch.cli import main  # noqa: E402
from src.batch.report import REPORT_NAME, BatchReport  # noqa: E402
from src.batch.songlist import song_key  # noqa: E402
from tests.test_batch_cli import LIST, fake_stages  # noqa: E402


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.audio = self.tmp / "audio"
        self.audio.mkdir()
        self.out = self.tmp / "out"
        self.lst = self.tmp / "list.csv"
        self.lst.write_text(LIST, encoding="utf-8")
        self.lib = self.tmp / "lib"
        self.lib.mkdir()

    def run_main(self, stages, *extra):
        return main([str(self.lst), "--audio-dir", str(self.audio),
                     "--output-dir", str(self.out), "--library",
                     str(self.lib), *extra], stages=stages)

    def data(self):
        return json.loads((self.out / REPORT_NAME).read_text("utf-8"))["songs"]

    def test_report_written_with_statuses(self):
        (self.audio / "Niño Sueño - Año Nuevo.mp3").touch()
        (self.lib / "Beyoncé - Halo").mkdir()
        self.run_main(fake_stages([]))
        songs = self.data()
        self.assertEqual(len(songs), 5)
        e = songs[song_key("Niño Sueño", "Año Nuevo")]
        self.assertEqual((e["status"], e["language"], e["artist"]),
                         ("done", "pt", "Niño Sueño"))
        self.assertTrue(e["started_at"] and e["finished_at"])
        self.assertEqual(songs[song_key("Beyoncé", "Halo")]["status"], "skipped")
        self.assertEqual(songs[song_key("Peso Pluma", "Ella Baila Sola")]
                         ["status"], "awaiting_audio")
        self.assertEqual(list(self.out.glob(".report-*")), [])

    def test_resume_skips_done_retries_awaiting(self):
        (self.audio / "Niño Sueño - Año Nuevo.mp3").touch()
        self.run_main(fake_stages([]))
        (self.audio / "Mägo de Oz - La Danza del Fuego.mp3").touch()
        calls = []
        res = self.run_main(fake_stages(calls))
        self.assertNotIn("Niño Sueño", [a for _, a in calls])
        self.assertTrue(res[4].reason.startswith("resumed"))
        self.assertEqual(res[4].status, "done")
        self.assertEqual(self.data()[song_key("Mägo de Oz", "La Danza del Fuego")]
                         ["status"], "done")
        self.assertIn("Mägo de Oz", [a for _, a in calls])
        calls2 = []
        self.run_main(fake_stages(calls2), "--retry-all")
        self.run_main(fake_stages(calls2), "--retry-all", "--force")
        self.assertIn("Niño Sueño", [a for _, a in calls2])

    def test_failed_retried_and_incremental_save(self):
        (self.audio / "Niño Sueño - Año Nuevo.mp3").touch()
        boom = fake_stages([])
        boom["write"] = lambda job: (_ for _ in ()).throw(RuntimeError("boom"))
        self.run_main(boom)
        e = self.data()[song_key("Niño Sueño", "Año Nuevo")]
        self.assertEqual(e["status"], "failed")
        self.assertIn("boom", e["reason"])
        self.run_main(fake_stages([]))
        self.assertEqual(self.data()[song_key("Niño Sueño", "Año Nuevo")]
                         ["status"], "done")

    def test_crash_keeps_earlier_songs(self):
        stages = fake_stages([])
        n = []

        def crash(job):
            n.append(1)
            if job.artist == "Peso Pluma":
                raise KeyboardInterrupt
        stages["lyrics"] = crash
        for f in ("Beyoncé - Halo", "João Gilberto - Chega de Saudade",
                  "Peso Pluma - Ella Baila Sola"):
            (self.audio / f"{f}.mp3").touch()
        with self.assertRaises(KeyboardInterrupt):
            self.run_main(stages)
        self.assertEqual(len(self.data()), 2)

    def test_corrupt_report_ignored_and_summary(self):
        self.out.mkdir()
        (self.out / REPORT_NAME).write_text("{trunc", encoding="utf-8")
        r = BatchReport(self.out / REPORT_NAME)
        r.record("k", "failed", "sem letra: señal", "Año", "Canción", "es")
        r.record("j", "review", "revisar", "João", "Saudade", "pt")
        s = r.summary()
        self.assertIn("failed=1", s)
        self.assertIn("review=1", s)
        self.assertIn("Año - Canción: sem letra: señal", s)


if __name__ == "__main__":
    unittest.main()
