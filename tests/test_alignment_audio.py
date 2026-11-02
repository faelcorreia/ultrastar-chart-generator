"""I13 / K2b: a janela de cada linha é alinhada pelo almighty sobre o áudio completo (ele separa a
voz e compara com o mix, alt_audio). O alinhamento de linhas sem tempo (plain) segue no vocal."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from almighty_fake import FakeAlmighty  # noqa: E402
from src.model.lyrics import Line, Lyrics  # noqa: E402
from src.model.song_job import SongJob  # noqa: E402
from src.planner import default_stages  # noqa: E402
from src.stages import almighty_runners as ar  # noqa: E402
from src.stages.verify_lyrics import analysis_audio  # noqa: E402

LINES = {
    "en": ["When I find myself", "In times of trouble"],
    "pt": ["Eu vi o menino", "Correndo na rua"],
    "es": ["Ojalá que llueva", "Café en el campo"],
}


class RecordingAligner:
    def __init__(self):
        self.audio = []

    def __call__(self, audio, words, language, window):
        self.audio.append(Path(audio).name)
        s, e = window if window else (0.0, float(len(words)))
        step = (e - s) / len(words)
        return [(s + i * step, s + (i + 1) * step) for i in range(len(words))]


class AlignmentAudioTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.mix = self.tmp / "mix.mp3"
        self.voc = self.tmp / "vocals.mp3"
        self.mix.write_bytes(b"m")
        self.voc.write_bytes(b"v")

    def tearDown(self):
        self._tmp.cleanup()

    def job(self, lang, synced, vocals=True):
        lines = [Line(text=t, start=(2.0 + 4 * i) if synced else None)
                 for i, t in enumerate(LINES[lang])]
        return SongJob(artist="A", title="T", language=lang,
                       audio_path=self.mix,
                       vocals_path=self.voc if vocals else None,
                       lyrics=Lyrics(lines=lines, language=lang))

    def test_window_alignment_goes_through_almighty_on_full_track(self):
        for lang in LINES:
            with self.subTest(lang):
                fake = FakeAlmighty(self.tmp)
                stages = default_stages(self.tmp, aligner=RecordingAligner())
                job = self.job(lang, synced=True)
                with mock.patch.object(ar, "_default_listen", return_value=fake):
                    stages["align_window"](job)
                call = fake.calls[0]
                self.assertEqual(call["path"], self.mix)
                self.assertEqual(call["want"], ["transcript", "alignment"])
                self.assertTrue(call["options"]["transcript"]["alt_audio"])
                self.assertEqual(call["work_dir"], self.tmp / "A - T" / ".work" / ".almighty")
                self.assertTrue(job.lyrics.has_syllable_timing)
                self.assertNotIn("review", job.extra)

    def test_window_alignment_does_not_need_vocals(self):
        fake = FakeAlmighty(self.tmp)
        stages = default_stages(self.tmp)
        job = self.job("pt", synced=True, vocals=False)
        with mock.patch.object(ar, "_default_listen", return_value=fake):
            stages["align_window"](job)
        self.assertEqual(fake.calls[0]["path"], self.mix)

    def test_failed_line_is_flagged_for_review(self):
        fake = FakeAlmighty(self.tmp, fail_on=("Eu",))
        stages = default_stages(self.tmp)
        job = self.job("pt", synced=True)
        with mock.patch.object(ar, "_default_listen", return_value=fake):
            stages["align_window"](job)
        self.assertEqual(job.extra["review"], "alignment failed on lines [0]")

    def test_plain_alignment_uses_vocals(self):
        for lang in LINES:
            with self.subTest(lang):
                al = RecordingAligner()
                stages = default_stages(self.tmp, aligner=al)
                job = self.job(lang, synced=False)
                stages["align_plain"](job)
                self.assertEqual(set(al.audio), {"vocals.mp3"})

    def test_missing_vocals_file_falls_back_to_mix(self):
        job = self.job("es", synced=True)
        self.voc.unlink()
        self.assertEqual(analysis_audio(job), self.mix)


if __name__ == "__main__":
    unittest.main()
