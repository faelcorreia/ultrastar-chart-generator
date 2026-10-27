"""I13: word alignment runs on the separated vocals, not on the full mix."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from src.model.lyrics import Line, Lyrics  # noqa: E402
from src.model.song_job import SongJob  # noqa: E402
from src.planner import default_stages  # noqa: E402
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

    def test_window_alignment_uses_vocals(self):
        for lang in LINES:
            with self.subTest(lang):
                al = RecordingAligner()
                stages = default_stages(self.tmp, aligner=al)
                job = self.job(lang, synced=True)
                stages["align_window"](job)
                self.assertEqual(set(al.audio), {"vocals.mp3", "mix.mp3"})
                self.assertEqual(al.audio[0], "vocals.mp3")
                self.assertTrue(job.lyrics.has_syllable_timing)

    def test_line_follows_the_pass_nearest_its_own_start(self):
        """Whichever pass drifts from the line start loses (vocals lose a
        soft onset, or the mix is masked by instruments)."""
        from src.stages.window_alignment import align_in_windows
        for lang in LINES:
            for drifting in ("vocals.mp3", "mix.mp3"):
                with self.subTest(lang=lang, drifting=drifting):
                    def aligner(audio, words, language, window,
                                drifting=drifting):
                        shift = 1.5 if Path(audio).name == drifting else 0.0
                        s = window[0] + shift
                        return [(s + i * 0.2, s + i * 0.2 + 0.2)
                                for i in range(len(words))]
                    job = self.job(lang, synced=True)
                    res = align_in_windows(
                        job.lyrics, self.voc, aligner=aligner,
                        alt_audio_path=self.mix, syllabifier=None)
                    first = res.lyrics.lines[0].syllables[0].start
                    self.assertAlmostEqual(first, 2.0)

    def test_plain_alignment_uses_vocals(self):
        for lang in LINES:
            with self.subTest(lang):
                al = RecordingAligner()
                stages = default_stages(self.tmp, aligner=al)
                job = self.job(lang, synced=False)
                stages["align_plain"](job)
                self.assertEqual(set(al.audio), {"vocals.mp3"})

    def test_falls_back_to_mix_without_vocals(self):
        for lang in LINES:
            with self.subTest(lang):
                al = RecordingAligner()
                stages = default_stages(self.tmp, aligner=al)
                job = self.job(lang, synced=True, vocals=False)
                stages["align_window"](job)
                self.assertEqual(set(al.audio), {"mix.mp3"})
                self.assertEqual(analysis_audio(job), self.mix)

    def test_missing_vocals_file_falls_back_to_mix(self):
        job = self.job("es", synced=True)
        self.voc.unlink()
        self.assertEqual(analysis_audio(job), self.mix)


if __name__ == "__main__":
    unittest.main()
