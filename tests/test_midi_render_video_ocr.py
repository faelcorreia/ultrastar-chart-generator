import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.readers.video_ocr import lyrics_from_samples, lyrics_from_video
from src.stages.midi_render import MidiRenderError, render_midi


class MidiRenderTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.mid = self.d / "a.mid"
        self.mid.write_bytes(b"MThd")
        self.sf = self.d / "x.sf2"
        self.sf.write_bytes(b"sf")
        self.out = self.d / "o" / "a.wav"

    def test_ok(self):
        calls = []

        def runner(cmd):
            calls.append(cmd)
            self.out.write_bytes(b"RIFF")
            return SimpleNamespace(returncode=0, stderr="")
        r = render_midi(self.mid, self.out, soundfont=self.sf, runner=runner,
                        which=lambda n: "/usr/bin/fluidsynth")
        self.assertEqual(r, self.out)
        self.assertIn(str(self.sf), calls[0])
        self.assertIn(str(self.mid), calls[0])

    def test_missing_fluidsynth(self):
        with self.assertRaisesRegex(MidiRenderError, "fluidsynth not found"):
            render_midi(self.mid, self.out, soundfont=self.sf,
                        which=lambda n: None)

    def test_missing_soundfont(self):
        with self.assertRaisesRegex(MidiRenderError, "soundfont not found"):
            render_midi(self.mid, self.out, soundfont=self.d / "no.sf2",
                        which=lambda n: "/x")

    def test_failure(self):
        r = lambda c: SimpleNamespace(returncode=1, stderr="boom")
        with self.assertRaisesRegex(MidiRenderError, "boom"):
            render_midi(self.mid, self.out, soundfont=self.sf, runner=r,
                        which=lambda n: "/x")


class VideoOcrTest(unittest.TestCase):
    def _run(self, texts, fps=1.0):
        frames = [(i / fps, f"f{i}") for i in range(len(texts))]
        m = dict(zip([f for _, f in frames], texts))
        return lyrics_from_video(
            "v.mp4", fps=fps, extractor=lambda v, f, d: frames,
            ocr=lambda p: m[p], language="xx")

    def test_en(self):
        ly = self._run(["", "Hello world", "Hello  world", "Hello wor1d",
                        "Second line", "", "Third"])
        self.assertEqual([l.text for l in ly.lines],
                         ["Hello world", "Second line", "Third"])
        self.assertEqual([l.start for l in ly.lines], [1.0, 4.0, 6.0])
        self.assertEqual(ly.lines[0].end, 4.0)
        self.assertEqual(ly.lines[1].end, 5.0)
        self.assertTrue(ly.is_synced)

    def test_pt_accents(self):
        ly = self._run(["Coração partido", "Coracao partido",
                        "Não vou chorar"])
        self.assertEqual(len(ly.lines), 2)
        self.assertEqual(ly.lines[0].text, "Coração partido")

    def test_es(self):
        ly = lyrics_from_samples([(0, "¿Qué pasó?"), (0.5, "¿Qué pasó?"),
                                  (1.0, "El niño canta")], fps=2.0)
        self.assertEqual([l.text for l in ly.lines],
                         ["¿Qué pasó?", "El niño canta"])
        self.assertEqual(ly.lines[1].end, 1.5)


if __name__ == "__main__":
    unittest.main()
