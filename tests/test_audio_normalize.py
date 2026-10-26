import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.stages.audio_normalize import (
    AudioNormalizeError, is_supported_audio, normalize_audio)

NAMES = ["Artist - Song - full track", "Beyoncé - Halo - full track",
         "Peso Pluma - Ella Baila Sola ñandú - full track",
         "João Gilberto - Garota de Ipanema - full track",
         "Mägo de Oz - La Danza del Fuego - full track"]


def fake_runner(calls, code=0, write=True, stderr=""):
    def run(cmd):
        calls.append(cmd)
        if write and not code:
            Path(cmd[-1]).write_bytes(b"mp3")
        return SimpleNamespace(returncode=code, stderr=stderr)
    return run


class AudioNormalizeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.which = lambda n: "/usr/bin/ffmpeg"

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, name):
        p = self.dir / name
        p.write_bytes(b"x")
        return p

    def test_converts_each_format_with_accented_names(self):
        for stem in NAMES:
            for ext in (".flac", ".wav", ".m4a", ".ogg", ".opus", ".FLAC", ".Opus"):
                src = self.make(stem + ext)
                calls = []
                out = normalize_audio(src, self.dir / "out", runner=fake_runner(calls),
                                      which=self.which)
                self.assertEqual(out, self.dir / "out" / (stem + ".mp3"))
                self.assertEqual(len(calls), 1)
                self.assertIn(str(src), calls[0])
                self.assertIn("libmp3lame", calls[0])

    def test_mp3_not_reencoded_case_insensitive(self):
        for name in ("Ñandú - Canción - full track.mp3", "A - B - full track.MP3"):
            src = self.make(name)
            calls = []
            out = normalize_audio(src, runner=fake_runner(calls), which=lambda n: None)
            self.assertEqual(out, src)
            self.assertEqual(calls, [])

    def test_unsupported_and_missing(self):
        with self.assertRaisesRegex(AudioNormalizeError, "unsupported"):
            normalize_audio(self.make("a - b - full track.wma"))
        with self.assertRaisesRegex(AudioNormalizeError, "not found"):
            normalize_audio(self.dir / "nope.flac")

    def test_missing_ffmpeg(self):
        with self.assertRaisesRegex(AudioNormalizeError, "ffmpeg not found"):
            normalize_audio(self.make("a.flac"), which=lambda n: None)

    def test_ffmpeg_failure_and_empty_output(self):
        src = self.make("a.wav")
        with self.assertRaisesRegex(AudioNormalizeError, "exit 1\\): boom"):
            normalize_audio(src, runner=fake_runner([], code=1, stderr="boom"),
                            which=self.which)
        with self.assertRaisesRegex(AudioNormalizeError, "no audio"):
            normalize_audio(src, runner=fake_runner([], write=False), which=self.which)

    def test_is_supported(self):
        self.assertTrue(is_supported_audio("Canção.OGG"))
        self.assertFalse(is_supported_audio("lyrics.txt"))


if __name__ == "__main__":
    unittest.main()
