import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from src.model.lyrics import Lyrics  # noqa: E402
from src.model.song_job import SongJob  # noqa: E402
from src.stages.cache import (  # noqa: E402
    StageCache, decode, encode, fingerprint, run_stage)

SONGS = {
    "en": ("The Beatles - Let It Be", "Whisper words\nLet it be"),
    "pt": ("Caetano Veloso - Força Estranha", "Eu vi o menino correndo\nação"),
    "es": ("Shakira - Sueño Mañana", "Ojalá que llueva café\nniño"),
}


class CacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip_lyrics_and_job(self):
        for lang, (name, text) in SONGS.items():
            job = SongJob(artist=name, language=lang,
                          audio_path=self.tmp / f"{name}.mp3",
                          lyrics=Lyrics.from_text(text, language=lang),
                          extra={"p": self.tmp / "x.json"})
            back = decode(encode(job))
            self.assertEqual(back, job)

    def test_stage_skipped_when_unchanged_and_recomputed_when_stale(self):
        for lang, (name, text) in SONGS.items():
            audio = self.tmp / f"{lang}.mp3"
            audio.write_bytes(b"a")
            out = self.tmp / f"{lang} - beats.json"
            calls = []

            def func(song, out=out):
                calls.append(1)
                out.write_text("{}")
                song["beats"] = out

            cache = StageCache(self.tmp, name)
            key = fingerprint(audio, lang, text)
            song = {}
            self.assertFalse(run_stage(cache, "beats", key, song,
                                       lambda: func(song)))
            song2 = {}
            self.assertTrue(run_stage(cache, "beats", key, song2,
                                      lambda: func(song2)))
            self.assertEqual(song2["beats"], out)
            self.assertEqual(len(calls), 1)
            self.assertTrue((self.tmp / name / ".work" / "beats.json").exists())

            # Changed lyrics -> new key -> recompute.
            key2 = fingerprint(audio, lang, text + "\nnova")
            song3 = {}
            self.assertFalse(run_stage(cache, "beats", key2, song3,
                                       lambda: func(song3)))
            # Deleted output -> recompute.
            out.unlink()
            song4 = {}
            self.assertFalse(run_stage(cache, "beats", key2, song4,
                                       lambda: func(song4)))
            # Force (disabled cache) recomputes and refreshes.
            forced = StageCache(self.tmp, name, enabled=False)
            song5 = {}
            self.assertFalse(run_stage(forced, "beats", key2, song5,
                                       lambda: func(song5)))
            self.assertEqual(len(calls), 4)

    def test_corrupt_entry_is_ignored(self):
        cache = StageCache(self.tmp, "A - B")
        cache.save("s", "k", {"a": 1})
        cache.path("s").write_text("{broken")
        self.assertIsNone(cache.load("s", "k"))


if __name__ == "__main__":
    unittest.main()
