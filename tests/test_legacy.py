import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from app import _collect_songs  # noqa: E402
from src.model.legacy import absorb, job_from_song, song_from_job  # noqa: E402


def _song(language):
    return {
        "artist": "Artist", "title": "Song", "language": language,
        "full_track": Path("in/Artist - Song - full track.mp3"),
        "lyrics": Path(f"in/Artist - Song - {language} lyrics.txt"),
        "cover": Path("in/Artist - Song - cover.jpg"),
    }


class LegacyBridgeTest(unittest.TestCase):
    def test_round_trip_is_identical_per_language(self):
        for language in ("en", "pt", "es"):
            with self.subTest(language=language):
                song = _song(language)
                job = job_from_song(song)
                self.assertEqual(job.language, language)
                self.assertEqual(job.lyrics.language, language)
                self.assertEqual(job.audio_path, song["full_track"])
                self.assertEqual(song_from_job(job), song)

    def test_language_is_carried_not_defaulted(self):
        job = job_from_song({"artist": "A", "title": "T"})
        self.assertIsNone(job.language)
        self.assertIsNone(song_from_job(job)["language"])

    def test_stage_outputs_are_absorbed(self):
        for language in ("en", "pt", "es"):
            with self.subTest(language=language):
                song = _song(language)
                job = job_from_song(song)
                song.update({
                    "instrumental": Path("out/inst.mp3"),
                    "vocals": Path("out/vocals.mp3"),
                    "beats": Path("out/beats.json"),
                })
                absorb(job, song)
                self.assertEqual(job.vocals_path, Path("out/vocals.mp3"))
                self.assertEqual(job.extra["beats"], Path("out/beats.json"))
                self.assertEqual(song_from_job(job), song)

    def test_collected_songs_keep_language_through_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            for language in ("en", "pt", "es"):
                base = f"Artist {language} - Song"
                for suffix in ("full track.mp3", f"{language} lyrics.txt",
                               "cover.jpg"):
                    (Path(tmp) / f"{base} - {suffix}").touch()
            songs = _collect_songs(Path(tmp))
            self.assertEqual(len(songs), 3)
            for name, song in songs.items():
                job = job_from_song(song)
                self.assertEqual(song_from_job(job), song)
                self.assertEqual(job.language, song["language"])


if __name__ == "__main__":
    unittest.main()
