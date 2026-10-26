import tempfile
import unittest
from pathlib import Path

from src.model import SongJob
from src.readers.metadata import parse_filename, read_metadata

NAMES = {
    "en": ("Journey - Don't Stop Believin'.mp3", "Journey", "Don't Stop Believin'"),
    "pt": ("Legião Urbana - Pais e Filhos.mp3", "Legião Urbana", "Pais e Filhos"),
    "es": ("Maná - Corazón Espinado.mp3", "Maná", "Corazón Espinado"),
}


class FilenameTest(unittest.TestCase):
    def test_languages(self):
        for lang, (name, artist, title) in NAMES.items():
            with self.subTest(lang):
                self.assertEqual(parse_filename(name), (artist, title))

    def test_track_number_and_hyphen_in_title(self):
        self.assertEqual(parse_filename("03 - Soda Stereo - De Música Ligera.mp3"),
                         ("Soda Stereo", "De Música Ligera"))
        self.assertEqual(parse_filename("A - B - C.mp3"), ("A", "B - C"))

    def test_no_separator(self):
        self.assertEqual(parse_filename("Sozinho.mp3"), (None, "Sozinho"))
        self.assertEqual(parse_filename("Jean-Michel.mp3"), (None, "Jean-Michel"))


class ReadMetadataTest(unittest.TestCase):
    def info(self, **tags):
        return {"format": {"duration": "213.4", "tags": tags},
                "streams": [{"codec_type": "audio"}]}

    def test_tags_win_over_filename(self):
        info = self.info(ARTIST="Legião Urbana", TITLE="Pais e Filhos",
                         ALBUM="As Quatro Estações", date="1989-05-01")
        job = read_metadata("x/Outro - Nome.mp3", probe=lambda p: info)
        self.assertEqual((job.artist, job.title), ("Legião Urbana", "Pais e Filhos"))
        self.assertEqual((job.album, job.year), ("As Quatro Estações", 1989))
        self.assertAlmostEqual(job.duration, 213.4)
        self.assertEqual(job.sources["artist"], "tags")
        self.assertEqual(job.audio_path, Path("x/Outro - Nome.mp3"))

    def test_filename_fallback_for_missing_tags(self):
        job = read_metadata("Maná - Corazón Espinado.mp3",
                            probe=lambda p: self.info(title="Corazón Espinado"))
        self.assertEqual(job.artist, "Maná")
        self.assertEqual(job.sources["artist"], "filename")
        self.assertEqual(job.sources["title"], "tags")

    def test_probe_failure_uses_filename(self):
        job = read_metadata("Journey - Don't Stop Believin'.mp3", probe=lambda p: None)
        self.assertEqual((job.artist, job.title), ("Journey", "Don't Stop Believin'"))
        self.assertIsNone(job.year)

    def test_existing_job_values_kept(self):
        job = SongJob(artist="Given")
        read_metadata("A - B.mp3", job=job, probe=lambda p: self.info(artist="Tag"))
        self.assertEqual(job.artist, "Given")
        self.assertEqual(job.title, "B")

    def test_embedded_cover(self):
        info = self.info(artist="A", title="B")
        info["streams"].append({"codec_type": "video"})

        def extract(audio, dest):
            dest.write_bytes(b"jpg")
            return True

        with tempfile.TemporaryDirectory() as tmp:
            job = read_metadata("A - B.mp3", probe=lambda p: info,
                                extract_cover=extract, cover_dir=tmp)
            self.assertEqual(job.cover_path.read_bytes(), b"jpg")
            self.assertEqual(job.sources["cover"], "tags")

    def test_no_cover_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = read_metadata("A - B.mp3", probe=lambda p: self.info(),
                                extract_cover=lambda a, d: True, cover_dir=tmp)
            self.assertIsNone(job.cover_path)


if __name__ == "__main__":
    unittest.main()
