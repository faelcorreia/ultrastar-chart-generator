import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from app import _parse_input_file  # noqa: E402


class InputFileParsingTest(unittest.TestCase):
    def test_full_track(self):
        parsed = _parse_input_file(Path(
            "Example Artist - Example Song - full track.mp3"
        ))

        self.assertEqual(
            parsed["artist_song"], "Example Artist - Example Song"
        )
        self.assertEqual(parsed["artist"], "Example Artist")
        self.assertEqual(parsed["title"], "Example Song")
        self.assertEqual(parsed["role"], "full_track")
        self.assertEqual(parsed["language"], "pt")

    def test_lyrics_with_language(self):
        parsed = _parse_input_file(
            Path("Artist-Name - Song Title - en lyrics.txt")
        )

        self.assertEqual(parsed["artist"], "Artist-Name")
        self.assertEqual(parsed["title"], "Song Title")
        self.assertEqual(parsed["role"], "lyrics")
        self.assertEqual(parsed["language"], "en")

    def test_lyrics_without_language_defaults_to_portuguese(self):
        parsed = _parse_input_file(Path("Artist - Song - lyrics.txt"))

        self.assertEqual(parsed["role"], "lyrics")
        self.assertEqual(parsed["language"], "pt")

    def test_cover(self):
        parsed = _parse_input_file(Path("Artist - Song - cover.jpg"))

        self.assertEqual(parsed["role"], "cover")

    def test_unrelated_file_is_ignored(self):
        self.assertIsNone(_parse_input_file(Path("notes.txt")))

    def test_missing_artist_title_separator_is_ignored(self):
        self.assertIsNone(_parse_input_file(Path("Song - full track.mp3")))


if __name__ == "__main__":
    unittest.main()
