import json
import unittest
from urllib.parse import parse_qs, urlparse

from src.providers.http import HttpResponse
from src.providers.lrclib import LrclibClient
from src.providers.name_clean import clean_name, lookup_variants
from src.readers.metadata import parse_filename

SYNCED = "[00:01.50] hello\n[00:05.00] world"

# (raw artist, raw title, expected cleaned artist, expected cleaned title)
CASES = {
    "en": ("03 - Journey", "Don't Stop Believin' (Official Video) [HD]",
           "Journey", "Don't Stop Believin'"),
    "pt": ("12 - Tribalistas", "Já Sei Namorar (Ao Vivo)",
           "Tribalistas", "Já Sei Namorar"),
    "es": ("07. Maná", "Corazón Espinado (En Vivo) [Remasterizado]",
           "Maná", "Corazón Espinado"),
}


class CascadeHttp:
    """Only answers /api/get for one exact (artist, title) pair."""

    def __init__(self, artist, title):
        self.target, self.calls = (artist, title), []

    def __call__(self, url, headers):
        parsed = urlparse(url)
        q = parse_qs(parsed.query)
        self.calls.append((parsed.path, q["artist_name"][0], q["track_name"][0]))
        if parsed.path == "/api/get" and (q["artist_name"][0],
                                          q["track_name"][0]) == self.target:
            body = {"id": 9, "artistName": self.target[0],
                    "trackName": self.target[1], "duration": 200,
                    "syncedLyrics": SYNCED, "plainLyrics": None}
            return HttpResponse(200, json.dumps(body).encode())
        if parsed.path == "/api/search":
            return HttpResponse(200, b"[]")
        return HttpResponse(404, b"{}")


class CleanNameTest(unittest.TestCase):
    def test_clean_all_languages(self):
        for lang, (a, t, ea, et) in CASES.items():
            with self.subTest(lang):
                self.assertEqual(clean_name(a), ea)
                self.assertEqual(clean_name(t), et)

    def test_non_noise_parentheses_kept_in_clean_name(self):
        self.assertEqual(clean_name("Song (Interlude)"), "Song (Interlude)")

    def test_numeric_titles_not_stripped(self):
        self.assertEqual(clean_name("99 Luftballons"), "Luftballons")
        self.assertEqual(clean_name("1979"), "1979")
        self.assertEqual(clean_name("22"), "22")

    def test_variants_ordered_specific_to_loose(self):
        v = lookup_variants("Tribalistas", "Já Sei Namorar (feat. X) (Live)")
        self.assertEqual(v[0], ("Tribalistas", "Já Sei Namorar (feat. X) (Live)"))
        self.assertEqual(v[-1], ("Tribalistas", "Já Sei Namorar"))
        self.assertEqual(len(v), len(set(v)))

    def test_clean_input_gives_single_variant(self):
        self.assertEqual(lookup_variants("Maná", "Rayando el Sol"),
                         [("Maná", "Rayando el Sol")])


class FilenameTest(unittest.TestCase):
    def test_prefix_variants(self):
        for name in ("12 - Tribalistas - Já Sei Namorar.mp3",
                     "12. Tribalistas - Já Sei Namorar.mp3",
                     "12 Tribalistas - Já Sei Namorar.mp3",
                     "12-Tribalistas - Já Sei Namorar.mp3"):
            with self.subTest(name):
                self.assertEqual(parse_filename(name),
                                 ("Tribalistas", "Já Sei Namorar"))

    def test_plain_name_unchanged(self):
        self.assertEqual(parse_filename("Journey - Faithfully.mp3"),
                         ("Journey", "Faithfully"))


class LrclibCascadeTest(unittest.TestCase):
    def test_cascade_finds_cleaned_name_all_languages(self):
        for lang, (a, t, ea, et) in CASES.items():
            with self.subTest(lang):
                http = CascadeHttp(ea, et)
                result = LrclibClient(http).find(a, t, duration=200)
                self.assertIsNotNone(result)
                self.assertEqual(result.id, 9)
                # exact name was tried first
                self.assertEqual(http.calls[0][1:], (a, t))

    def test_exact_name_wins_without_extra_calls(self):
        http = CascadeHttp("Maná", "Rayando el Sol")
        self.assertIsNotNone(LrclibClient(http).find("Maná", "Rayando el Sol"))
        self.assertEqual(len(http.calls), 1)

    def test_no_match_returns_none(self):
        http = CascadeHttp("x", "y")
        self.assertIsNone(LrclibClient(http).find("01 - A", "B (Live)"))

    def test_feat_dropped_as_loosest_step(self):
        http = CascadeHttp("Tribalistas", "Já Sei Namorar")
        r = LrclibClient(http).find("Tribalistas", "Já Sei Namorar (feat. Z)")
        self.assertIsNotNone(r)


if __name__ == "__main__":
    unittest.main()
