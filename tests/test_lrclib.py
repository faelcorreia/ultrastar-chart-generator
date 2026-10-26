import json
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse

from src.providers.http import HttpResponse
from src.providers.lrclib import LrclibClient

SYNCED = "[00:01.50] Não vou ficar\n[00:05.00] Coração\n[00:09.25]"
SONGS = {
    "en": ("Journey", "Don't Stop Believin'", 250),
    "pt": ("Legião Urbana", "Pais e Filhos", 308),
    "es": ("Maná", "Corazón Espinado", 270),
}


def rec(i, artist, title, dur, synced=None, plain=None, instrumental=False):
    return {"id": i, "artistName": artist, "trackName": title, "duration": dur,
            "syncedLyrics": synced, "plainLyrics": plain,
            "instrumental": instrumental}


class FakeHttp:
    def __init__(self, get=None, search=None):
        self.get, self.search, self.calls = get, search, []

    def __call__(self, url, headers):
        self.calls.append((url, headers))
        path = urlparse(url).path
        data = self.get if path == "/api/get" else self.search
        if data is None:
            return HttpResponse(404, b"{}")
        return HttpResponse(200, json.dumps(data).encode())


class LrclibTest(unittest.TestCase):
    def test_synced_from_get_all_languages(self):
        for lang, (artist, title, dur) in SONGS.items():
            with self.subTest(lang):
                http = FakeHttp(get=rec(1, artist, title, dur + 1, SYNCED, "x"))
                result = LrclibClient(http).find(artist, title, duration=dur)
                self.assertEqual(result.kind, "synced")
                self.assertEqual(len(http.calls), 1)
                q = parse_qs(urlparse(http.calls[0][0]).query)
                self.assertEqual(q["artist_name"], [artist])
                self.assertIn("User-Agent", http.calls[0][1])

    def test_lyrics_conversion(self):
        http = FakeHttp(get=rec(7, "a", "b", 100, SYNCED))
        lyrics = LrclibClient(http).find("a", "b", duration=100).to_lyrics("pt")
        self.assertEqual([l.text for l in lyrics.lines], ["Não vou ficar", "Coração"])
        self.assertAlmostEqual(lyrics.lines[0].start, 1.5)
        self.assertTrue(lyrics.is_synced)
        self.assertEqual(lyrics.source, "lrclib:7")

    def test_duration_outside_tolerance_rejected(self):
        http = FakeHttp(get=rec(1, "a", "b", 200, SYNCED),
                        search=[rec(2, "a", "b", 204, SYNCED)])
        self.assertIsNone(LrclibClient(http).find("a", "b", duration=190))

    def test_tolerance_boundary_accepts_3s(self):
        http = FakeHttp(get=rec(1, "a", "b", 193, SYNCED))
        self.assertIsNotNone(LrclibClient(http).find("a", "b", duration=190))

    def test_search_prefers_synced_and_closest(self):
        http = FakeHttp(search=[
            rec(1, "a", "b", 200, plain="p"),
            rec(2, "a", "b", 202, SYNCED),
            rec(3, "a", "b", 200.5, SYNCED),
            rec(4, "a", "b", 100, SYNCED)])
        result = LrclibClient(http).find("a", "b", duration=200)
        self.assertEqual(result.id, 3)

    def test_plain_get_upgraded_by_synced_search(self):
        http = FakeHttp(get=rec(1, "a", "b", 200, plain="p"),
                        search=[rec(2, "a", "b", 201, SYNCED)])
        self.assertEqual(LrclibClient(http).find("a", "b", duration=200).id, 2)

    def test_plain_only_when_no_synced(self):
        http = FakeHttp(get=rec(1, "El niño", "b", 200, plain="¿Qué será?"),
                        search=[rec(2, "a", "b", 200, plain="other")])
        result = LrclibClient(http).find("a", "b", duration=200)
        self.assertEqual((result.id, result.kind), (1, "plain"))
        self.assertEqual(result.to_lyrics("es").lines[0].text, "¿Qué será?")

    def test_instrumental_and_empty_ignored(self):
        http = FakeHttp(search=[rec(1, "a", "b", 200, instrumental=True),
                                rec(2, "a", "b", 200)])
        self.assertIsNone(LrclibClient(http).find("a", "b", duration=200))

    def test_no_duration_accepts_any(self):
        http = FakeHttp(get=rec(1, "a", "b", 999, SYNCED))
        self.assertIsNotNone(LrclibClient(http).find("a", "b"))

    def test_network_failure(self):
        self.assertIsNone(LrclibClient(lambda u, h: None).find("a", "b"))

    def test_disk_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            http = FakeHttp(get=rec(1, "Maná", "Corazón Espinado", 270, SYNCED))
            first = LrclibClient(http, cache_dir=tmp).find(
                "Maná", "Corazón Espinado", duration=270)
            calls = len(http.calls)
            second = LrclibClient(FakeHttp(), cache_dir=tmp).find(
                "maná", "corazón espinado", duration=270)
            self.assertEqual(second, first)
            self.assertEqual(len(http.calls), calls)

    def test_miss_is_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            http = FakeHttp()
            client = LrclibClient(http, cache_dir=tmp)
            self.assertIsNone(client.find("a", "b"))
            n = len(http.calls)
            self.assertIsNone(client.find("a", "b"))
            self.assertEqual(len(http.calls), n)


if __name__ == "__main__":
    unittest.main()
