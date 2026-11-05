import json
import tempfile
import unittest

from src.model import SongJob
from src.providers.http import HttpResponse
from src.providers.musicbrainz import MusicBrainzClient

UA = "UltraStarChartGenerator/0.1 (test@example.com)"


def recording(score=100, date="1981-10-01", rid="rel-1", title="Escape"):
    return {"recordings": [{
        "score": score, "first-release-date": date,
        "releases": [{"id": rid, "title": title, "status": "Official",
                      "date": date}]}]}


class FakeHttp:
    def __init__(self, mb, cover=b"IMG"):
        self.mb, self.cover, self.calls = mb, cover, []

    def __call__(self, url, headers):
        self.calls.append((url, headers))
        if "musicbrainz.org" in url:
            return HttpResponse(200, json.dumps(self.mb).encode())
        if self.cover is None:
            return HttpResponse(404, b"")
        return HttpResponse(200, self.cover)


class Clock:
    def __init__(self):
        self.t, self.slept = 100.0, []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class MusicBrainzTest(unittest.TestCase):
    def client(self, http, tmp=None, clock=None):
        clock = clock or Clock()
        return MusicBrainzClient(UA, http=http, cover_dir=tmp,
                                 clock=clock.now, sleep=clock.sleep)

    def test_fills_missing_all_languages(self):
        songs = [("Journey", "Don't Stop Believin'", "Escape"),
                 ("Legião Urbana", "Pais e Filhos", "As Quatro Estações"),
                 ("Maná", "Corazón Espinado", "¿Dónde Jugarán los Niños?")]
        for artist, title, album in songs:
            with self.subTest(artist), tempfile.TemporaryDirectory() as tmp:
                http = FakeHttp(recording(title=album))
                job = self.client(http, tmp).fill(SongJob(artist=artist, title=title))
                self.assertEqual((job.year, job.album), (1981, album))
                self.assertEqual(job.cover_path.read_bytes(), b"IMG")
                self.assertEqual(job.sources["year"], "musicbrainz")
                self.assertEqual(http.calls[0][1]["User-Agent"], UA)

    def test_existing_values_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            http = FakeHttp(recording())
            job = SongJob(artist="a", title="b", year=1999, album="Mine")
            self.client(http, tmp).fill(job)
            self.assertEqual((job.year, job.album), (1999, "Mine"))
            self.assertIsNotNone(job.cover_path)

    def test_nothing_missing_no_calls(self):
        http = FakeHttp(recording())
        job = SongJob(artist="a", title="b", year=1, album="x", cover_path="c.jpg")
        self.client(http, "d").fill(job)
        self.assertEqual(http.calls, [])

    def test_no_cover_dir_means_no_cover_request(self):
        http = FakeHttp(recording())
        job = self.client(http).fill(SongJob(artist="a", title="b"))
        self.assertIsNone(job.cover_path)
        self.assertEqual(len(http.calls), 1)

    def test_low_score_ignored(self):
        job = self.client(FakeHttp(recording(score=40))).fill(
            SongJob(artist="a", title="b"))
        self.assertIsNone(job.year)

    def test_missing_cover_art(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = self.client(FakeHttp(recording(), cover=None), tmp).fill(
                SongJob(artist="a", title="b"))
            self.assertIsNone(job.cover_path)
            self.assertEqual(job.year, 1981)

    def test_rate_limit(self):
        clock = Clock()
        client = self.client(FakeHttp(recording()), clock=clock)
        client.fill(SongJob(artist="a", title="b"))
        client.fill(SongJob(artist="c", title="d"))
        self.assertEqual(len(clock.slept), 1)
        self.assertGreaterEqual(clock.slept[0], 1.0)

    def test_user_agent_required(self):
        with self.assertRaises(ValueError):
            MusicBrainzClient("  ")


if __name__ == "__main__":
    unittest.main()
