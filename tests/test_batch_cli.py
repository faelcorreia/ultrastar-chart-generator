import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from src.batch.cli import main  # noqa: E402
from src.batch.locate import (find_cdg, index_audio, index_cdg,  # noqa: E402
                              library_keys)
from src.batch.runner import SKIPPED, run_batch  # noqa: E402
from src.batch.songlist import parse_song_list, song_key  # noqa: E402
from src.model.lyrics import Line, Lyrics, Syllable  # noqa: E402
from src.planner import DONE, AWAITING_AUDIO  # noqa: E402

LIST = """artista;título;ano;idioma
# comment

Beyoncé;Halo;2008;en
João Gilberto;Chega de Saudade;1959
Peso Pluma;Ella Baila Sola;2023;es
Mägo de Oz;La Danza del Fuego;1998;es
Niño Sueño;Año Nuevo;2001
"""


def fake_stages(calls):
    def mk(name):
        def stage(job):
            calls.append((name, job.artist))
            if name == "language" and not job.language:
                job.language = "pt"
            if name == "lyrics":
                job.lyrics = Lyrics(lines=[Line(text="la")],
                                    language=job.language)
            if name == "align_plain":
                job.lyrics = Lyrics(lines=[Line(syllables=[
                    Syllable("la", 0.0, 1.0)])], language=job.language)
            if name == "write":
                job.extra["written"] = True
        return stage
    names = ["normalize", "metadata", "lyrics", "language", "vocals", "align",
             "reconcile", "sync_validate", "align_window", "align_plain",
             "pitches", "beats", "syllables",
             "write"]
    return {n: mk(n) for n in names}


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.audio = self.tmp / "audio"
        self.audio.mkdir()
        self.lst = self.tmp / "list.csv"
        self.lst.write_text(LIST, encoding="utf-8")

    def test_parse(self):
        e = parse_song_list(self.lst)
        self.assertEqual(len(e), 5)
        self.assertEqual((e[0].artist, e[0].year, e[0].language),
                         ("Beyoncé", 2008, "en"))
        self.assertIsNone(e[1].language)
        self.assertEqual(e[4].line, 8)

    def test_bad_year(self):
        self.lst.write_text("a;b;xx\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            parse_song_list(self.lst)

    def test_audio_match_accent_case_tolerant(self):
        (self.audio / "beyonce - HALO.mp3").touch()
        (self.audio / "Joao Gilberto - Chega de Saudade.flac").touch()
        (self.audio / "peso pluma - ella baila sola.mp3").touch()
        (self.audio / "Mago de Oz - La Danza del Fuego.mp3").touch()
        idx = index_audio(self.audio)
        self.assertIn(song_key("Beyoncé", "Halo"), idx)
        self.assertIn(song_key("João Gilberto", "Chega de Saudade"), idx)
        self.assertIn(song_key("Mägo de Oz", "La Danza del Fuego"), idx)
        self.assertEqual(len(idx), 4)

    def test_cdg_folder_entries_strip_catalog_prefix(self):
        from src.batch.cdg_cli import entries_from_cdg_folder
        for name in ("SF337-01 - Saturdays - Not Giving Up.cdg",
                     "SFWS072-01 - Elton John - Bennie & the Jets.cdg",
                     "U2 - One.cdg", "AKB48 - Koi Suru Fortune Cookie.cdg"):
            (self.audio / name).touch()
        entries = entries_from_cdg_folder(self.audio)
        self.assertEqual(
            [(e.artist, e.title) for e in entries],
            [("AKB48", "Koi Suru Fortune Cookie"),
             ("Saturdays", "Not Giving Up"),
             ("Elton John", "Bennie & the Jets"), ("U2", "One")])

    def test_cdg_folder_entries_strip_label(self):
        from src.batch.cdg_cli import entries_from_cdg_folder
        (self.audio / "Van Morrison - Brown Eyed Girl [HM Karaoke].cdg").touch()
        (self.audio / "not a pair.cdg").touch()
        entries = entries_from_cdg_folder(self.audio)
        self.assertEqual([(e.artist, e.title) for e in entries],
                         [("Van Morrison", "Brown Eyed Girl")])

    def test_cdg_pair_with_label_suffix_matches_clean_title(self):
        (self.audio / "Van Morrison - Brown Eyed Girl [HM Karaoke].mp3").touch()
        (self.audio / "Van Morrison - Brown Eyed Girl [HM Karaoke].cdg").touch()
        idx = index_audio(self.audio)
        self.assertEqual(
            find_cdg(index_cdg(self.audio), "Van Morrison", "Brown Eyed Girl"),
            self.audio / "Van Morrison - Brown Eyed Girl [HM Karaoke].cdg")
        self.assertIn(song_key("Van Morrison", "Brown Eyed Girl"), idx)

    def test_run_skip_existing_and_force(self):
        for n in ("Beyoncé - Halo", "Niño Sueño - Año Nuevo"):
            (self.audio / f"{n}.mp3").touch()
        lib = self.tmp / "lib"
        (lib / "BEYONCE - halo").mkdir(parents=True)
        self.assertEqual(library_keys(lib), {song_key("Beyoncé", "Halo")})
        calls = []
        out = self.tmp / "out"
        res = main([str(self.lst), "--audio-dir", str(self.audio),
                    "--output-dir", str(out), "--library", str(lib)],
                   stages=fake_stages(calls))
        self.assertEqual([r.status for r in res],
                         [SKIPPED, AWAITING_AUDIO, AWAITING_AUDIO,
                          AWAITING_AUDIO, DONE])
        self.assertNotIn("Beyoncé", [a for _, a in calls])
        # language came from list when given, else from the language stage
        self.assertEqual(res[4].job.language, "pt")
        res2 = main([str(self.lst), "--audio-dir", str(self.audio),
                     "--output-dir", str(self.tmp / "out2"), "--library",
                     str(lib), "--force-existing"],
                    stages=fake_stages([]))
        self.assertNotEqual(res2[0].status, SKIPPED)
        self.assertEqual(res2[0].job.language, "en")
        self.assertEqual(res2[2].job.language, "es")

    def test_sequential_results_in_order(self):
        seen = []
        e = parse_song_list(self.lst)
        run_batch(e, self.audio, self.tmp / "o", fake_stages([]),
                  on_result=lambda en, r: seen.append(en.title))
        self.assertEqual(seen, [x.title for x in e])


if __name__ == "__main__":
    unittest.main()
