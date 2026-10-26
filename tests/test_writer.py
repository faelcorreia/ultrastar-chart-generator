import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

import src.core.config as config  # noqa: E402
from src.core.ultrastar_formatter import UltraStarFormatter as OldPath  # noqa: E402
from src.model import Line, Lyrics, Syllable  # noqa: E402
from src.writer import (UltraStarFormatter, lyrics_to_syllable_dicts,  # noqa: E402
                        syllable_dicts_to_lyrics)

SAMPLES = {
    "en": ("English", [["Hel", "lo"], ["dar", "kness"]]),
    "pt": ("Portuguese", [["Co", "ra", "ção"], ["não", "dói"]]),
    "es": ("Spanish", [["Ma", "ña", "na"], ["can", "ción"]]),
}


def build(code):
    """Return (dicts, Lyrics) with the same timing for a language sample."""
    _, lines = SAMPLES[code]
    dicts, model_lines, t = [], [], 0.5
    for words in lines:
        syls = []
        for i, text in enumerate(words):
            dicts.append({"syllable": text, "start": t, "end": t + 0.4,
                          "line_break": i == len(words) - 1,
                          "first_syllable": i == 0})
            syl = Syllable(text, t, t + 0.4)
            if i:
                syl.word_start = False
            syls.append(syl)
            t += 0.5
        model_lines.append(Line(syllables=syls))
        t += 0.3
    return dicts, Lyrics(lines=model_lines, language=code)


def run(root, name, code, mode):
    sources = {k: root / f"{k}.mp3" for k in
               ("full_track", "instrumental", "vocals")}
    sources["cover"] = root / "cover.jpg"
    for p in sources.values():
        p.write_bytes(b"x")
    dicts, lyrics = build(code)
    times = [i * 0.01 for i in range(400)]
    (root / "p.json").write_text(json.dumps({
        "times": times, "midi_notes": [60.0 + (i // 100) for i in range(400)]}))
    (root / "b.json").write_text(json.dumps({"bpm": 100}))
    (root / "s.json").write_text(json.dumps(dicts), encoding="utf-8")
    song = {**sources, "artist": "Ärtist", "title": "Canción", "language": code,
            "pitches": root / "p.json", "beats": root / "b.json",
            "syllables": root / "s.json"}
    out = root / "out"
    fmt = UltraStarFormatter({name: song})
    with mock.patch.object(config, "OUTPUT_DIR", out), \
            mock.patch("src.core.ultrastar_formatter.subprocess.run"):
        if mode == "lyrics":
            fmt.format_ultrastar_song_folder(name, lyrics=lyrics)
        else:
            fmt.format_ultrastar_song_folder(name)
    return (out / name / f"{name}.txt").read_bytes()


class WriterTest(unittest.TestCase):
    def test_old_import_path_is_the_same_class(self):
        self.assertIs(OldPath, UltraStarFormatter)

    def test_lyrics_input_matches_json_input_in_en_pt_es(self):
        for code, (display, _) in SAMPLES.items():
            with self.subTest(code), tempfile.TemporaryDirectory() as tmp:
                a = run(Path(tmp), "A - B", code, "json")
                with tempfile.TemporaryDirectory() as tmp2:
                    b = run(Path(tmp2), "A - B", code, "lyrics")
                self.assertEqual(a, b)
                text = a.decode("utf-8")
                self.assertIn(f"#LANGUAGE:{display}", text)
                self.assertIn("#ARTIST:Ärtist", text)

    def test_unicode_syllables_are_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = run(Path(tmp), "A - B", "pt", "lyrics").decode("utf-8")
        self.assertIn("ção", text)

    def test_dict_round_trip(self):
        for code in SAMPLES:
            dicts, lyrics = build(code)
            self.assertEqual(lyrics_to_syllable_dicts(lyrics), dicts)
            again = syllable_dicts_to_lyrics(dicts, code)
            self.assertEqual(lyrics_to_syllable_dicts(again), dicts)

    def test_lyrics_in_song_dict(self):
        dicts, lyrics = build("es")
        self.assertEqual(len(lyrics_to_syllable_dicts(lyrics)), len(dicts))


if __name__ == "__main__":
    unittest.main()
