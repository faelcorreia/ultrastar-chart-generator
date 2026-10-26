import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

import src.core.config as config  # noqa: E402
from src.writer import UltraStarFormatter  # noqa: E402

# (language, held syllable, following syllable)
SAMPLES = {
    "en": ("Love", "you"),
    "pt": ("Amor", "não"),
    "es": ("Corazón", "mañana"),
}


def write_song(root, code, held, nxt):
    sources = {k: root / f"{k}.mp3" for k in
               ("full_track", "instrumental", "vocals")}
    sources["cover"] = root / "cover.jpg"
    for p in sources.values():
        p.write_bytes(b"x")
    # Held syllable: 1.5 s, pitch 60 for 0.6 s then 64 for 0.9 s.
    # Next syllable: 0.5 s on a steady 62.
    times = [i * 0.01 for i in range(300)]
    midi = [60.0 if t < 0.6 else 64.0 if t < 1.5 else 62.0 for t in times]
    (root / "p.json").write_text(json.dumps(
        {"times": times, "midi_notes": midi}))
    (root / "b.json").write_text(json.dumps({"bpm": 100}))
    (root / "s.json").write_text(json.dumps([
        {"syllable": held, "start": 0.0, "end": 1.5,
         "line_break": False, "first_syllable": True},
        {"syllable": nxt, "start": 1.5, "end": 2.0,
         "line_break": True, "first_syllable": True},
    ]), encoding="utf-8")
    return {**sources, "artist": "A", "title": "T", "language": code,
            "pitches": root / "p.json", "beats": root / "b.json",
            "syllables": root / "s.json"}


class HeldNoteTest(unittest.TestCase):
    def test_held_syllable_continues_with_tilde_en_pt_es(self):
        for code, (held, nxt) in SAMPLES.items():
            with self.subTest(code), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out = root / "out"
                fmt = UltraStarFormatter(
                    {"s": write_song(root, code, held, nxt)})
                with mock.patch.object(config, "OUTPUT_DIR", out), \
                        mock.patch("src.core.ultrastar_formatter."
                                   "subprocess.run"):
                    fmt.format_ultrastar_song_folder("s")
                lines = (out / "s" / "s.txt").read_text(
                    encoding="utf-8").splitlines()
                notes = [ln for ln in lines if ln.startswith(": ")]
                self.assertEqual(len(notes), 3)
                self.assertTrue(notes[0].endswith(f" {held}"))
                self.assertTrue(notes[1].endswith(" ~"))
                self.assertNotIn(held, notes[1])
                self.assertTrue(notes[2].endswith(f" {nxt}"))
                self.assertEqual(sum(ln.endswith(" ~") for ln in notes), 1)
                # The continuation has a different pitch than the start.
                self.assertNotEqual(notes[0].split()[3], notes[1].split()[3])


if __name__ == "__main__":
    unittest.main()
