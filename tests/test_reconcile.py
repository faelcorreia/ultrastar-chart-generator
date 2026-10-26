import unittest
from pathlib import Path

from src.model.lyrics import Lyrics
from src.stages.reconcile import (
    DISPLACED_VERSE, LOST_PASSAGE, MISSING_CHORUS, reconcile_lyrics,
)

SONGS = {
    "en": ("en", ["I walk alone down the empty street tonight",
                  "Carry me home under the pale moonlight",
                  "Sing it loud for the whole wide world to hear",
                  "We are the dreamers of the golden years",
                  "Morning will find us on the open road"]),
    "pt": ("pt", ["Eu caminho sozinho pela rua vazia hoje",
                  "Leva-me para casa sob a luz da lua",
                  "Canta bem alto para o mundo inteiro ouvir",
                  "Somos os sonhadores dos anos dourados",
                  "A manhã vai nos encontrar na estrada aberta"]),
    "es": ("es", ["Camino solo por la calle vacía esta noche",
                  "Llévame a casa bajo la luz de la luna",
                  "Canta fuerte para que el mundo entero oiga",
                  "Somos los soñadores de los años dorados",
                  "La mañana nos hallará en el camino abierto"]),
}


def whisper(lines, t0=0.0):
    out, t = [], t0
    for line in lines:
        for w in line.split():
            out.append((w, t, t + 0.4))
            t += 0.5
    return out


class ReconcileTest(unittest.TestCase):
    def test_perfect_match_ok(self):
        for lang, (_, v) in SONGS.items():
            with self.subTest(lang=lang):
                r = reconcile_lyrics(Lyrics.from_text("\n".join(v)),
                                     whisper(v), language=lang)
                self.assertEqual(r.status, "ok")
                self.assertEqual(r.issues, [])
                self.assertAlmostEqual(r.confidence, 1.0)
                self.assertEqual(len(r.decisions), 3)

    def test_missing_chorus_copied_and_confirmed(self):
        for lang, (_, v) in SONGS.items():
            with self.subTest(lang=lang):
                chorus = v[2:4]
                lyr = v[:2] + chorus + [v[4]]          # chorus written once
                heard = v[:2] + chorus + [v[4]] + chorus   # sung again
                calls = []

                def conf(audio, words, language, window):
                    calls.append((words, language, window))
                    return 0.9
                r = reconcile_lyrics(Lyrics.from_text("\n".join(lyr)),
                                     whisper(heard), Path("a.wav"), lang, conf)
                iss = r.issues_of(MISSING_CHORUS)
                self.assertEqual(len(iss), 1)
                self.assertTrue(iss[0].confirmed)
                self.assertFalse(iss[0].flagged)
                self.assertEqual([l.text for l in r.lyrics.lines],
                                 lyr + chorus)
                self.assertEqual(calls[0][1], lang)
                self.assertEqual(r.status, "ok")

    def test_unconfirmed_chorus_flagged(self):
        _, v = SONGS["pt"]
        lyr = v
        heard = v + v[2:4]
        r = reconcile_lyrics(Lyrics.from_text("\n".join(lyr)), whisper(heard),
                             Path("a.wav"), "pt", lambda *a: 0.1)
        iss = r.issues_of(MISSING_CHORUS)[0]
        self.assertFalse(iss.confirmed)
        self.assertTrue(iss.flagged)
        self.assertEqual(r.status, "review")

    def test_displaced_verse_moved(self):
        for lang, (_, v) in SONGS.items():
            with self.subTest(lang=lang):
                lyr = [v[0], v[3], v[1], v[2], v[4]]   # v3 displaced
                r = reconcile_lyrics(Lyrics.from_text("\n".join(lyr)),
                                     whisper(v), language=lang)
                self.assertTrue(r.issues_of(DISPLACED_VERSE))
                self.assertEqual([l.text for l in r.lyrics.lines], v)
                self.assertGreater(r.confidence, r.confidence_before - 1e-9)

    def test_lost_passage_and_review(self):
        _, v = SONGS["es"]
        r = reconcile_lyrics(Lyrics.from_text("\n".join(v)),
                             whisper([v[0], v[4]]), language="es")
        lost = r.issues_of(LOST_PASSAGE)
        self.assertEqual(lost[0].lines, [1, 2, 3])
        self.assertEqual(len(r.lyrics.lines), 5)
        self.assertEqual(r.status, "review")
        self.assertLess(r.confidence, 0.8)

    def test_accents_and_case_ignored(self):
        _, v = SONGS["es"]
        heard = [l.upper().replace("Í", "I") for l in v]
        r = reconcile_lyrics(Lyrics.from_text("\n".join(v)), whisper(heard),
                             language="es")
        self.assertEqual(r.status, "ok")

    def test_language_required(self):
        with self.assertRaises(ValueError):
            reconcile_lyrics(Lyrics.from_text("hello world"), [])


if __name__ == "__main__":
    unittest.main()
