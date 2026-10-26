import unittest

from src.core.text_norm import fold_for_compare

from src.model.lyrics import Line, Lyrics
from src.stages.sync_validate import (
    DRIFT, LINES, NONE, OFFSET, SHIFTED, UNCERTAIN, OK,
    estimate_from_whisper_words, validate_sync,
)

SONGS = {
    "en": ["I walk alone down the empty street tonight",
           "Carry me home under the pale moonlight",
           "Sing it loud for the whole wide world to hear",
           "We are the dreamers of the golden years",
           "Morning will find us on the open road",
           "Hold on to the light we found"],
    "pt": ["Eu caminho sozinho pela rua vazia hoje",
           "Leva-me para casa sob a luz da lua",
           "Canta bem alto para o mundo inteiro ouvir",
           "Somos os sonhadores dos anos dourados",
           "A manhã vai nos encontrar na estrada aberta",
           "Segura a luz que encontramos"],
    "es": ["Camino solo por la calle vacía esta noche",
           "Llévame a casa bajo la luz de la luna",
           "Canta fuerte para que el mundo entero oiga",
           "Somos los soñadores de los años dorados",
           "La mañana nos hallará en el camino abierto",
           "Sujeta la luz que encontramos"],
}
TRUE = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def lyr(lang, starts):
    return Lyrics(lines=[Line(text=t, start=s, end=s + 5)
                         for t, s in zip(SONGS[lang], starts)], language=lang)


def est(times):
    return lambda a, l, lang: list(times)


def run(lang, src, w=TRUE, c=TRUE):
    return validate_sync("a.mp3", lyr(lang, src), lang, est(w), est(c))


class SyncValidateTest(unittest.TestCase):
    def test_all_ok(self):
        for lang in SONGS:
            r = run(lang, [t + 0.3 for t in TRUE])
            self.assertEqual(r.pct_ok, 100.0)
            self.assertEqual(r.correction.kind, NONE)
            self.assertEqual(r.status, "ok")

    def test_global_offset(self):
        for lang in SONGS:
            r = run(lang, [t + 3 for t in TRUE])
            self.assertEqual(r.pct_shifted, 100.0)
            self.assertEqual(r.correction.kind, OFFSET)
            self.assertEqual(r.pct_ok_after, 100.0)
            self.assertAlmostEqual(r.lyrics.lines[2].start, 30.0)
            self.assertAlmostEqual(r.lyrics.lines[2].end, 35.0)

    def test_linear_drift(self):
        for lang in SONGS:
            r = run(lang, [t * 1.03 for t in TRUE])
            self.assertEqual(r.correction.kind, DRIFT)
            self.assertEqual(r.pct_ok_after, 100.0)
            self.assertAlmostEqual(r.lyrics.lines[5].start, 60.0, places=3)

    def test_only_bad_lines(self):
        for lang in SONGS:
            src = list(TRUE)
            src[1] += 4
            src[4] -= 6
            r = run(lang, src)
            self.assertEqual([c.status for c in r.checks],
                             [OK, SHIFTED, OK, OK, SHIFTED, OK])
            self.assertEqual(r.correction.kind, LINES)
            self.assertEqual(r.correction.lines, [1, 4])
            self.assertAlmostEqual(r.lyrics.lines[1].start, 20.0)
            self.assertEqual(r.pct_ok_after, 100.0)

    def test_uncertain_goes_to_review(self):
        for lang in SONGS:
            w = list(TRUE)
            c = list(TRUE)
            c[3] = 47.0
            r = run(lang, TRUE, w, c)
            chk = r.checks[3]
            self.assertEqual(chk.status, UNCERTAIN)
            self.assertEqual((chk.source, chk.whisper, chk.ctc),
                             (40.0, 40.0, 47.0))
            self.assertEqual(r.status, "review")
            self.assertAlmostEqual(r.pct_uncertain, 100 / 6)
            self.assertEqual(r.lyrics.lines[3].start, 40.0)

    def test_missing_estimate_is_uncertain(self):
        r = run("en", TRUE, w=[None] + TRUE[1:])
        self.assertEqual(r.checks[0].status, UNCERTAIN)

    def test_whisper_words_helper(self):
        for lang in SONGS:
            words = []
            for t, text in zip(TRUE, SONGS[lang]):
                for k, w in enumerate(fold_for_compare(text, lang).split()):
                    words.append((w.lower(), t + k * 0.4, t + k * 0.4 + 0.3))
            out = estimate_from_whisper_words(lyr(lang, TRUE), words, lang)
            self.assertEqual(out, TRUE)


if __name__ == "__main__":
    unittest.main()
