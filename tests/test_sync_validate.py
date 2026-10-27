import unittest

from src.core.text_norm import fold_for_compare

from src.model.lyrics import Line, Lyrics
from src.stages.sync_validate import (
    DRIFT, LINES, NONE, OFFSET, ROBUST, SHIFTED, UNCERTAIN, OK,
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
    def test_deaf_whisper_falls_back_to_ctc(self):
        for lang in SONGS:
            r = run(lang, [t + 3.0 for t in TRUE], w=[None] * 6, c=TRUE)
            self.assertTrue(r.ctc_only)
            self.assertEqual(r.status, "review")
            for line, t in zip(r.lyrics.lines, TRUE):
                self.assertAlmostEqual(line.start, t, places=3)

    def test_single_anchor_does_not_shift_whole_song(self):
        for lang in SONGS:
            none = [None] * 6
            w = [TRUE[0] - 6.0] + none[1:]
            r = run(lang, TRUE, w=w, c=w)
            self.assertNotIn(r.correction.kind, (OFFSET, DRIFT))
            self.assertEqual([l.start for l in r.lyrics.lines][1:], TRUE[1:])

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

    def _heard(self, lang, mangle):
        words = []
        for i, (t, text) in enumerate(zip(TRUE, SONGS[lang])):
            toks = fold_for_compare(text, lang).split()
            if i in mangle:
                toks = mangle[i](toks)
            for k, w in enumerate(toks):
                words.append((w, t + k * 0.4, t + k * 0.4 + 0.3))
        return words

    def test_fuzzy_match_finds_misheard_lines(self):
        # Whisper mishears a letter or two per word, drops or splits words.
        def typo(toks):
            return [w[:-1] + "x" if len(w) > 3 else w for w in toks]

        def split_merge(toks):
            return [toks[0] + toks[1]] + toks[2:]
        for lang in SONGS:
            words = self._heard(lang, {0: typo, 2: split_merge, 4: typo})
            exact = estimate_from_whisper_words(
                lyr(lang, TRUE), words, lang, cutoff=1.0)
            fuzzy = estimate_from_whisper_words(lyr(lang, TRUE), words, lang)
            self.assertIsNone(exact[0])
            self.assertEqual(fuzzy, TRUE)

    def test_fuzzy_does_not_match_unrelated_text(self):
        for lang in SONGS:
            words = [(w, 5.0 + k, 5.3 + k) for k, w in enumerate(
                "zzz qqq www kkk jjj hhh".split())]
            out = estimate_from_whisper_words(lyr(lang, TRUE), words, lang)
            self.assertEqual(out, [None] * 6)

    def test_fuzzy_keeps_line_order(self):
        for lang in SONGS:
            words = self._heard(lang, {})
            out = estimate_from_whisper_words(lyr(lang, TRUE), words, lang)
            self.assertEqual(out, sorted(out))

    # I9: CTC alone is garbage on some lines (quiet intro in the stem); the
    # source's own relative timing must survive there.
    def test_ctc_only_garbage_lines_keep_the_model(self):
        for lang in SONGS:
            src = [t + 2.4 for t in TRUE]
            ctc = [TRUE[0] - 9.0, TRUE[1] - 14.0, TRUE[2] + 0.6,
                   TRUE[3] - 0.4, TRUE[4] + 0.3, TRUE[5] + 0.5]
            r = run(lang, src, w=[None] * 6, c=ctc)
            self.assertTrue(r.ctc_only)
            self.assertEqual(r.correction.kind, ROBUST)
            self.assertEqual(r.status, "review")
            starts = [l.start for l in r.lyrics.lines]
            for i in (0, 1):  # garbage estimates are not used
                self.assertAlmostEqual(starts[i], TRUE[i], delta=0.6)
            for i in (2, 3, 4, 5):  # supported estimates are used
                self.assertAlmostEqual(starts[i], ctc[i])
            self.assertEqual(r.correction.lines, [2, 3, 4, 5])
            # the line keeps its length when it moves
            self.assertAlmostEqual(r.lyrics.lines[0].end - starts[0], 5.0)

    def test_ctc_only_robust_drift(self):
        for lang in SONGS:
            src = [t * 1.08 for t in TRUE]
            ctc = [TRUE[0] + 8.0, TRUE[1] + 9.0] + TRUE[2:]
            r = run(lang, src, w=[None] * 6, c=ctc)
            self.assertEqual(r.correction.kind, ROBUST)
            self.assertAlmostEqual(r.correction.scale, 1 / 1.08, places=2)
            starts = [l.start for l in r.lyrics.lines]
            for i in range(6):
                self.assertAlmostEqual(starts[i], TRUE[i], delta=0.3)

    def test_ctc_only_without_consensus_falls_back_to_lines(self):
        for lang in SONGS:
            ctc = [t + d for t, d in zip(TRUE, (-9, 7, -4, 12, -15, 5))]
            r = run(lang, TRUE, w=[None] * 6, c=ctc)
            self.assertNotEqual(r.correction.kind, ROBUST)

    def test_whisper_and_ctc_agreeing_does_not_use_the_robust_model(self):
        for lang in SONGS:
            src = list(TRUE)
            src[1] += 4
            r = run(lang, src)
            self.assertEqual(r.correction.kind, LINES)


if __name__ == "__main__":
    unittest.main()
