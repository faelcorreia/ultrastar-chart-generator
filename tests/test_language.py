import unittest

from src.core.language import LanguageError, normalize_code, resolve_language
from src.core.text_norm import fold_for_compare, normalize_text
from src.core.syllables_language import LanguageMixin


class ResolveLanguageTest(unittest.TestCase):
    def test_explicit_wins(self):
        for given, want in (("pt-BR", "pt"), ("EN", "en"), ("es_419", "es")):
            self.assertEqual(resolve_language(given), want)

    def test_unsupported_explicit_fails(self):
        with self.assertRaises(LanguageError):
            resolve_language("fr")

    def test_tags_before_detection(self):
        notes = []
        got = resolve_language(None, "a.mp3", notes=notes,
                               tag_reader=lambda p: "es",
                               detector=lambda p: self.fail("no detect"))
        self.assertEqual(got, "es")
        self.assertIn("tags", notes[0])

    def test_detection_is_flagged(self):
        notes = []
        got = resolve_language(None, "a.mp3", notes=notes,
                               tag_reader=lambda p: None,
                               detector=lambda p: "en")
        self.assertEqual(got, "en")
        self.assertIn("DETECTED", notes[0])

    def test_impossible_fails_clearly(self):
        with self.assertRaises(LanguageError) as ctx:
            resolve_language(None, "a.mp3", tag_reader=lambda p: None,
                             detector=lambda p: None, label="'A - B'")
        self.assertIn("lyrics.txt", str(ctx.exception))
        with self.assertRaises(LanguageError):
            resolve_language(None, None)

    def test_detector_error_fails(self):
        def boom(p):
            raise RuntimeError("no model")
        with self.assertRaises(LanguageError):
            resolve_language(None, "a.mp3", tag_reader=lambda p: None,
                             detector=boom)

    def test_codes(self):
        self.assertEqual(normalize_code("por"), "pt")
        self.assertIsNone(normalize_code("xx"))

    def test_syllables_mixin_has_no_pt_fallback(self):
        with self.assertRaises(ValueError):
            LanguageMixin()._resolve_language(None)
        with self.assertRaises(ValueError):
            LanguageMixin()._pyphen_language(None)


class TextNormTest(unittest.TestCase):
    def test_english_apostrophes(self):
        self.assertEqual(normalize_text("Don’t stop ‘believing’", "en"),
                         "Don't stop believing'")
        self.assertEqual(normalize_text("I can't  go", "en-US"), "I can't go")

    def test_portuguese_accents_kept(self):
        self.assertEqual(normalize_text("coração  não", "pt"), "coração não")
        self.assertEqual(normalize_text("d'água", "pt-BR"), "dágua")
        self.assertEqual(fold_for_compare("Coração, Não!", "pt"), "coracao nao")

    def test_spanish_enye(self):
        decomposed = "mañana"
        self.assertEqual(normalize_text(decomposed, "es"), "mañana")
        self.assertEqual(fold_for_compare("¿Mañana?", "es"), "mañana")
        self.assertNotEqual(fold_for_compare("año", "es"),
                            fold_for_compare("ano", "es"))

    def test_unsupported(self):
        with self.assertRaises(ValueError):
            normalize_text("x", "fr")


if __name__ == "__main__":
    unittest.main()
