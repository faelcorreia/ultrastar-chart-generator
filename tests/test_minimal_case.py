import unittest

from src.core.text_norm import minimal_case


class MinimalCaseTest(unittest.TestCase):
    def test_en(self):
        self.assertEqual(minimal_case("LET IT BE, Mary", "en"), "Let it be, mary")
        self.assertEqual(
            minimal_case("and i know i'm fine, I’ll stay", "en"),
            "And I know I'm fine, I’ll stay")
        self.assertEqual(minimal_case("i'd say it's a Wish, i've", "en"),
                         "I'd say it's a wish, I've")
        self.assertEqual(minimal_case("'cause Rock 'n' Roll", "en"),
                         "'Cause rock 'n' roll")

    def test_en_keeps_words_starting_with_i(self):
        self.assertEqual(minimal_case("In IT is ill", "en"), "In it is ill")

    def test_pt(self):
        self.assertEqual(minimal_case("FORÇA Estranha, Coração", "pt"),
                         "Força estranha, coração")
        self.assertEqual(minimal_case("é a Ação d'água", "pt"),
                         "É a ação d'água")
        # "i" is not special outside English
        self.assertEqual(minimal_case("eu i você", "pt"), "Eu i você")

    def test_es(self):
        self.assertEqual(minimal_case("¿Dónde Está El NIÑO?", "es"),
                         "¿Dónde está el niño?")
        self.assertEqual(minimal_case("ñandú Canción", "es"), "Ñandú canción")

    def test_spelling_untouched(self):
        self.assertEqual(minimal_case("Gonna wanna lovin'", "en"),
                         "Gonna wanna lovin'")

    def test_no_letters(self):
        self.assertEqual(minimal_case("...", "en"), "...")


if __name__ == "__main__":
    unittest.main()
