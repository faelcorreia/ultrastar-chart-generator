import os
import tempfile
import unittest

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from src.core.syllables_transcriber import SyllablesTranscriber  # noqa: E402


class LanguageModelsTest(unittest.TestCase):
    def setUp(self):
        self.transcriber = SyllablesTranscriber({})

    def test_portuguese_models(self):
        self.assertEqual(
            self.transcriber._models_for("pt"),
            ("portuguese_mfa", "portuguese_brazil_mfa",
             "portuguese_brazil_mfa"),
        )

    def test_english_models(self):
        self.assertEqual(
            self.transcriber._models_for("en-US"),
            ("english_mfa", "english_us_mfa", "english_us_mfa"),
        )

    def test_spanish_models(self):
        self.assertEqual(
            self.transcriber._models_for("es-419"),
            ("spanish_mfa", "spanish_latin_america_mfa",
             "spanish_latin_america_mfa"),
        )

    def test_explicit_models_override_language(self):
        transcriber = SyllablesTranscriber(
            {}, acoustic_model="custom_acoustic",
            dictionary="custom_dict", g2p="custom_g2p",
        )
        self.assertEqual(
            transcriber._models_for("fr"),
            ("custom_acoustic", "custom_dict", "custom_g2p"),
        )

    def test_unsupported_language_raises(self):
        with self.assertRaises(ValueError):
            self.transcriber._models_for("fr")

    def test_pyphen_language(self):
        self.assertEqual(self.transcriber._pyphen_language("pt-BR"), "pt_BR")
        self.assertEqual(self.transcriber._pyphen_language("en"), "en_US")
        self.assertEqual(self.transcriber._pyphen_language("es"), "es")


class LyricsTokensTest(unittest.TestCase):
    def setUp(self):
        self.transcriber = SyllablesTranscriber({})

    def test_clean_lyrics_matches_normalization_sent_to_mfa(self):
        lyrics = "Noite clara, flores-do-campo!\nAlguém ainda quer dançar?"

        self.assertEqual(
            self.transcriber._clean_lyrics(lyrics),
            "noite clara flores do campo alguém ainda quer dançar",
        )

    def test_flatten_lyrics_keeps_original_spelling(self):
        lyrics = "Noite clara, flores-do-campo\nAlguém ainda quer dançar?"

        tokens = self.transcriber._flatten_lyrics(lyrics)

        self.assertEqual(
            [token["display"] for token in tokens],
            ["Noite", "clara,", "flores", "do", "campo",
             "Alguém", "ainda", "quer", "dançar?"],
        )
        self.assertFalse(tokens[3]["line_break"])
        self.assertTrue(tokens[4]["line_break"])
        self.assertTrue(tokens[-1]["line_break"])

    def test_display_word_parts_preserve_case_and_punctuation(self):
        import pyphen
        dic = pyphen.Pyphen(lang="pt_BR")

        self.assertEqual(
            self.transcriber._display_word_parts(dic, "cidade", "Cidade"),
            ["Ci", "da", "de"],
        )
        self.assertEqual(
            self.transcriber._display_word_parts(dic, "cidade", "cidade?"),
            ["ci", "da", "de?"],
        )

    def test_map_parts_skips_punctuation_not_in_mfa_word(self):
        self.assertEqual(
            self.transcriber._map_parts(["can", "tar"], "cantar", "can*tar"),
            ["can*", "tar"],
        )


class WhisperAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.transcriber = SyllablesTranscriber({})

    def test_aligns_words_and_leaves_unrecognized_ones_empty(self):
        tokens = self.transcriber._lyrics_tokens(
            "Noite clara\nAlguém vem?"
        )
        words = [
            {"word": "Noite", "start": 1.0, "end": 2.0},
            {"word": "Alguém", "start": 4.0, "end": 5.0},
            {"word": "vem", "start": 5.0, "end": 6.0},
        ]

        bounds = self.transcriber._align_whisper_words(words, tokens)

        self.assertEqual(bounds[0], (1.0, 2.0))
        self.assertIsNone(bounds[1])
        self.assertEqual(bounds[2], (4.0, 5.0))
        self.assertEqual(bounds[3], (5.0, 6.0))

    def test_distributes_one_whisper_word_across_several_tokens(self):
        tokens = self.transcriber._lyrics_tokens("passatempo")
        words = [
            {"word": "passa", "start": 2.0, "end": 2.5},
            {"word": "tempo", "start": 2.5, "end": 3.0},
        ]

        bounds = self.transcriber._align_whisper_words(words, tokens)

        self.assertEqual(bounds, [(2.0, 3.0)])

    def test_drifted_line_requires_a_severe_gap(self):
        from types import SimpleNamespace

        tokens = [
            {"line": 0, "display": "a"},
            {"line": 0, "display": "b"},
            {"line": 1, "display": "c"},
            {"line": 1, "display": "d"},
        ]
        word_intervals = [
            SimpleNamespace(minTime=0.0, maxTime=1.0),
            SimpleNamespace(minTime=1.0, maxTime=2.0),
            SimpleNamespace(minTime=10.0, maxTime=10.1),
            SimpleNamespace(minTime=10.1, maxTime=10.2),
        ]
        whisper_bounds = [(0.0, 1.0), (1.0, 2.0), (3.0, 3.1), (3.1, 3.2)]

        drifted = self.transcriber._drifted_lines(
            tokens, word_intervals, whisper_bounds
        )

        self.assertEqual(drifted, {1})

    def test_ignores_whisper_line_with_internal_gap(self):
        from types import SimpleNamespace

        tokens = [
            {"line": 0, "display": "a"},
            {"line": 0, "display": "b"},
            {"line": 1, "display": "c"},
            {"line": 1, "display": "d"},
        ]
        word_intervals = [
            SimpleNamespace(minTime=0.0, maxTime=0.5),
            SimpleNamespace(minTime=0.5, maxTime=1.0),
            SimpleNamespace(minTime=10.0, maxTime=10.5),
            SimpleNamespace(minTime=10.5, maxTime=11.0),
        ]
        # Whisper glued the first word to the previous phrase and left a big
        # gap inside the line, so its timings are not trustworthy.
        whisper_bounds = [
            (0.0, 0.5),
            (0.5, 1.0),
            (5.0, 5.1),
            (9.0, 9.5),
        ]

        drifted = self.transcriber._drifted_lines(
            tokens, word_intervals, whisper_bounds
        )

        self.assertEqual(drifted, set())

    def test_large_shift_without_gap_is_not_corrected(self):
        from types import SimpleNamespace

        tokens = [
            {"line": 0, "display": "a"},
            {"line": 0, "display": "b"},
        ]
        word_intervals = [
            SimpleNamespace(minTime=10.0, maxTime=10.5),
            SimpleNamespace(minTime=10.5, maxTime=11.0),
        ]
        whisper_bounds = [(0.0, 0.5), (0.5, 1.0)]

        drifted = self.transcriber._drifted_lines(
            tokens, word_intervals, whisper_bounds
        )

        self.assertEqual(drifted, set())


class VowelDetectionTest(unittest.TestCase):
    def setUp(self):
        self.transcriber = SyllablesTranscriber({})

    def test_ipa_vowels(self):
        for phone in ("a", "ẽ", "ɐ", "ɔ", "aj", "əw", "ɚ"):
            self.assertTrue(
                self.transcriber._is_vowel_phone(phone), phone
            )

    def test_ipa_consonants(self):
        for phone in ("s", "x", "ɾ", "m", "w̃", "ʒ"):
            self.assertFalse(
                self.transcriber._is_vowel_phone(phone), phone
            )

    def test_arpabet(self):
        self.assertTrue(self.transcriber._is_vowel_phone("AA1"))
        self.assertTrue(self.transcriber._is_vowel_phone("IY0"))
        self.assertFalse(self.transcriber._is_vowel_phone("B"))
        self.assertFalse(self.transcriber._is_vowel_phone("Y"))


class SyllableSplittingTest(unittest.TestCase):
    def setUp(self):
        self.transcriber = SyllablesTranscriber({})

    def test_groups_phones_by_vowel_nucleus(self):
        phones = [
            ("v", 0.0, 0.05),
            ("ẽ", 0.05, 0.20),
            ("t", 0.20, 0.25),
            ("u", 0.25, 0.40),
        ]
        syllables = self.transcriber._group_phones_into_syllables(phones)

        self.assertEqual(len(syllables), 2)
        self.assertEqual(syllables[0]["syllable"], "vẽ")
        self.assertEqual(syllables[1]["syllable"], "tu")

    def test_proportional_split_snaps_to_phone_boundaries(self):
        phones = [
            ("k", 0.0, 0.05),
            ("a", 0.05, 0.20),
            ("m", 0.20, 0.25),
            ("i", 0.25, 0.40),
            ("n", 0.40, 0.45),
            ("u", 0.45, 0.60),
        ]
        syllables = self.transcriber._split_word_proportionally(
            phones, ["ca", "mi", "nu"]
        )

        self.assertEqual(len(syllables), 3)
        self.assertEqual(syllables[0]["start"], 0.0)
        self.assertEqual(syllables[-1]["end"], 0.60)
        for previous, current in zip(syllables, syllables[1:]):
            self.assertEqual(previous["end"], current["start"])

    def test_word_parts_uses_pyphen(self):
        import pyphen
        dic = pyphen.Pyphen(lang="pt_BR")
        self.assertEqual(
            self.transcriber._word_parts(dic, "janelas"),
            ["ja", "ne", "las"],
        )


if __name__ == "__main__":
    unittest.main()
