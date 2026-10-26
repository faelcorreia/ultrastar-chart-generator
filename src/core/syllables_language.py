"""Language resolution and lyrics tokenisation."""

import re

import pyphen


class LanguageMixin:
    _LANG_MAP = {
        "pt": "pt_BR",
        "pt-br": "pt_BR",
        "en": "en_US",
        "en-us": "en_US",
        "es": "es",
        "es-es": "es",
        "es-419": "es",
    }

    # language -> (acoustic model, pronunciation dictionary, G2P model)
    _LANGUAGE_MODELS = {
        "pt": ("portuguese_mfa", "portuguese_brazil_mfa", "portuguese_brazil_mfa"),
        "en": ("english_mfa", "english_us_mfa", "english_us_mfa"),
        "es": ("spanish_mfa", "spanish_latin_america_mfa",
               "spanish_latin_america_mfa"),
    }

    def _resolve_language(self, language: str | None) -> str:
        """Return a configured language code (base language) for ``language``."""
        if not language:
            return "pt"

        lang = language.strip().lower().replace("_", "-")
        if lang in self._LANGUAGE_MODELS:
            return lang

        base = lang.split("-", 1)[0]
        if base in self._LANGUAGE_MODELS:
            return base

        supported = ", ".join(sorted(self._LANGUAGE_MODELS))
        raise ValueError(
            f"No MFA models configured for language '{language}'. "
            f"Supported languages: {supported}. For other languages, pass "
            "acoustic_model, dictionary and g2p to SyllablesTranscriber."
        )

    def _models_for(self, language: str | None) -> tuple[str, str, str | None]:
        """Return the MFA models to use for a song.

        Explicit models passed to the constructor take precedence over the
        per-language defaults.
        """
        if self.acoustic_model and self.dictionary:
            return self.acoustic_model, self.dictionary, self.g2p

        acoustic, dictionary, g2p = self._LANGUAGE_MODELS[
            self._resolve_language(language)
        ]
        return (
            self.acoustic_model or acoustic,
            self.dictionary or dictionary,
            self.g2p or g2p,
        )

    def _pyphen_language(self, language: str | None) -> str:
        lang = (language or "pt").strip().lower().replace("_", "-")
        if lang in self._LANG_MAP:
            return self._LANG_MAP[lang]

        # Allow custom models in other languages without breaking hyphenation,
        # as long as pyphen has a matching dictionary.
        base = lang.split("-", 1)[0]
        for available in pyphen.LANGUAGES:
            if available.lower() == base or available.lower().startswith(
                base + "_"
            ):
                return available
        return "pt_BR"

    # --------------------------------------------------------------- lyrics
    def _lyrics_tokens(self, lyrics: str) -> list[dict]:
        """Ordered word tokens of the lyrics, keeping the original spelling.

        ``display`` holds the word exactly as written (case, accents and
        punctuation), while ``mfa`` holds the normalized form MFA aligns
        (lowercase, without punctuation or digits). Hyphens split compound
        words, matching the text sent to MFA.
        """
        tokens: list[dict] = []
        lines = [ln.strip()
                 for ln in lyrics.strip().splitlines() if ln.strip()]

        for line_index, line in enumerate(lines):
            line_tokens = []
            for raw_word in line.replace("-", " ").split():
                mfa_word = re.sub(r"[^\w\s']", "", raw_word)
                mfa_word = re.sub(r"\d+", "", mfa_word).lower()
                # Punctuation-only tokens are ignored by MFA as well.
                if not mfa_word:
                    continue
                line_tokens.append({"display": raw_word, "mfa": mfa_word})

            for i, token in enumerate(line_tokens):
                token["line_break"] = i == len(line_tokens) - 1
                token["line"] = line_index
            tokens.extend(line_tokens)

        return tokens

    def _clean_lyrics(self, lyrics: str) -> str:
        """Normalize the lyrics exactly as MFA will see them."""
        return " ".join(token["mfa"] for token in self._lyrics_tokens(lyrics))

    def _flatten_lyrics(self, lyrics: str) -> list[dict]:
        """Ordered tokens of the original lyrics, marking the last word of
        each line (used to decide where to break lines in UltraStar)."""
        return self._lyrics_tokens(lyrics)

    # ------------------------------------------------------------------- MFA
