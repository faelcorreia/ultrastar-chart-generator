"""Per-language text normalization for lyrics (en, pt, es)."""

import re
import unicodedata

SUPPORTED = ("en", "pt", "es")

# Typographic apostrophes and quotes that appear in pasted lyrics.
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'",
                              "`": "'", "´": "'"})
_DASHES = str.maketrans({"–": "-", "—": "-"})


def base_language(language: str) -> str:
    return language.strip().lower().replace("_", "-").split("-", 1)[0]


def normalize_text(text: str, language: str) -> str:
    """Normalize ``text`` for ``language``.

    Always: NFC form (so accents and ``n`` + tilde compose to ``ñ``),
    typographic apostrophes/dashes straightened, whitespace collapsed.
    English keeps word-internal apostrophes (don't, rock 'n' roll) and drops
    quote marks around words; pt/es drop apostrophes entirely, keeping accents
    and ``ñ``/``ç`` intact.
    """
    lang = base_language(language)
    if lang not in SUPPORTED:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported: {', '.join(SUPPORTED)}."
        )
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_APOSTROPHES).translate(_DASHES)
    if lang == "en":
        # Strip apostrophes used as quotes: ' at word start/end, except the
        # elisions ('til, 'cause, 'n', singin') which are one-sided on purpose.
        text = re.sub(r"(?<![\w])'(?=\w+'(?![\w]))", "", text)
    else:
        text = text.replace("'", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def fold_for_compare(text: str, language: str) -> str:
    """Lowercase, accent-free, punctuation-free form for matching/WER.

    ``ñ`` is kept distinct from ``n`` in Spanish (it changes the word).
    """
    lang = base_language(language)
    text = normalize_text(text, lang).lower()
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch):
            # keep the tilde on n (ñ); drop other accents
            if ch == "̃" and out and out[-1] == "n" and lang == "es":
                out.append(ch)
            continue
        out.append(ch)
    text = unicodedata.normalize("NFC", "".join(out))
    keep = r"[^\w\s']" if lang == "en" else r"[^\w\s]"
    text = re.sub(keep, " ", text)
    return re.sub(r"\s+", " ", text).strip()


_EN_I = re.compile(r"(?<![\w'’])i(?=$|[^\w'’]|['’](?:m|ll|d|ve)(?![\w]))",
                   re.IGNORECASE)


def minimal_case(line: str, language: str) -> str:
    """I11: uppercase only the first letter of the verse, lowercase the rest.

    English keeps ``I`` and its contractions (I'm, I'll, I'd, I've) capital.
    Spelling and grammar are never touched (artistic spelling is valid), and
    only case changes, so accents, ``ñ`` and apostrophes pass through.
    """
    lang = base_language(language)
    text = line.lower()
    if lang == "en":
        text = _EN_I.sub("I", text)
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text
