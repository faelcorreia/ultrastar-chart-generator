"""I7: clean artist/title strings and build LRCLIB lookup variants.

Variants go from most specific (the name as given) to loosest (no
parentheses, no featuring credit). Only the first lookup that finds lyrics
is used, so the exact name always wins.
"""
import re
from typing import List, Tuple

# "12 - ", "12. ", "12) ", "12 " (only when followed by a letter), "12-"
_NUM_PREFIX = re.compile(r"^\s*\d{1,3}\s*(?:[-–.)_]\s*|\s+(?=[^\W\d_]))")
_BRACKETS = re.compile(r"\s*\[[^\]]*\]")
_PARENS = re.compile(r"\s*\([^)]*\)")
_NOISE_WORDS = (
    "official", "oficial", "live", "ao vivo", "en vivo", "en directo",
    "remaster", "remastered", "remasterizado", "lyric", "lyrics", "letra",
    "video", "vídeo", "videoclip", "clipe", "clip", "audio", "áudio",
    "hd", "hq", "4k", "version", "versão", "versión", "mono", "stereo",
    "karaoke", "explicit", "bonus", "single", "radio edit", "mv")
_NOISE_PAREN = re.compile(
    r"\s*[\(\[][^)\]]*\b(?:" + "|".join(re.escape(w) for w in _NOISE_WORDS)
    + r")\b[^)\]]*[\)\]]", re.IGNORECASE)
_FEAT = re.compile(r"\s*[\(\[]?\s*\b(?:feat|ft|featuring|part|com|con)\.?\s+"
                   r".*$", re.IGNORECASE)
_FEAT_STRICT = re.compile(r"\s*[\(\[]?\s*\b(?:feat|ft|featuring)\.?\s+.*$",
                          re.IGNORECASE)
_SPACES = re.compile(r"\s{2,}")


def strip_number_prefix(text: str) -> str:
    return _NUM_PREFIX.sub("", text or "", count=1).strip()


def _tidy(text: str) -> str:
    return _SPACES.sub(" ", text).strip(" -–_.,")


def strip_tags(text: str) -> str:
    """Drop [..] tags and noise parentheses such as (Live) or (Official Video)."""
    text = _BRACKETS.sub("", text or "")
    text = _NOISE_PAREN.sub("", text)
    return _tidy(text)


def clean_name(text: str) -> str:
    return strip_tags(strip_number_prefix(text))


def lookup_variants(artist: str, title: str) -> List[Tuple[str, str]]:
    """(artist, title) pairs to try, most specific first, deduplicated."""
    artist, title = artist or "", title or ""
    c_artist, c_title = clean_name(artist), clean_name(title)
    no_feat_title = _tidy(_FEAT_STRICT.sub("", c_title))
    no_paren_title = _tidy(_PARENS.sub("", no_feat_title))
    no_feat_artist = _tidy(_FEAT_STRICT.sub("", c_artist))
    candidates = [
        (artist, title),
        (c_artist, c_title),
        (c_artist, no_feat_title),
        (no_feat_artist, no_feat_title),
        (no_feat_artist, no_paren_title),
    ]
    seen, out = set(), []
    for a, t in candidates:
        key = (a.casefold(), t.casefold())
        if a and t and key not in seen:
            seen.add(key)
            out.append((a, t))
    return out
