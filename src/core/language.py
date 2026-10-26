"""Explicit language resolution: list/filename, then tags, then Whisper."""

from pathlib import Path

SUPPORTED = ("en", "pt", "es")
_ALIASES = {"eng": "en", "por": "pt", "spa": "es", "esp": "es", "ptbr": "pt"}


class LanguageError(ValueError):
    pass


def normalize_code(code: str | None) -> str | None:
    """Return 'en'/'pt'/'es' for a code like 'pt-BR', 'eng', or None."""
    if not code:
        return None
    lang = code.strip().lower().replace("_", "-")
    base = lang.split("-", 1)[0]
    base = _ALIASES.get(lang, _ALIASES.get(base, base))
    return base if base in SUPPORTED else None


def tag_language(audio_path) -> str | None:
    """Language from the audio file tags (ffprobe), when present."""
    from src.readers.metadata import ffprobe, _tags
    info = ffprobe(Path(audio_path))
    if not info:
        return None
    return normalize_code(_tags(info).get("language"))


def detect_language(audio_path, model_size: str = "small") -> str | None:
    """Detect the language with Whisper (loads the model lazily)."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    _, info = model.transcribe(str(audio_path), language=None, beam_size=1,
                               without_timestamps=True)
    return normalize_code(info.language)


def resolve_language(explicit=None, audio_path=None, *, notes=None,
                     tag_reader=tag_language, detector=detect_language,
                     label="song") -> str:
    """Resolve a language or raise LanguageError (never defaults).

    Order: explicit (list/filename), file tags, Whisper detection. Detection
    is appended to ``notes`` so the report flags it.
    """
    if explicit:
        code = normalize_code(explicit)
        if code is None:
            raise LanguageError(
                f"Unsupported language '{explicit}' for {label}. "
                f"Supported: {', '.join(SUPPORTED)}."
            )
        return code
    if audio_path is not None:
        code = tag_reader(audio_path)
        if code:
            if notes is not None:
                notes.append(f"language '{code}' taken from file tags")
            return code
        try:
            code = detector(audio_path)
        except Exception as exc:  # model missing, decode failure, ...
            raise LanguageError(
                f"Could not determine the language of {label}: detection "
                f"failed ({exc}). Name the lyrics file "
                "'Artist - Title - <en|pt|es> lyrics.txt'."
            ) from exc
        if code:
            if notes is not None:
                notes.append(
                    f"language '{code}' DETECTED by Whisper (not given); "
                    "please verify"
                )
            return code
    raise LanguageError(
        f"Could not determine the language of {label}. Name the lyrics file "
        "'Artist - Title - <en|pt|es> lyrics.txt'."
    )
