"""Tesseract CLI adapter for the CDG reader (numpy RGB image -> text).

The CDG reader injects OCR so it stays testable without tesseract. This is
the real implementation: it writes a PPM to tesseract's stdin and reads the
text from stdout, so no Pillow/pytesseract dependency is needed. The image
arrives already cropped and scaled by the reader (CDG glyphs are 12 px tall).

Language codes are Tesseract's (ISO 639-2/3), not the project's (ISO 639-1).
A missing traineddata file is reported as ``OcrError`` with the package name
so the caller can tell the user what to install.
"""
import shutil
import subprocess
from typing import Callable, Dict, Optional

import numpy as np

# Project language -> Tesseract traineddata. Extend as needed; the project
# only promises pt/en/es, but karaoke collections are multilingual.
TESSERACT_LANGUAGE: Dict[str, str] = {
    "pt": "por", "pt-br": "por", "pt-pt": "por",
    "en": "eng", "en-us": "eng", "en-gb": "eng",
    "es": "spa", "es-419": "spa", "es-es": "spa",
    "fr": "fra", "de": "deu", "it": "ita", "nl": "nld",
    "sv": "swe", "da": "dan", "no": "nor", "fi": "fin",
    "pl": "pol", "ru": "rus", "ja": "jpn", "ko": "kor",
    "zh": "chi_sim", "tr": "tur", "el": "ell", "he": "heb",
    "ar": "ara", "hi": "hin", "id": "ind", "vi": "vie",
}

_PACKAGE = {
    "eng": "tesseract-data-eng", "por": "tesseract-data-por",
    "spa": "tesseract-data-spa", "fra": "tesseract-data-fra",
    "deu": "tesseract-data-deu", "ita": "tesseract-data-ita",
}


class OcrError(RuntimeError):
    pass


def tesseract_code(language: Optional[str]) -> str:
    return TESSERACT_LANGUAGE.get((language or "en").lower().strip(), "eng")


def _ppm(image: np.ndarray) -> bytes:
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    header = f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode()
    return header + image.tobytes()


def make_ocr(language: Optional[str] = None, psm: int = 7
             ) -> Callable[[np.ndarray], str]:
    """Return `ocr(image) -> str` bound to one Tesseract language."""
    code = tesseract_code(language)
    if shutil.which("tesseract") is None:
        raise OcrError("tesseract not found in PATH")

    def ocr(image: np.ndarray) -> str:
        try:
            run = subprocess.run(
                ["tesseract", "stdin", "stdout", "--psm", str(psm),
                 "-l", code],
                input=_ppm(image), capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            raise OcrError(f"tesseract failed: {exc}") from exc
        if run.returncode:
            detail = run.stderr.decode("utf-8", "replace").strip()
            if "Failed loading language" in detail or "tessdata" in detail:
                package = _PACKAGE.get(code, f"tesseract-data-{code}")
                raise OcrError(
                    f"tesseract has no '{code}' data; install {package} "
                    f"(or use --language with an installed one)")
            raise OcrError(f"tesseract failed: {detail or run.returncode}")
        return run.stdout.decode("utf-8", "replace").strip()

    return ocr


_default = None


def tesseract_ocr(image: np.ndarray) -> str:
    """Default (English) for callers that just need a callable."""
    global _default
    if _default is None:
        _default = make_ocr("en")
    return _default(image)
