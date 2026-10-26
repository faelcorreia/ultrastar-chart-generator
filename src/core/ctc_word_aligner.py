"""Word-level forced alignment with a CTC model (ctc-forced-aligner, MMS).

Unlike MFA (an HMM-GMM speech aligner), a CTC aligner tolerates long
instrumental gaps and sustained vowels, which is what sung vocals look like.
It only returns word boundaries, so ``to_textgrid`` writes them as a
TextGrid whose phone tier carries one interval per letter; the rest of the
pipeline then treats it exactly like the MFA output.
"""
import os
from pathlib import Path

import textgrid
from ctc_forced_aligner import (
    MODEL_URL,
    Tokenizer,
    ensure_onnx_model,
    generate_emissions,
    get_alignments,
    get_spans,
    load_audio,
    postprocess_results,
    preprocess_text,
)
import onnxruntime

# ISO 639-3 codes understood by ctc-forced-aligner's text normalizer.
_ISO3 = {"en": "eng", "pt": "por", "es": "spa"}

_MODEL_PATH = Path(os.environ.get(
    "CTC_ALIGNER_MODEL",
    Path.home() / ".cache" / "ctc_forced_aligner" / "model.onnx",
))


def _iso3(language: str) -> str:
    base = language.lower().split("-")[0].split("_")[0]
    return _ISO3.get(base, base)


def align_words(
    audio_path: Path, words: list[str], language: str
) -> list[tuple[float, float]]:
    """Return ``(start, end)`` in seconds for each word, in lyric order."""
    ensure_onnx_model(str(_MODEL_PATH), MODEL_URL)
    session = onnxruntime.InferenceSession(str(_MODEL_PATH))
    tokenizer = Tokenizer()

    waveform = load_audio(str(audio_path))
    emissions, stride = generate_emissions(session, waveform, batch_size=4)

    tokens_starred, text_starred = preprocess_text(
        " ".join(words), romanize=True, language=_iso3(language),
        split_size="word", star_frequency="segment",
    )
    segments, scores, blank = get_alignments(
        emissions, tokens_starred, tokenizer
    )
    spans = get_spans(tokens_starred, segments, blank)
    results = postprocess_results(text_starred, spans, stride, scores)

    if len(results) != len(words):
        raise RuntimeError(
            f"CTC aligned {len(results)} word(s) for {len(words)} lyric word(s)"
        )
    return [(r["start"], r["end"]) for r in results]


def to_textgrid(
    words: list[str],
    bounds: list[tuple[float, float]],
    duration: float,
    output_path: Path,
) -> Path:
    """Write ``words`` and ``phones`` tiers (letters as phones)."""
    words_tier = textgrid.IntervalTier(name="words", minTime=0, maxTime=duration)
    phones_tier = textgrid.IntervalTier(name="phones", minTime=0, maxTime=duration)

    for word, (start, end) in zip(words, bounds):
        end = max(end, start + 0.001)
        words_tier.add(start, end, word)
        step = (end - start) / len(word)
        for i, letter in enumerate(word):
            phones_tier.add(start + i * step, start + (i + 1) * step, letter)

    tg = textgrid.TextGrid(minTime=0, maxTime=duration)
    tg.append(words_tier)
    tg.append(phones_tier)
    tg.write(str(output_path))
    return output_path
