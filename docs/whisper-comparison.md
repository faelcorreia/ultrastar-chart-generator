# Whisper model comparison (small / medium)

Report only. The pipeline default model is unchanged; decide after reading
the numbers.

## How to run

1. Pick one short song per language (en, pt, es): an audio file (ideally the
   separated vocals) and a plain-text reference lyrics file.
2. Write a `manifest.json` next to them:

   ```json
   [
     {"language": "en", "audio": "en.mp3", "reference": "en.txt"},
     {"language": "pt", "audio": "pt.mp3", "reference": "pt.txt"},
     {"language": "es", "audio": "es.mp3", "reference": "es.txt"}
   ]
   ```
3. Run (models are downloaded by faster-whisper on first use):

   ```
   .venv/bin/python scripts/compare_whisper_models.py manifest.json \
       --models small medium --out docs/whisper-results.md
   ```

The script prints and optionally saves a table with word error rate (WER),
transcription time and model load time per model and language. Text is
normalized with `src/core/text_norm.py` (case, accents, punctuation) before
scoring, so WER reflects wrong words rather than spelling variants.

## Reading the results

Choose the cheapest model whose WER is acceptable in all three languages;
compare time on the same machine (CPU, int8).

## Results

Measured 2026-10-27 (CPU 8 cores, int8, full mix, language pt, reference =
LRCLIB plain lyrics; script defaults, no VAD):

| model | audio | WER | time (s) |
|---|---|---|---|
| small | Tribalistas - Ja sei namorar (dense pt) | 0.973 | 37.2 |
| small | Jota Quest - O vento | 0.929 | 43.0 |
| medium | Tribalistas - Ja sei namorar (dense pt) | 0.973 | 245.4 |
| medium | Jota Quest - O vento | 0.980 | 294.7 |

Both models fail on sung Portuguese (WER near 1.0, repetition loops); medium
is 6-7x slower with no gain here. The separated vocal stems were worse: the
Jota Quest stem is silent (mean -84 dB, 0 words) and Tribalistas looped one
sentence. Decision (I8) stays: medium only, lyrics strongly recommended.
