# Whisper model comparison (small / medium / large)

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
       --models small medium large-v3 --out docs/whisper-results.md
   ```

The script prints and optionally saves a table with word error rate (WER),
transcription time and model load time per model and language. Text is
normalized with `src/core/text_norm.py` (case, accents, punctuation) before
scoring, so WER reflects wrong words rather than spelling variants.

## Reading the results

Choose the cheapest model whose WER is acceptable in all three languages;
compare time on the same machine (CPU, int8).

## Results

_Not measured yet._
