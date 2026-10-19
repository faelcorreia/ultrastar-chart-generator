# UltraStar Chart Generator

Convert a song into a ready-to-use [UltraStar](https://usdx.eu/) file — fully
automated, from audio to synchronized lyrics with pitch detection.

> Compatible with UltraStar Deluxe, UltraStar Play / Melody Mania and any other
> karaoke application that supports the UltraStar `.txt` format.

---

## How It Works

UltraStar Chart Generator takes a raw song and runs it through a pipeline of
audio analysis tools:

1. **[Demucs](https://github.com/facebookresearch/demucs)** — separates the
   vocals from the full track.
2. **[Librosa](https://librosa.org/)** — estimates the tempo and extracts the
   vocal pitch with `pyin`, including a per-frame confidence value.
3. **[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)**
   — force-aligns the supplied lyrics to the isolated vocals, producing
   word- and phone-level timestamps. A G2P model supplies pronunciations for
   out-of-vocabulary words.
4. **[Whisper](https://github.com/SYSTRAN/faster-whisper)** — used as a second
   opinion when MFA compresses sustained words or jumps over long instrumental
   gaps. Those lyric lines are re-timed from Whisper word timestamps.
5. **[Pyphen](https://github.com/Kozea/Pyphen)** — splits words into the
   syllable text displayed on screen.
6. **Post-processing** — the pitch contour is split into stable notes,
   octave-jump errors are corrected, voiced tails that the aligner classified
   as silence are given to the correct syllable, and the notes of each syllable
   are merged into a single note (held breaths stay inside the syllable, with
   no `~` continuation lines).

The result is a complete `.txt` file in UltraStar format, ready to drop into
your game.

---

## Input File Convention

Place your files in the same directory, following this naming pattern:

| File | Pattern | Example |
|------|---------|---------|
| Cover image | `Artist - Song Title - cover.xxx` | `Queen - Bohemian Rhapsody - cover.jpg` |
| Full audio track | `Artist - Song Title - full track.xxx` | `Queen - Bohemian Rhapsody - full track.mp3` |
| Lyrics + language | `Artist - Song Title - lang lyrics.xxx` | `Queen - Bohemian Rhapsody - en lyrics.txt` |

`Artist - Song Title` may contain extra hyphens, as long as the final suffix is
one of the patterns above. The language tag is optional and defaults to `pt`.

### Supported languages

The language tag selects both the hyphenation dictionary and the MFA models:

| Tag | MFA models |
|-----|------------|
| `pt` / `pt-br` | `portuguese_mfa` + `portuguese_brazil_mfa` + G2P |
| `en` / `en-us` | `english_mfa` + `english_us_mfa` + G2P |
| `es` / `es-419` / `es-es` | `spanish_mfa` + `spanish_latin_america_mfa` + G2P |

For another language, pass custom `acoustic_model`, `dictionary` and `g2p`
values to `SyllablesTranscriber`.

---

## Requirements

- [Conda](https://docs.conda.io/) (needed for the Montreal Forced Aligner)
- Python 3.10+
- CUDA-capable GPU recommended (for Demucs)
- `ffmpeg` installed and available in `PATH`
- Docker (optional, for containerized usage)

Copy the example environment file and adjust the input and output directories:

```bash
cp .env.example .env
```

```dotenv
INPUT_DIR=./input
OUTPUT_DIR=./output
```

The Docker runner mounts these same directories into the container.

---

## Installation

### Local

Run the setup script to create the Conda environment and install all
dependencies (Python packages, the Montreal Forced Aligner and the Portuguese,
English and Spanish models):

```bash
bash prepare.sh
```

Then activate the environment and start the program:

```bash
conda activate audio-processor
python app.py
```

### Docker

To run UltraStar Chart Generator in a Docker container:

```bash
bash docker.sh
```

The image bundles a Conda environment with the Montreal Forced Aligner and the
models for the three supported languages, so the first build takes a while and
the resulting image is sizeable.

---

## Output

The program generates one folder per song, containing the full audio, the
instrumental and vocal stems, the cover, and an UltraStar `.txt` file with:

- Song metadata (`#ARTIST`, `#TITLE`, `#MP3`, `#AUDIO`, `#COVER`, ...)
- `#BPM` and `#GAP`
- `#PREVIEWSTART` (in seconds) at the first vocal line
- `#LANGUAGE`
- One note per syllable: held notes and breaths stay inside the syllable
  duration instead of being split into `~` continuation lines
- The original lyrics spelling (capitalization, accents and punctuation) is
  kept in the chart
- Line breaks (`-`) at the end of each lyric line

The `#BPM` value is intentionally scaled by a power of two (usually ×2 or ×4)
so the integer beat grid is fine enough (~30–60 ms) for karaoke timing. The
timing is unchanged; only the resolution of the beat positions improves.
UltraStar Deluxe quadruples the header BPM internally, so the file beat unit is
a subdivision of the musical beat.

The emitted note pitches are semitones relative to C4 (`0 = C4`), as required
by the UltraStar specification. Internally the pipeline uses MIDI note numbers
(`60 = C4`), so the value written to the file is `midi - 60`. Both UltraStar
Deluxe and UltraStar Play convert this back to MIDI (adding 60).

---

## Quality Notes and Limitations

- **MFA is a speech aligner.** Long sustained vowels can be compressed or
  classified as silence by the acoustic model. The formatter gives voiced
  holds to the syllable that was actually sustained, and Whisper is used as a
  second opinion when MFA jumps over a long instrumental gap. Automatic output
  should still be reviewed in an editor such as Yass.
- **The lyrics must match the recording.** Extra repeated lines or sections
  that are not sung in the audio will be squeezed into the remaining time; no
  aligner can recover missing audio.
- **Pitch tracking is heuristic.** `pyin` confidence filtering and the octave
  correction pass fix the common cases; strongly reverberated or overlapped
  vocals may still contain wrong notes.
- **Duets are not generated.** The aligner runs in single-speaker mode and the
  input format has no singer labels. Adding `P1`/`P2` sections requires manual
  editing (or a dedicated input format).
- **Golden notes and freestyle sections are not generated.** They are musical
  authoring decisions rather than automatic transcription output.
- **No tempo map.** The UltraStar format uses a single `#BPM`. Note times are
  derived from the aligned syllable timestamps, so tempo changes do not cause
  drift; the BPM value mainly controls the beat grid resolution.

---

## Running the Tests

The unit tests cover input filename parsing, language/model selection, vowel
detection, syllable splitting, pitch segmentation, octave correction and
unvoiced-syllable filling. They do not download models or run Demucs/MFA.

```bash
python -m unittest discover -s tests -v
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [Demucs](https://github.com/facebookresearch/demucs) | Vocal/instrumental separation |
| [Librosa](https://librosa.org/) | Pitch detection and beat analysis |
| [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) | Word/phone alignment and G2P for OOV words |
| [Whisper](https://github.com/SYSTRAN/faster-whisper) | Second-opinion word timing on sung phrases |
| [Pyphen](https://github.com/Kozea/Pyphen) | Dictionary-based word hyphenation |
| [FFmpeg](https://ffmpeg.org/) | Audio format normalization |

---

## Project Structure

```
ultrastar-chart-generator/
├── app.py          # Main entry point and input file discovery
├── prepare.sh      # Local dependency/model installer
├── docker.sh       # Docker runner
├── Dockerfile      # Container image with MFA and language models
├── .env.example    # Input/output directory configuration template
├── tests/          # Unit tests (no model downloads)
└── src/
    └── core/
        ├── beats_analyzer.py
        ├── pitches_analyzer.py
        ├── syllables_transcriber.py
        ├── ultrastar_formatter.py
        └── vocals_separator.py
```

---

## Legal Disclaimer

This project does **not** include or distribute any music, lyrics, cover art, or
generated charts. You must provide your own files and ensure you have the
rights to use them. Generated charts are derivative works of the supplied
lyrics and audio — do not redistribute them without permission.

This project is not affiliated with or endorsed by UltraStar Deluxe,
UltraStar Play / Melody Mania, or any music rights holder.

---

## License

[MIT](LICENSE)
