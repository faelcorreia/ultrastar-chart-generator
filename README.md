# Song2UltraStar 🎤

Convert any song into a ready-to-use [UltraStar](https://usdx.eu/) file — fully automated, from audio to synchronized lyrics with pitch detection.

> Compatible with UltraStar and any other karaoke app that supports the UltraStar format (`.txt`).

---

## How It Works

Song2UltraStar takes a raw song and runs it through a pipeline of audio analysis tools:

1. **[Demucs](https://github.com/facebookresearch/demucs)** — separates vocals from the full track
2. **[Faster Whisper](https://github.com/SYSTRAN/faster-whisper)** — transcribes lyrics with word-level timestamps
3. **[Librosa](https://librosa.org/)** — analyzes pitch and beat timing
4. Outputs a complete `.txt` file in UltraStar format, ready to drop into your game

---

## Input File Convention

Place your files in the same directory, following this naming pattern:

| File | Pattern | Example |
|------|---------|---------|
| Cover image | `Artist - Song Title - cover.xxx` | `Queen - Bohemian Rhapsody - cover.jpg` |
| Full audio track | `Artist - Song Title - full track.xxx` | `Queen - Bohemian Rhapsody - full track.mp3` |
| Lyrics language | `Artist - Song Title - lang lyrics.xxx` | `Queen - Bohemian Rhapsody - en lyrics.txt` |

The language tag (`lang`) must be a language code supported by Faster Whisper, such as `en`, `pt`, `es`, `de`, `fr`, etc.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (for Demucs and Faster Whisper)
- `ffmpeg` installed and available in PATH
- Docker (optional, for containerized usage)

---

## Installation

### Local

Run the setup script to install all Python dependencies:

```bash
bash prepare.sh
```

Then start the program:

```bash
python app.py
```

### Docker

To run Song2UltraStar in a Docker container:

```bash
bash docker.sh
```

---

## Output

The program generates a `.txt` file in the UltraStar format alongside your input files. It includes:

- Song metadata (artist, title, cover, audio path)
- BPM and gap values derived from beat analysis
- Note lines with pitch, duration, and syllable-level lyrics

The output is ready to be loaded directly into UltraStar Deluxe or any compatible karaoke application.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [Demucs](https://github.com/facebookresearch/demucs) | Vocal extraction |
| [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) | Speech recognition & word timestamps |
| [Librosa](https://librosa.org/) | Pitch detection & beat analysis |

---

## Project Structure

```
Song2UltraStar/
├── app.py          # Main entry point
├── prepare.sh      # Dependency installer
├── docker.sh       # Docker runner
└── ...
```
