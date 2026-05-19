import src.core.config as config
from src.core.vocals_separator import VocalsSeparator
from src.core.words_transcriber import WordsTranscriber
from src.core.ultrastar_formatter import UltraStarFormatter
from src.core.beats_analyzer import BeatsAnalyzer
from src.core.pitches_analyzer import PitchesAnalyzer
from pathlib import Path


def get_test_data(artist_song, songs):
    songs[artist_song]["instrumental"] = config.OUTPUT_DIR / \
        f"{artist_song} - instrumental.mp3"
    songs[artist_song]["vocals"] = config.OUTPUT_DIR / \
        f"{artist_song} - vocals.mp3"

    songs[artist_song]["pitches"] = config.OUTPUT_DIR / \
        f"{artist_song} - pitches.json"

    songs[artist_song]["beats"] = config.OUTPUT_DIR / \
        f"{artist_song} - beats.json"

    songs[artist_song]["words"] = config.OUTPUT_DIR / \
        f"{artist_song} - words.json"


if __name__ == "__main__":
    songs = {}
    input_dir = Path(config.INPUT_DIR)
    for file in input_dir.iterdir():
        stem = Path(file).stem
        if stem.endswith(("full track", "lyrics", "cover")):
            artist_song = stem.rsplit("-", 1)[0].strip()
            if artist_song not in songs:
                artist, title = artist_song.split("-")
                artist = artist.strip()
                title = title.strip()
                songs[artist_song] = {
                    "artist": artist,
                    "title": title
                }
            # Get input audio file
            if stem.endswith("full track"):
                songs[artist_song]["full_track"] = input_dir / file
            # Get lyrics and language
            elif stem.endswith("lyrics"):
                songs[artist_song]["lyrics"] = input_dir / file
                songs[artist_song]["language"] = stem.rsplit(
                    "-", 1)[1].split("lyrics")[0].strip()
            # Get cover image
            elif stem.endswith("cover"):
                songs[artist_song]["cover"] = input_dir / file

    if len(songs) > 0:
        vocals = VocalsSeparator(songs)
        pitches = PitchesAnalyzer(songs)
        beats = BeatsAnalyzer(songs)
        words = WordsTranscriber(songs)
        ultrastar = UltraStarFormatter(songs)
        for artist_song in songs:
            print(f"Analyzing {artist_song}")

            print(f"=> Running Demucs to extract vocals")
            vocals.separate_vocals(artist_song=artist_song)

            print(f"=> Running Librosa to extract vocals pitches")
            pitches.analyze_pitches(artist_song=artist_song)

            print(f"=> Running Librosa to extract BPM")
            beats.analyze_beats(artist_song=artist_song)

            print(
                f"=> Running Faster Whisper to timestamp lyrics")
            words.transcribe_vocals(
                artist_song=artist_song)

            print(f"=> Creating UltraStar song folder")
            ultrastar.format_ultrastar_song_folder(artist_song=artist_song)
