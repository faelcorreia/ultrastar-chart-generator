from src.core.audio_processor import AudioProcessor
import librosa
import json


class BeatsAnalyzer(AudioProcessor):

    def __init__(self, songs: dict):
        super().__init__(songs)

    def analyze_beats(self, artist_song: str) -> None:
        audio_path = self.songs[artist_song]["instrumental"]

        output_json_name = f"{artist_song} - beats.json"
        output_json_path = self.output_dir / output_json_name

        # Load audio file with its native sampling rate
        y, sr = librosa.load(audio_path, sr=None)

        # Track beat positions and estimate tempo
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

        # Handle different librosa version return types for tempo
        bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)

        # Convert frame indices to timestamps in seconds
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        result_data = {"bpm": bpm, "beats": beat_times}

        # Export structured data to a JSON file
        with open(output_json_path, "w", encoding="utf-8") as file_handle:
            json.dump(result_data, file_handle, ensure_ascii=False)

        self.songs[artist_song]["beats"] = output_json_path
