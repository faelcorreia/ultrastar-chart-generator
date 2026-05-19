import json
import sys
import librosa

if __name__ == "__main__":
    audio_path = sys.argv[1]
    output_json = sys.argv[2]

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
    with open(output_json, "w", encoding="utf-8") as file_handle:
        json.dump(result_data, file_handle, ensure_ascii=False)
