from src.core.audio_processor import AudioProcessor
import json
import math
import librosa


class PitchesAnalyzer(AudioProcessor):

    def __init__(self, songs: dict):
        super().__init__(songs)

    def analyze_pitches(self, artist_song: str) -> None:
        audio_path = self.songs[artist_song]["vocals"]

        output_json_name = f"{artist_song} - pitches.json"
        output_json_path = self.output_dir / output_json_name

        hop_length = 160
        sr_target = 16000

        # Load and resample to 16 kHz mono in one step
        audio, sample_rate = librosa.load(audio_path, sr=sr_target, mono=True)

        # pyin: probabilistic YIN — fast, accurate for vocals, no GPU needed.
        frequencies, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),  # ~65 Hz
            fmax=librosa.note_to_hz("C7"),  # ~2093 Hz
            sr=sample_rate,
            hop_length=hop_length,
            fill_na=0.0,  # unvoiced frames → 0 instead of NaN
        )

        frequencies = frequencies.tolist()

        # Convert Hz to MIDI note (0.0 stays 0 — silence/unvoiced)
        midi_notes = [
            round(float(librosa.hz_to_midi(f)), 2) if f > 0.0 else 0.0
            for f in frequencies
        ]

        times = [
            float(i * hop_length) / sample_rate for i in range(len(frequencies))
        ]

        # Per-frame confidence (0.0 when no pitch was detected). The formatter
        # uses this to discard noisy/transitional frames.
        voiced_probability = [
            float(p) if p is not None and not math.isnan(float(p)) else 0.0
            for p in voiced_prob
        ]

        result_data = {
            "times": times,
            "frequencies": frequencies,
            "midi_notes": midi_notes,
            "voiced_prob": voiced_probability,
        }

        with open(output_json_path, "w", encoding="utf-8") as fh:
            json.dump(result_data, fh, ensure_ascii=False)

        self.songs[artist_song]["pitches"] = output_json_path
