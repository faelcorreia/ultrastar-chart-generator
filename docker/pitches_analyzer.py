import json
import sys
import librosa

if __name__ == "__main__":
    audio_path = sys.argv[1]
    output_json = sys.argv[2]

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

    result_data = {
        "times": times,
        "frequencies": frequencies,
        "midi_notes": midi_notes,
    }

    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(result_data, fh, ensure_ascii=False)