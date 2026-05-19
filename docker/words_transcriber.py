import sys
import json

from faster_whisper import WhisperModel

if __name__ == "__main__":
    audio_file, lyrics_txt, out, language, size = sys.argv[1:6]
    with open(lyrics_txt, 'r', encoding='utf-8') as f:
        lyrics = f.read()
    model = WhisperModel(size, device='cpu', compute_type='int8')
    segments, _ = model.transcribe(audio_file, language=language,
                                   initial_prompt=lyrics, word_timestamps=True)
    words_data = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                words_data.append({'word': word.word.strip(),
                                   'start': word.start, 'end': word.end})
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(words_data, f, ensure_ascii=False)
