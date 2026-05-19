from src.core.audio_processor import AudioProcessor
import json
from faster_whisper import WhisperModel


class WordsTranscriber(AudioProcessor):

    def __init__(
        self,
        songs: dict,
        model_size: str = "small",
    ):
        super().__init__(songs)
        self.model_size = model_size

    def transcribe_vocals(
        self, artist_song: str
    ) -> list[dict]:
        audio_path = self.songs[artist_song]["vocals"]

        lyrics_path = self.songs[artist_song]["lyrics"]
        language = self.songs[artist_song]["language"]

        output_json_name = f"{artist_song} - words.json"
        output_json_path = self.output_dir / output_json_name

        with open(lyrics_path, 'r', encoding='utf-8') as f:
            lyrics = f.read()
        model = WhisperModel(
            self.model_size, device='cpu', compute_type='int8')
        segments, _ = model.transcribe(audio_path, language=language,
                                       initial_prompt=lyrics, word_timestamps=True)
        words_data = []
        for segment in segments:
            if segment.words:
                for word in segment.words:
                    words_data.append({'word': word.word.strip(),
                                       'start': word.start, 'end': word.end})
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(words_data, f, ensure_ascii=False)

        self.songs[artist_song]["words"] = output_json_path
