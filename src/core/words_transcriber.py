from src.core.docker_manager import DockerManager
from pathlib import Path


class WordsTranscriber(DockerManager):

    def __init__(
        self,
        inputs: dict,
        model_size: str = "small",
    ):
        super().__init__()
        self.inputs = inputs
        self.model_size = model_size

    def transcribe_vocals(
        self, artist_song: str
    ) -> list[dict]:
        audio_path = self.inputs[artist_song]["vocals"]

        lyrics_path = self.inputs[artist_song]["lyrics"]

        words_file_name = f"{artist_song} - words.json"
        output_json_path = self.output_dir / words_file_name

        args = [
            "python3",
            "/app/words_transcriber.py",
            f"/data/output/{Path(audio_path).name}",
            f"/data/input/{Path(lyrics_path).name}",
            f"/data/output/{words_file_name}",
            self.inputs[artist_song]["language"],
            self.model_size,
        ]
        self.run_container(args)

        self.inputs[artist_song]["words"] = output_json_path
