import shutil
from src.core.docker_manager import DockerManager
from pathlib import Path


class VocalsSeparator(DockerManager):

    def __init__(
        self,
        inputs: dict,
        model_name: str = "htdemucs",
    ):
        super().__init__()
        self.inputs = inputs
        self.model_name = model_name

    def separate_vocals(self, artist_song: str) -> None:
        audio_input = self.inputs[artist_song]["full_track"]

        args = [
            "demucs",
            "--two-stems=vocals",
            "--mp3",
            "-n",
            self.model_name,
            "-o",
            "/data/output",
            f"/data/input/{Path(audio_input).name}",
        ]

        self.run_container(args)

        model_folder = self.output_dir / self.model_name
        song_folder = model_folder / Path(audio_input).stem
        vocals_source = song_folder / "vocals.mp3"
        instrumental_source = song_folder / "no_vocals.mp3"

        vocals_destination = self.output_dir / f"{artist_song} - vocals.mp3"
        instrumental_destination = self.output_dir / \
            f"{artist_song} - no vocals.mp3"

        vocals_source.move(vocals_destination)
        instrumental_source.move(instrumental_destination)

        shutil.rmtree(model_folder)

        self.inputs[artist_song].update({
            "no vocals": instrumental_destination,
            "vocals": vocals_destination
        })
