import shutil
from src.core.audio_processor import AudioProcessor
from pathlib import Path
import demucs.separate


class VocalsSeparator(AudioProcessor):

    def __init__(
        self,
        songs: dict,
        model_name: str = "htdemucs",
    ):
        super().__init__(songs)
        self.model_name = model_name

    def separate_vocals(self, artist_song: str) -> None:
        audio_input = self.songs[artist_song]["full_track"]

        demucs.separate.main([
            "--two-stems=vocals",
            "--mp3",
            "-n", self.model_name,
            "-o", str(self.output_dir),
            str(audio_input)
        ])

        model_folder = self.output_dir / self.model_name
        song_folder = model_folder / Path(audio_input).stem
        vocals_source = song_folder / "vocals.mp3"
        instrumental_source = song_folder / "no_vocals.mp3"

        vocals_destination = self.output_dir / f"{artist_song} - vocals.mp3"
        instrumental_destination = self.output_dir / \
            f"{artist_song} - instrumental.mp3"

        vocals_source.move(vocals_destination)
        instrumental_source.move(instrumental_destination)

        shutil.rmtree(model_folder)

        self.songs[artist_song].update({
            "instrumental": instrumental_destination,
            "vocals": vocals_destination
        })
