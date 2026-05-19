from pathlib import Path
from src.core.docker_manager import DockerManager


class BeatsAnalyzer(DockerManager):

    def __init__(self, inputs: dict):
        super().__init__()
        self.inputs = inputs

    def analyze_beats(self, artist_song: str) -> None:
        audio_path: Path = self.inputs[artist_song]["no_vocals"]
        file_name = Path(audio_path).name

        if Path(audio_path.parent) == self.output_dir:
            container_audio = f"/data/output/{file_name}"
        else:
            container_audio = f"/data/input/{file_name}"

        output_json_name = f"{artist_song} - beats.json"
        host_json_path = self.output_dir / output_json_name
        container_json = f"/data/output/{output_json_name}"

        args = [
            "python3",
            "/app/beats_analyzer.py",
            container_audio,
            container_json,
        ]
        self.run_container(args)

        self.inputs[artist_song]["beats"] = host_json_path
