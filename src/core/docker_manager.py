import os
import subprocess
import src.core.config as config
from pathlib import Path


class DockerManager:

    def __init__(
        self
    ):
        self.input_dir = Path(config.INPUT_DIR)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir = Path(config.OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True)
        self.local_cache = Path("./.cache")
        self.local_cache.mkdir(exist_ok=True)
        self.image_name = "local/audio-processor"

    def run_container(self, args: list[str]) -> None:
        host_user_mapping = f"{os.getuid()}:{os.getgid()}"

        container_command = [
            "docker",
            "run",
            "--rm",
            "-u",
            host_user_mapping,
            "-v",
            f"{self.input_dir}:/data/input",
            "-v",
            f"{self.output_dir}:/data/output",
            "-v",
            f"{self.local_cache}:/data/cache",
            "-e",
            "XDG_CACHE_HOME=/data/cache",
            "-e",
            "TORCH_HOME=/data/cache/torch",
            "-e",
            "NUMBA_CACHE_DIR=/tmp",
            "-e",
            "USER=runner",  # Bypass getpwuid missing UID error in Python utilities
            self.image_name
        ]
        container_command.extend(args)

        subprocess.run(container_command, capture_output=False, check=True)
