import src.core.config as config
from pathlib import Path


class AudioProcessor:
    def __init__(self, songs):
        self.input_dir = Path(config.INPUT_DIR)
        self.output_dir = Path(config.OUTPUT_DIR)
        self.local_cache = Path("./.cache")
        self.image_name = "local/audio-processor"
        self.songs = songs
