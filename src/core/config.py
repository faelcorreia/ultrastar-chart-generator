from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

INPUT_DIR = Path(os.environ["INPUT_DIR"]).resolve()
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"]).resolve()
