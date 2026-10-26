"""Compatibility shim: the writer now lives in src/writer."""
import subprocess  # noqa: F401  (tests patch src.core.ultrastar_formatter.subprocess)

from src.writer.constants import *  # noqa: F401,F403
from src.writer.formatter import UltraStarFormatter  # noqa: F401
