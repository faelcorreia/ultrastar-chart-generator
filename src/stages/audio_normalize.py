"""Audio normalization (B1): accept FLAC/WAV/M4A/OGG/OPUS/MP3, output MP3.

The ffmpeg call is injectable (`runner`) so tests need no ffmpeg.
An input that is already MP3 is never re-encoded.
"""
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

SUPPORTED_EXTENSIONS = (".mp3", ".flac", ".wav", ".m4a", ".ogg", ".opus")

Runner = Callable[[List[str]], object]


class AudioNormalizeError(RuntimeError):
    """Unsupported input, missing ffmpeg, or failed conversion."""


def _default_runner(cmd: List[str]):
    return subprocess.run(cmd, capture_output=True, text=True)


def is_supported_audio(path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def normalize_audio(input_path, output_dir=None, runner: Optional[Runner] = None,
                    which: Callable[[str], Optional[str]] = shutil.which,
                    quality: int = 2) -> Path:
    """Return a path to an MP3 for `input_path`.

    MP3 inputs are returned unchanged. Others are converted with ffmpeg
    (libmp3lame VBR `quality`, 0=best) into `output_dir` (default: next to
    the input) as `<stem>.mp3`.
    """
    src = Path(input_path)
    if not src.is_file():
        raise AudioNormalizeError(f"audio file not found: {src}")
    ext = src.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AudioNormalizeError(
            f"unsupported audio format '{src.suffix}' for {src.name}; "
            f"supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    if ext == ".mp3":
        return src
    exe = which("ffmpeg")
    if not exe:
        raise AudioNormalizeError(
            "ffmpeg not found in PATH; install it (e.g. pacman -S ffmpeg / "
            "apt install ffmpeg)")
    out_dir = Path(output_dir) if output_dir is not None else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (src.stem + ".mp3")
    cmd = [exe, "-y", "-nostdin", "-i", str(src), "-vn",
           "-codec:a", "libmp3lame", "-q:a", str(quality), str(out)]
    result = (runner or _default_runner)(cmd)
    code = getattr(result, "returncode", 0)
    if code:
        err = (getattr(result, "stderr", "") or "").strip()
        raise AudioNormalizeError(f"ffmpeg failed (exit {code}): {err}")
    if not out.is_file() or out.stat().st_size == 0:
        raise AudioNormalizeError(f"ffmpeg produced no audio: {out}")
    return out
