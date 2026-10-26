"""MIDI audio render (G2) via FluidSynth + soundfont.

Last resort: the original MP3 has priority over a synthesized render.
The subprocess call is injectable (`runner`) so tests need no fluidsynth.
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

SOUNDFONT_ENV = "ULTRASTAR_SOUNDFONT"
DEFAULT_SOUNDFONTS = (
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/soundfonts/default.sf2",
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/freepats-general-midi.sf2",
)

Runner = Callable[[List[str]], object]


class MidiRenderError(RuntimeError):
    """FluidSynth, the soundfont or the render itself is unavailable."""


def _default_runner(cmd: List[str]):
    return subprocess.run(cmd, capture_output=True, text=True)


def find_soundfont(soundfont: Union[str, Path, None] = None,
                   candidates: Sequence[str] = DEFAULT_SOUNDFONTS
                   ) -> Optional[Path]:
    """Explicit path, then $ULTRASTAR_SOUNDFONT, then common locations."""
    if soundfont:
        return Path(soundfont)
    env = os.environ.get(SOUNDFONT_ENV)
    if env:
        return Path(env)
    for c in candidates:
        if Path(c).is_file():
            return Path(c)
    return None


def render_midi(midi_path, output_path, soundfont=None, sample_rate: int = 44100,
                runner: Optional[Runner] = None,
                which: Callable[[str], Optional[str]] = shutil.which,
                gain: float = 0.8) -> Path:
    """Render `midi_path` to a WAV file; return the output path."""
    midi_path, output_path = Path(midi_path), Path(output_path)
    if not midi_path.is_file():
        raise MidiRenderError(f"MIDI file not found: {midi_path}")
    exe = which("fluidsynth")
    if not exe:
        raise MidiRenderError(
            "fluidsynth not found in PATH; install it (e.g. pacman -S "
            "fluidsynth / apt install fluidsynth)")
    sf = find_soundfont(soundfont)
    if sf is None:
        raise MidiRenderError(
            f"no soundfont found; pass soundfont= or set {SOUNDFONT_ENV}")
    if not sf.is_file():
        raise MidiRenderError(f"soundfont not found: {sf}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-ni", "-g", str(gain), "-r", str(sample_rate),
           "-F", str(output_path), str(sf), str(midi_path)]
    result = (runner or _default_runner)(cmd)
    code = getattr(result, "returncode", 0)
    if code:
        err = (getattr(result, "stderr", "") or "").strip()
        raise MidiRenderError(f"fluidsynth failed (exit {code}): {err}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise MidiRenderError(f"fluidsynth produced no audio: {output_path}")
    return output_path
