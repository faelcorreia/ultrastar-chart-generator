"""Forced-alignment backends (MFA / CTC) and TextGrid discovery."""

import subprocess
from pathlib import Path


class AlignersMixin:
    def _prepare_mfa_input(
        self, audio_path: Path, lyrics: str, input_dir: Path
    ) -> None:
        # Use a simple, fixed name to avoid issues with special characters in MFA.
        wav_dest = input_dir / "audio.wav"

        # Always normalize to 16 kHz mono. MFA requires that sample rate, and
        # .wav inputs may also arrive at 44.1/48 kHz.
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", str(wav_dest)],
            check=True, capture_output=False,
        )

        transcript_path = input_dir / "audio.lab"
        transcript_path.write_text(
            self._clean_lyrics(lyrics), encoding="utf-8"
        )

    def _run_mfa(
        self,
        input_dir: Path,
        output_dir: Path,
        acoustic_model: str,
        dictionary: str,
        g2p: str | None,
    ) -> None:
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # mfa align [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY
        cmd_list = [
            "mfa", "align",
            "--clean",
            "--single_speaker",
            str(input_dir.absolute()),
            dictionary,
            acoustic_model,
            str(output_dir.absolute()),
            "--beam", "100",
            "--retry_beam", "400",
        ]

        # Generate pronunciations for OOV words (e.g. "solar" is missing from
        # the pt-BR dictionary, which only has "solares"). Without this, MFA
        # leaves the word without phones and the formatter ends up
        # interpolating its timing.
        if g2p:
            cmd_list += ["--g2p_model_path", g2p]

        print(
            f"DEBUG: MFA acoustic={acoustic_model} "
            f"dictionary={dictionary} g2p={g2p or '<none>'}"
        )
        print(f"DEBUG: Running MFA command: {' '.join(cmd_list)}")

        result = subprocess.run(cmd_list, capture_output=False, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"MFA failed with exit code {result.returncode}")

    def _run_ctc(
        self, audio_path: Path, lyrics: str, language: str, output_dir: Path
    ) -> Path:
        """Align the lyrics with the CTC aligner and write a TextGrid."""
        import librosa
        from src.core import ctc_word_aligner

        words = [t["mfa"] for t in self._lyrics_tokens(lyrics)]
        print(f"DEBUG: CTC aligner, {len(words)} word(s), language={language}")
        bounds = ctc_word_aligner.align_words(audio_path, words, language)
        duration = librosa.get_duration(path=str(audio_path))
        return ctc_word_aligner.to_textgrid(
            words, bounds, duration, output_dir / "audio.TextGrid"
        )

    def _find_textgrid(self, output_dir: Path) -> Path:
        """Return the TextGrid produced by the current run.

        The work directory is cleaned before every run, so a stale file from a
        previous song cannot be picked up here.
        """
        exact = output_dir / "audio.TextGrid"
        if exact.exists():
            return exact

        candidates = sorted(output_dir.rglob("*.TextGrid"))
        if not candidates:
            raise FileNotFoundError("MFA did not produce any TextGrid")
        if len(candidates) > 1:
            print(
                f"[WARNING] MFA produced {len(candidates)} TextGrids; using "
                f"{candidates[0].name}"
            )
        return candidates[0]

    # --------------------------------------------------------------- whisper
