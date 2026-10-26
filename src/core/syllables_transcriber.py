from src.core.audio_processor import AudioProcessor
from src.core.syllables_language import LanguageMixin
from src.core.syllables_aligners import AlignersMixin
from src.core.syllables_whisper import WhisperCheckMixin
from src.core.syllables_parse import SyllabifyMixin
from src.core.syllables_validate import ValidationMixin
import json
import os
import re
import shutil


class SyllablesTranscriber(
    LanguageMixin, AlignersMixin, WhisperCheckMixin, SyllabifyMixin, ValidationMixin,
    AudioProcessor,
):
    """Align the provided lyrics to the isolated vocals with MFA and split
    them into syllables.

    The singer-facing text stays orthographic (pyphen), while the timings come
    from the phone alignment produced by MFA.
    """

    MIN_VALID_SYLLABLE_DURATION = 0.03  # seconds
    UNALIGNED_WARN_RATIO = 0.05

    # MFA's speech models compress sustained vowels. A word shorter than this
    # (per syllable) is a sign that the alignment needs a second opinion.
    MIN_WORD_DURATION_PER_SYLLABLE = 0.08
    # Seconds between words of the same lyric line that indicate the aligner
    # jumped over a section. Sustained notes can leave shorter gaps, so this
    # must stay above them or repeated choruses get matched to the wrong
    # repetition.
    SUSPICIOUS_LINE_GAP = 2.5
    # Mean word shift (seconds) at which a whole lyric line is re-timed with
    # Whisper instead of MFA.
    WHISPER_DRIFT = 1.0

    def __init__(
        self,
        songs: dict,
        acoustic_model: str | None = None,
        dictionary: str | None = None,
        g2p: str | None = None,
        whisper_model: str = "small",
    ):
        super().__init__(songs)
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary
        self.g2p = g2p
        # Used only as a second opinion when MFA drifts on sung phrases.
        self.whisper_model = whisper_model

    # ------------------------------------------------------------------ models

    # -------------------------------------------------------------- pipeline
    def transcribe_vocals(self, artist_song: str) -> list[dict]:
        song = self.songs[artist_song]
        audio_path = song["vocals"]
        lyrics_path = song["lyrics"]
        language = song.get("language", "pt")
        acoustic_model, dictionary, g2p = self._models_for(language)

        output_json_path = self.output_dir / f"{artist_song} - syllables.json"

        with open(lyrics_path, encoding="utf-8") as f:
            lyrics = f.read()

        # Separate corpus/output directories, and a directory per song.
        # Previously input_dir and output_dir were the same path, leaving
        # stale TextGrids from previous runs behind.
        work_root = self.output_dir / ".mfa_work" / self._safe_name(artist_song)
        input_dir = work_root / "input"
        output_dir = work_root / "output"

        if work_root.exists():
            shutil.rmtree(work_root)
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        use_ctc = os.environ.get("ALIGNER", "mfa").lower() == "ctc"
        if use_ctc:
            tg_path = self._run_ctc(audio_path, lyrics, language, output_dir)
        else:
            self._prepare_mfa_input(audio_path, lyrics, input_dir)
            self._run_mfa(
                input_dir, output_dir, acoustic_model, dictionary, g2p
            )
            tg_path = self._find_textgrid(output_dir)

        whisper_bounds = None
        # The Whisper drift check exists to patch MFA; CTC does not need it.
        if not use_ctc and self.whisper_model and self._needs_whisper_check(
            tg_path, lyrics, language
        ):
            whisper_bounds = self._whisper_bounds(
                audio_path, lyrics, language
            )

        syllables_data = self._parse_textgrid(
            tg_path, lyrics, language=language,
            whisper_bounds=whisper_bounds,
        )
        self._validate_alignment(syllables_data, language)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(syllables_data, f, ensure_ascii=False, indent=2)

        self.songs[artist_song]["syllables"] = output_json_path
        return syllables_data

    @staticmethod
    def _safe_name(name: str) -> str:
        safe = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._")
        return safe[:80] or "song"
