"""Whisper second opinion used to detect and fix MFA drift."""

import re
from difflib import SequenceMatcher
from pathlib import Path

import pyphen
import textgrid


class WhisperCheckMixin:
    @staticmethod
    def _normalize_word(word: str) -> str:
        return re.sub(r"[^\w]", "", word.lower().strip())

    def _align_whisper_words(
        self, words: list[dict], tokens: list[dict]
    ) -> list[tuple[float, float] | None]:
        """Align Whisper words to the lyric tokens.

        Returns one (start, end) per token, or None when Whisper did not
        recognize that word (the formatter then interpolates its timing).
        """
        lyric_norms = [self._normalize_word(t["display"]) for t in tokens]
        trans_norms = [self._normalize_word(w["word"]) for w in words]

        matcher = SequenceMatcher(
            None, lyric_norms, trans_norms, autojunk=False
        )
        bounds: list[tuple[float, float] | None] = [None] * len(tokens)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    bounds[i1 + k] = (
                        words[j1 + k]["start"], words[j1 + k]["end"]
                    )
            elif tag == "replace" and j2 > j1:
                start = words[j1]["start"]
                end = words[j2 - 1]["end"]
                count = i2 - i1
                step = (end - start) / count
                for k in range(count):
                    bounds[i1 + k] = (
                        round(start + k * step, 4),
                        round(start + (k + 1) * step, 4),
                    )

        return bounds

    def _whisper_bounds(
        self, audio_path: Path, lyrics: str, language: str
    ) -> list[tuple[float, float] | None] | None:
        """Word timings from Whisper, used to correct drifted MFA lines."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("[WARNING] faster-whisper is not installed; skipping "
                  "Whisper drift correction.")
            return None

        print(f"[Whisper] Checking alignment with model "
              f"'{self.whisper_model}'")
        model = WhisperModel(
            self.whisper_model, device="cpu", compute_type="int8"
        )
        segments, _ = model.transcribe(
            str(audio_path),
            language=self._resolve_language(language),
            word_timestamps=True,
            # VAD keeps Whisper from hallucinating lyrics over long
            # instrumental sections (e.g. after the last sung line).
            vad_filter=True,
        )

        words = []
        for segment in segments:
            for word in segment.words or []:
                if word.end - word.start < 0.02:
                    continue
                words.append({
                    "word": word.word.strip(),
                    "start": float(word.start),
                    "end": float(word.end),
                })

        return self._align_whisper_words(
            words, self._lyrics_tokens(lyrics)
        )

    def _needs_whisper_check(
        self, tg_path: Path, lyrics: str, language: str
    ) -> bool:
        """Detect MFA alignments that deserve a Whisper cross-check.

        Speech models compress sustained singing, and very short words are
        the clearest symptom. Large gaps between words of the same lyric line
        also indicate that the aligner drifted over an instrumental section.
        """
        tg = textgrid.TextGrid.fromFile(str(tg_path))
        words_tier = next(
            (t for t in tg.tiers if t.name.lower() == "words"), None
        )
        if words_tier is None:
            return False

        word_intervals = [
            wi for wi in words_tier
            if wi.mark.strip() and wi.mark.strip() not in {"sil", "sp", "spn"}
        ]
        tokens = self._lyrics_tokens(lyrics)
        if len(word_intervals) != len(tokens):
            return True

        dic = pyphen.Pyphen(lang=self._pyphen_language(language))

        for idx, word_interval in enumerate(word_intervals):
            token = tokens[idx]
            parts = self._display_word_parts(
                dic, word_interval.mark.strip(), token["display"]
            )
            minimum = self.MIN_WORD_DURATION_PER_SYLLABLE * len(parts)
            if word_interval.maxTime - word_interval.minTime < minimum:
                return True

            if idx > 0 and tokens[idx - 1]["line"] == token["line"]:
                gap = word_interval.minTime - word_intervals[idx - 1].maxTime
                if gap > self.SUSPICIOUS_LINE_GAP:
                    return True

        return False

    def _drifted_lines(
        self,
        tokens: list[dict],
        word_intervals: list,
        whisper_bounds: list[tuple[float, float] | None],
    ) -> set[int]:
        """Lyric lines that MFA broke and Whisper can repair.

        A large mean shift alone is not enough: repetitions in the lyrics can
        make Whisper match a different repetition. A long gap inside the line
        (or right before it) is the signature of the aligner jumping over an
        instrumental section or a sustained phrase.
        """
        shifts: dict[int, list[float]] = {}
        severe_lines: set[int] = set()
        bounds_by_line: dict[int, list[tuple[float, float]]] = {}

        for idx, word_interval in enumerate(word_intervals):
            if idx >= len(tokens):
                continue

            if idx > 0:
                previous = word_intervals[idx - 1]
                gap = word_interval.minTime - previous.maxTime
                if gap > self.SUSPICIOUS_LINE_GAP:
                    severe_lines.add(tokens[idx]["line"])

            if idx >= len(whisper_bounds):
                continue
            bound = whisper_bounds[idx]
            if bound is None:
                continue
            line = tokens[idx]["line"]
            shifts.setdefault(line, []).append(
                abs(word_interval.minTime - bound[0])
            )
            bounds_by_line.setdefault(line, []).append(bound)

        # If Whisper itself leaves a large gap inside a lyric line, it glued
        # the line's words to the wrong phrase or repetition. That line is
        # not a trustworthy correction.
        unreliable = set()
        for line, bounds in bounds_by_line.items():
            for (_, previous_end), (next_start, _) in zip(bounds, bounds[1:]):
                if next_start - previous_end > self.SUSPICIOUS_LINE_GAP:
                    unreliable.add(line)
                    break

        return {
            line for line, values in shifts.items()
            if line in severe_lines
            and line not in unreliable
            and sum(values) / len(values) > self.WHISPER_DRIFT
        }

    # ------------------------------------------------------------- phonemes
