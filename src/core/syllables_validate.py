"""Sanity checks on the final alignment."""

import statistics


class ValidationMixin:
    def _validate_alignment(self, syllables: list[dict], language: str) -> None:
        aligned = [s for s in syllables if s.get("start") is not None]
        unaligned = [s for s in syllables if s.get("unaligned")]

        if not aligned:
            raise RuntimeError(
                "MFA did not align any syllable. Make sure the lyrics match "
                f"the audio and that the '{language}' models are installed."
            )

        short = [
            s for s in aligned
            if s["end"] - s["start"] < self.MIN_VALID_SYLLABLE_DURATION
        ]
        words = sorted({s.get("word", s["syllable"]) for s in unaligned})

        print(
            f"[MFA] {len(syllables)} syllables: {len(aligned)} aligned, "
            f"{len(unaligned)} unaligned."
        )
        if words:
            print(f"[MFA] Unaligned words: {', '.join(words)}")
        if short:
            print(
                f"[WARNING] {len(short)} syllable(s) shorter than "
                f"{self.MIN_VALID_SYLLABLE_DURATION * 1000:.0f} ms — the "
                "alignment may be inaccurate."
            )
        if len(unaligned) / len(syllables) > self.UNALIGNED_WARN_RATIO:
            print(
                "[WARNING] Too many unaligned syllables. Check that the audio "
                "is the isolated vocal, that the lyrics match the song, and "
                "that the language G2P model is installed."
            )

        durations_by_line: dict[int, list[float]] = {}
        for syllable in aligned:
            line = syllable.get("line")
            if line is not None:
                durations_by_line.setdefault(line, []).append(
                    syllable["end"] - syllable["start"]
                )

        compressed_threshold = self.MIN_VALID_SYLLABLE_DURATION * 2
        compressed_lines = sorted(
            line + 1
            for line, durations in durations_by_line.items()
            if statistics.median(durations) < compressed_threshold
        )
        if compressed_lines:
            print(
                "[WARNING] Lyric line(s) "
                + ", ".join(str(line) for line in compressed_lines)
                + " are compressed (median syllable < "
                f"{compressed_threshold * 1000:.0f} ms). Check that the "
                "lyrics match the recording; extra repeated lines cannot be "
                "aligned."
            )
