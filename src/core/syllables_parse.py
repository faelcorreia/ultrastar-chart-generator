"""Phone grouping and TextGrid parsing into syllables."""

import re
from pathlib import Path

import pyphen
import textgrid


class SyllabifyMixin:
    # IPA vowels used by the Portuguese/Spanish/English IPA acoustic models.
    # The ARPAbet set keeps english_us_arpa models working too.
    _IPA_VOWELS = set(
        "aeiouyæəɑɛɪɔʊʌɐɜɞɘɵɤɨʉɯøœɶɒɚɝɞɤ"
    )
    _ARPA_VOWELS = {
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER",
        "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    }

    def _is_vowel_phone(self, phone: str) -> bool:
        """Return True if the phone is (or contains) a vowel nucleus."""
        phone = phone.strip()
        if not phone:
            return False

        # ARPAbet: AA1, EH0, Y, ... (lexical units, not characters). Decide
        # only from the ARPAbet set so the consonantal "Y" is not mistaken for
        # the IPA vowel "y".
        if re.fullmatch(r"[A-Z]+[0-9]?", phone):
            base = re.sub(r"\d", "", phone)
            return base in self._ARPA_VOWELS

        # IPA: vowels may carry diacritics (ẽ, ã, aː, ...), so a single
        # matching character is enough.
        return any(ch in self._IPA_VOWELS for ch in phone.lower())

    def _group_phones_into_syllables(
        self, phones: list[tuple[str, float, float]]
    ) -> list[dict]:
        """Group phones into syllables using the vowel nucleus as delimiter."""
        syllables: list[dict] = []
        buffer: list[tuple[str, float, float]] = []

        for phone, start, end in phones:
            buffer.append((phone, start, end))
            if self._is_vowel_phone(phone):          # nucleus found
                syllables.append({
                    "syllable": "".join(p for p, _, _ in buffer),
                    "start": round(buffer[0][1], 3),
                    "end":   round(end, 3),
                })
                buffer = []

        # Leftover consonants (final coda) attach to the last syllable.
        if buffer and syllables:
            syllables[-1]["end"] = round(buffer[-1][2], 3)

        return syllables

    def _split_word_proportionally(
        self, word_phones: list[tuple[str, float, float]], parts: list[str]
    ) -> list[dict]:
        """Fallback when the pyphen syllable count does not match the number of
        vowel nuclei found by MFA.

        Divides the word duration into equal parts but snaps each boundary to
        the nearest phone boundary (instead of cutting through a phone).
        """
        n_syls = len(parts)
        t0, t1 = word_phones[0][1], word_phones[-1][2]

        boundaries = [i + 1 for i in range(len(word_phones) - 1)]
        split_points: list[int] = []
        last_cut = 0

        for s in range(1, n_syls):
            target = t0 + (t1 - t0) * s / n_syls
            candidates = [i for i in boundaries if i > last_cut]
            if not candidates:
                break
            cut = min(candidates, key=lambda i: abs(word_phones[i][1] - target))
            split_points.append(cut)
            last_cut = cut

        # If there are not enough phone boundaries, use pure time splitting.
        if len(split_points) != n_syls - 1:
            step = (t1 - t0) / n_syls
            return [
                {
                    "syllable": part,
                    "start": round(t0 + s * step, 3),
                    "end": round(t0 + (s + 1) * step, 3),
                }
                for s, part in enumerate(parts)
            ]

        edges = [0] + split_points + [len(word_phones)]
        result = []
        for s, part in enumerate(parts):
            seg = word_phones[edges[s]:edges[s + 1]]
            result.append({
                "syllable": part,
                "start": round(seg[0][1], 3),
                "end": round(seg[-1][2], 3),
            })
        return result

    def _parse_textgrid(
        self,
        tg_path: Path,
        lyrics: str,
        language: str = "pt",
        whisper_bounds: list[tuple[float, float] | None] | None = None,
    ) -> list[dict]:
        tg = textgrid.TextGrid.fromFile(str(tg_path))

        words_tier = phones_tier = None
        for tier in tg.tiers:
            name = tier.name.lower()
            if name == "words":
                words_tier = tier
            elif name == "phones":
                phones_tier = tier

        if not phones_tier:
            raise ValueError("Tier 'phones' not found in the TextGrid")

        all_phones = [
            (interval.mark.strip(), interval.minTime, interval.maxTime)
            for interval in phones_tier
            if interval.mark.strip() not in {"sil", "sp", "spn", ""}
        ]

        if words_tier is None:
            return self._group_phones_into_syllables(all_phones)

        dic = pyphen.Pyphen(lang=self._pyphen_language(language))
        syllables: list[dict] = []

        # MFA aligns exactly the text it received, word by word, so the order
        # of the aligned words matches the original lyrics.
        word_intervals = [
            wi for wi in words_tier
            if wi.mark.strip() and wi.mark.strip() not in {"sil", "sp", "spn"}
        ]
        flat_lyrics = self._flatten_lyrics(lyrics)

        if len(word_intervals) != len(flat_lyrics):
            print(f"[WARNING] MFA aligned {len(word_intervals)} word(s), but "
                  f"the lyrics have {len(flat_lyrics)}; line breaks may not "
                  f"match the original lyrics exactly.")

        drifted_lines: set[int] = set()
        if whisper_bounds:
            if len(whisper_bounds) == len(word_intervals):
                drifted_lines = self._drifted_lines(
                    flat_lyrics, word_intervals, whisper_bounds
                )
                if drifted_lines:
                    print(
                        f"[Whisper] Re-timing {len(drifted_lines)} drifted "
                        f"lyric line(s) from word timestamps."
                    )
            else:
                print("[WARNING] Whisper and MFA word counts differ; "
                      "keeping the MFA alignment.")

        for idx, word_interval in enumerate(word_intervals):
            word = word_interval.mark.strip()
            w_start, w_end = word_interval.minTime, word_interval.maxTime

            # MFA aligns the normalized words in the same order as the lyrics,
            # so the matching original token supplies the spelling used in the
            # UltraStar file (case, accents and punctuation).
            token = flat_lyrics[idx] if idx < len(flat_lyrics) else {}
            display_word = token.get("display") or word
            line_break = token.get("line_break", False)

            line_index = token.get("line")
            drifted = line_index in drifted_lines
            whisper = (
                whisper_bounds[idx]
                if whisper_bounds and idx < len(whisper_bounds) else None
            )

            if drifted and whisper is None:
                # Whisper did not recognize this word: leave its timing empty
                # so the formatter interpolates between its neighbors.
                parts = self._display_word_parts(dic, word, display_word)
                for s, part in enumerate(parts):
                    syllables.append({
                        "syllable": part,
                        "word": display_word,
                        "start": None,
                        "end": None,
                        "unaligned": True,
                        "line": line_index,
                        "first_syllable": s == 0,
                        "line_break": line_break and s == len(parts) - 1,
                    })
                continue

            use_whisper = drifted and whisper is not None
            if use_whisper:
                w_start, w_end = whisper

            word_phones = [
                (ph, st, en) for ph, st, en in all_phones
                if st >= w_start - 0.001 and en <= w_end + 0.001
            ]

            # ── word without phones: MFA could not align it (OOV) ───────────
            if not word_phones and not use_whisper:
                print(f"[WARNING] Word without alignment: '{display_word}' "
                      f"({w_start:.3f}s – {w_end:.3f}s) — check the MFA dictionary")
                parts = self._display_word_parts(dic, word, display_word)
                for s, part in enumerate(parts):
                    syllables.append({
                        "syllable": part,
                        "word": display_word,
                        "start": None,   # no reliable timestamp
                        "end":   None,
                        "unaligned": True,
                        "line": line_index,
                        "first_syllable": s == 0,
                        "line_break": line_break and s == len(parts) - 1,
                    })
                continue

            syl_parts = self._display_word_parts(dic, word, display_word)
            n_syls = len(syl_parts)

            if n_syls == 1:
                syllables.append({
                    "syllable": syl_parts[0],
                    "start": round(
                        w_start if use_whisper else word_phones[0][1], 3
                    ),
                    "end": round(
                        w_end if use_whisper else word_phones[-1][2], 3
                    ),
                    "line": line_index,
                    "first_syllable": True,
                    "line_break": line_break,
                })
                continue

            vowel_idxs = [
                i for i, (ph, _, _) in enumerate(word_phones)
                if self._is_vowel_phone(ph)
            ]

            word_syllables: list[dict] = []
            if len(vowel_idxs) == n_syls:
                for s, vi in enumerate(vowel_idxs):
                    seg_start = vowel_idxs[s - 1] + 1 if s > 0 else 0
                    seg = word_phones[seg_start: vi + 1]
                    if seg:
                        word_syllables.append({
                            "syllable": syl_parts[s],
                            "start": round(seg[0][1], 3),
                            "end":   round(seg[-1][2], 3),
                        })

                if use_whisper:
                    # Keep the MFA split inside the word, but make the first
                    # and last syllable match Whisper's word boundaries.
                    for piece in word_syllables:
                        piece["start"] = round(
                            max(w_start, min(w_end, piece["start"])), 3
                        )
                        piece["end"] = round(
                            max(piece["start"], min(w_end, piece["end"])), 3
                        )
                    if word_syllables:
                        word_syllables[0]["start"] = round(w_start, 3)
                        word_syllables[-1]["end"] = round(w_end, 3)
                elif (word_syllables
                      and vowel_idxs[-1] < len(word_phones) - 1):
                    word_syllables[-1]["end"] = round(word_phones[-1][2], 3)

            if not word_syllables:
                if use_whisper:
                    # No usable MFA phones inside the corrected word:
                    # distribute the Whisper interval evenly.
                    step = (w_end - w_start) / n_syls
                    word_syllables = [
                        {
                            "syllable": syl_parts[s],
                            "start": round(w_start + s * step, 3),
                            "end": round(w_start + (s + 1) * step, 3),
                        }
                        for s in range(n_syls)
                    ]
                else:
                    word_syllables = self._split_word_proportionally(
                        word_phones, syl_parts
                    )

            for s, syl in enumerate(word_syllables):
                syl["line"] = line_index
                syl["first_syllable"] = s == 0
                syl["line_break"] = line_break and s == len(word_syllables) - 1
                syllables.append(syl)

        return syllables

    @staticmethod
    def _word_parts(dic: pyphen.Pyphen, word: str) -> list[str]:
        parts = [p for p in dic.inserted(word, hyphen="|").split("|") if p]
        return parts or [word]

    @classmethod
    def _display_word_parts(
        cls, dic: pyphen.Pyphen, mfa_word: str, display_word: str
    ) -> list[str]:
        """Split ``display_word`` into syllables without losing its spelling.

        pyphen handles capitalization and punctuation, so the display word is
        hyphenated directly. If its syllable count ever differs from the MFA
        word (digits or unusual punctuation), the MFA boundaries are mapped
        back onto the display text instead, because that count is what the
        timing logic uses.
        """
        display_parts = cls._word_parts(dic, display_word)
        mfa_parts = cls._word_parts(dic, mfa_word)
        if len(display_parts) == len(mfa_parts):
            return display_parts
        return cls._map_parts(mfa_parts, mfa_word, display_word)

    @staticmethod
    def _map_parts(
        parts: list[str], mfa_word: str, display_word: str
    ) -> list[str]:
        """Slice ``display_word`` at the boundaries of ``parts``.

        ``display_word`` is ``mfa_word`` with its original case restored and
        punctuation added back, so every character of ``mfa_word`` can be
        located in the display text. Slicing the original text at the mapped
        boundaries keeps any punctuation that MFA dropped.
        """
        positions = []
        display_idx = 0
        for char in mfa_word:
            while (display_idx < len(display_word)
                   and display_word[display_idx].lower() != char):
                display_idx += 1
            if display_idx == len(display_word):
                return parts
            positions.append(display_idx)
            display_idx += 1

        cuts = [0]
        mfa_idx = 0
        for part in parts[:-1]:
            mfa_idx += len(part)
            cuts.append(positions[mfa_idx])
        cuts.append(len(display_word))

        return [display_word[start:end]
                for start, end in zip(cuts, cuts[1:])]

    # ------------------------------------------------------------ validation
