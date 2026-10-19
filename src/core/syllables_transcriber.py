from src.core.audio_processor import AudioProcessor
import json
import re
import shutil
import statistics
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
import pyphen
import textgrid


class SyllablesTranscriber(AudioProcessor):
    """Align the provided lyrics to the isolated vocals with MFA and split
    them into syllables.

    The singer-facing text stays orthographic (pyphen), while the timings come
    from the phone alignment produced by MFA.
    """

    # IPA vowels used by the Portuguese/Spanish/English IPA acoustic models.
    # The ARPAbet set keeps english_us_arpa models working too.
    _IPA_VOWELS = set(
        "aeiouyæəɑɛɪɔʊʌɐɜɞɘɵɤɨʉɯøœɶɒɚɝɞɤ"
    )
    _ARPA_VOWELS = {
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER",
        "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    }

    _LANG_MAP = {
        "pt": "pt_BR",
        "pt-br": "pt_BR",
        "en": "en_US",
        "en-us": "en_US",
        "es": "es",
        "es-es": "es",
        "es-419": "es",
    }

    # language -> (acoustic model, pronunciation dictionary, G2P model)
    _LANGUAGE_MODELS = {
        "pt": ("portuguese_mfa", "portuguese_brazil_mfa", "portuguese_brazil_mfa"),
        "en": ("english_mfa", "english_us_mfa", "english_us_mfa"),
        "es": ("spanish_mfa", "spanish_latin_america_mfa",
               "spanish_latin_america_mfa"),
    }

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
    def _resolve_language(self, language: str | None) -> str:
        """Return a configured language code (base language) for ``language``."""
        if not language:
            return "pt"

        lang = language.strip().lower().replace("_", "-")
        if lang in self._LANGUAGE_MODELS:
            return lang

        base = lang.split("-", 1)[0]
        if base in self._LANGUAGE_MODELS:
            return base

        supported = ", ".join(sorted(self._LANGUAGE_MODELS))
        raise ValueError(
            f"No MFA models configured for language '{language}'. "
            f"Supported languages: {supported}. For other languages, pass "
            "acoustic_model, dictionary and g2p to SyllablesTranscriber."
        )

    def _models_for(self, language: str | None) -> tuple[str, str, str | None]:
        """Return the MFA models to use for a song.

        Explicit models passed to the constructor take precedence over the
        per-language defaults.
        """
        if self.acoustic_model and self.dictionary:
            return self.acoustic_model, self.dictionary, self.g2p

        acoustic, dictionary, g2p = self._LANGUAGE_MODELS[
            self._resolve_language(language)
        ]
        return (
            self.acoustic_model or acoustic,
            self.dictionary or dictionary,
            self.g2p or g2p,
        )

    def _pyphen_language(self, language: str | None) -> str:
        lang = (language or "pt").strip().lower().replace("_", "-")
        if lang in self._LANG_MAP:
            return self._LANG_MAP[lang]

        # Allow custom models in other languages without breaking hyphenation,
        # as long as pyphen has a matching dictionary.
        base = lang.split("-", 1)[0]
        for available in pyphen.LANGUAGES:
            if available.lower() == base or available.lower().startswith(
                base + "_"
            ):
                return available
        return "pt_BR"

    # --------------------------------------------------------------- lyrics
    def _lyrics_tokens(self, lyrics: str) -> list[dict]:
        """Ordered word tokens of the lyrics, keeping the original spelling.

        ``display`` holds the word exactly as written (case, accents and
        punctuation), while ``mfa`` holds the normalized form MFA aligns
        (lowercase, without punctuation or digits). Hyphens split compound
        words, matching the text sent to MFA.
        """
        tokens: list[dict] = []
        lines = [ln.strip()
                 for ln in lyrics.strip().splitlines() if ln.strip()]

        for line_index, line in enumerate(lines):
            line_tokens = []
            for raw_word in line.replace("-", " ").split():
                mfa_word = re.sub(r"[^\w\s']", "", raw_word)
                mfa_word = re.sub(r"\d+", "", mfa_word).lower()
                # Punctuation-only tokens are ignored by MFA as well.
                if not mfa_word:
                    continue
                line_tokens.append({"display": raw_word, "mfa": mfa_word})

            for i, token in enumerate(line_tokens):
                token["line_break"] = i == len(line_tokens) - 1
                token["line"] = line_index
            tokens.extend(line_tokens)

        return tokens

    def _clean_lyrics(self, lyrics: str) -> str:
        """Normalize the lyrics exactly as MFA will see them."""
        return " ".join(token["mfa"] for token in self._lyrics_tokens(lyrics))

    def _flatten_lyrics(self, lyrics: str) -> list[dict]:
        """Ordered tokens of the original lyrics, marking the last word of
        each line (used to decide where to break lines in UltraStar)."""
        return self._lyrics_tokens(lyrics)

    # ------------------------------------------------------------------- MFA
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

        self._prepare_mfa_input(audio_path, lyrics, input_dir)
        self._run_mfa(
            input_dir, output_dir, acoustic_model, dictionary, g2p
        )

        tg_path = self._find_textgrid(output_dir)

        whisper_bounds = None
        if self.whisper_model and self._needs_whisper_check(
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
