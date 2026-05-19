import src.core.config as config
import os
import shutil
import json
from pathlib import Path
from collections import Counter
import re
from difflib import SequenceMatcher

MIDI_MIN = 40       # E2 — lowest realistic singing note
MIDI_MAX = 84       # C6 — highest realistic singing note
MIDI_FALLBACK = 60  # Middle C — used when no valid pitch is detected


class UltraStarFormatter:
    def __init__(self, songs: dict):
        self.songs = songs

    def _seconds_to_beat(self, seconds: float, bpm: float, gap_s: float = 0.0) -> int:
        return round((seconds - gap_s) * bpm * 4 / 60)

    def _get_dominant_midi(self, pitch_data: dict, word_start: float, word_end: float):
        """Return the most frequent MIDI note within [word_start, word_end].
        Only considers notes in the valid vocal range (MIDI_MIN–MIDI_MAX).
        Returns None if no voiced frames are found."""
        times = pitch_data["times"]
        midi_notes = pitch_data["midi_notes"]

        notes = [
            round(midi)
            for t, midi in zip(times, midi_notes)
            if word_start <= t <= word_end
            and MIDI_MIN <= midi <= MIDI_MAX
        ]

        if not notes:
            return None

        return Counter(notes).most_common(1)[0][0]

    def _normalize(self, word: str) -> str:
        """Lowercase + strip punctuation (preserves hyphens) for fuzzy comparison."""
        return re.sub(r"[^\w-]", "", word.lower().strip())

    def _interpolate_timings(self, result: list):
        """Fill None start/end values by linear interpolation between known neighbors."""
        n = len(result)
        i = 0
        while i < n:
            if result[i]["start"] is None:
                j = i
                while j < n and result[j]["start"] is None:
                    j += 1

                prev_end = result[i - 1]["end"] if i > 0 else 0.0
                next_start = result[j]["start"] if j < n else prev_end + 1.0

                count = j - i
                step = (next_start - prev_end) / (count + 1)

                for k in range(count):
                    result[i + k]["start"] = round(prev_end + k * step, 4)
                    result[i + k]["end"] = round(prev_end + (k + 1) * step, 4)
                i = j
            else:
                i += 1

    def align_lyrics_to_words(self, words: list, lyrics_str: str) -> list:
        """
        Align a Whisper word list to the ground-truth lyrics string.

        Parameters
        ----------
        words : list of {"word": str, "start": float, "end": float}
            Raw Whisper transcription word-by-word.
        lyrics_str : str
            Full lyrics text. Empty lines (between strophes) are ignored.
            Line breaks within strophes define UltraStar line breaks.

        Returns
        -------
        list of {"word": str, "start": float, "end": float, "line_break": bool}
            `line_break=True` on the last word of each lyric line.

        How it works
        ------------
        1. Lyrics are split into lines → flat token list with line_break flags.
        2. Both sequences are normalized (lowercase, no punctuation).
        3. difflib.SequenceMatcher aligns the two sequences and emits opcodes:
        - equal   → lyric text  + transcription timing  (1-to-1)
        - replace → most common case:
            1 lyric : N trans  →  merge trans tokens    ("girassol" ← "gira"+"sol")
            N lyric : 1 trans  →  split timing evenly   (rare)
            N lyric : M trans  →  distribute timing proportionally
        - insert  → lyric word missing from transcription → timing interpolated
        - delete  → extra transcription word not in lyrics → discarded
        4. Any None timings are filled by linear interpolation.
        """

        # ── 1. Parse lyrics ────────────────────────────────────────────────────
        lines = [ln.strip()
                 for ln in lyrics_str.strip().splitlines() if ln.strip()]

        lyric_tokens = []
        for line in lines:
            line_words = line.split()
            for i, w in enumerate(line_words):
                lyric_tokens.append({
                    "text":       w,
                    "norm":       self._normalize(w),
                    "line_break": i == len(line_words) - 1,
                })

        # ── 2. Normalize sequences ─────────────────────────────────────────────
        lyric_norms = [t["norm"] for t in lyric_tokens]
        trans_norms = [self._normalize(w["word"]) for w in words]

        # ── 3. Align ───────────────────────────────────────────────────────────
        sm = SequenceMatcher(None, lyric_norms, trans_norms, autojunk=False)
        result = []

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            lyr = lyric_tokens[i1:i2]
            tra = words[j1:j2]

            if tag == "equal":
                for lt, tw in zip(lyr, tra):
                    result.append({
                        "word":       lt["text"],
                        "start":      tw["start"],
                        "end":        tw["end"],
                        "line_break": lt["line_break"],
                    })

            elif tag == "replace":
                if len(lyr) == 1:
                    # Merge all trans tokens into one lyric word
                    # e.g. ["gira", "sol"] → "girassol"
                    result.append({
                        "word":       lyr[0]["text"],
                        "start":      tra[0]["start"] if tra else None,
                        "end":        tra[-1]["end"] if tra else None,
                        "line_break": lyr[0]["line_break"],
                    })

                elif len(tra) == 1:
                    # Split one trans token timing across multiple lyric words
                    duration = tra[0]["end"] - tra[0]["start"]
                    step = duration / len(lyr)
                    for k, lt in enumerate(lyr):
                        result.append({
                            "word":       lt["text"],
                            "start":      round(tra[0]["start"] + k * step, 4),
                            "end":        round(tra[0]["start"] + (k + 1) * step, 4),
                            "line_break": lt["line_break"],
                        })

                else:
                    # N lyric : M trans → distribute timing proportionally
                    t_start = tra[0]["start"] if tra else None
                    t_end = tra[-1]["end"] if tra else None
                    if t_start is not None:
                        duration = t_end - t_start
                        step = duration / len(lyr)
                        for k, lt in enumerate(lyr):
                            result.append({
                                "word":       lt["text"],
                                "start":      round(t_start + k * step, 4),
                                "end":        round(t_start + (k + 1) * step, 4),
                                "line_break": lt["line_break"],
                            })
                    else:
                        for lt in lyr:
                            result.append({"word": lt["text"], "start": None,
                                           "end": None, "line_break": lt["line_break"]})

            elif tag == "insert":
                # Lyric words not found in transcription — timing filled later
                for lt in lyr:
                    result.append({"word": lt["text"], "start": None,
                                   "end": None, "line_break": lt["line_break"]})

            # tag == "delete": extra trans words not in lyrics → discard

        # ── 4. Fill missing timings ────────────────────────────────────────────
        self._interpolate_timings(result)

        return result

    def format_ultrastar_song_folder(self, artist_song: str):
        base_directory = os.path.join(config.OUTPUT_DIR, artist_song)
        song = self.songs[artist_song]
        os.makedirs(base_directory, exist_ok=True)

        # Copy audio and image files
        new_full_track_file = f"{artist_song}{Path(song['full_track']).suffix}"
        shutil.copy(song["full_track"], os.path.join(
            base_directory, new_full_track_file))

        new_instrumental_file = f"{artist_song} [INSTR]{Path(song['instrumental']).suffix}"
        shutil.copy(song["instrumental"], os.path.join(
            base_directory, new_instrumental_file))

        new_vocals_file = f"{artist_song} [VOC]{Path(song['vocals']).suffix}"
        shutil.copy(song["vocals"], os.path.join(
            base_directory, new_vocals_file))

        new_cover_file = f"{artist_song} [CO]{Path(song['cover']).suffix}"
        shutil.copy(song["cover"], os.path.join(
            base_directory, new_cover_file))

        # Define the final UltraStar file
        ultrastar_file = os.path.join(base_directory, f"{artist_song}.txt")

        # Read relevant data
        with open(song["words"]) as f:
            words = json.load(f)
        with open(song["pitches"]) as f:
            pitches = json.load(f)
        with open(song["beats"]) as f:
            beats = json.load(f)
        with open(song["lyrics"]) as f:
            lyrics = f.read()

        bpm = beats["bpm"]
        gap_s = float(words[0]["start"])
        gap_ms = round(gap_s * 1000)

        file_lines = [
            "#VERSION:1.1.0",
            f"#COMMENT:Generated using Song2UltraStar (https://github.com/faelcorreia/song-2-ultrastar)",
            f"#ARTIST:{song['artist']}",
            f"#TITLE:{song['title']}",
            f"#AUDIO:{new_full_track_file}",
            f"#INSTRUMENTAL:{new_instrumental_file}",
            f"#VOCALS:{new_vocals_file}",
            f"#COVER:{new_cover_file}",
            f"#BPM:{bpm}",
            f"#GAP:{gap_ms}"
        ]

        cursor_beat = 0  # tracks the end beat of the last note to prevent overlaps

        # Align words to original lyrics
        aligned = self.align_lyrics_to_words(words, lyrics)

        for i, word_obj in enumerate(aligned):
            word: str = word_obj["word"]
            w_start = float(word_obj["start"])
            w_end = float(word_obj["end"])

            midi = self._get_dominant_midi(pitches, w_start, w_end)
            if midi is None:
                midi = MIDI_FALLBACK

            start_beat = self._seconds_to_beat(w_start, bpm, gap_s)
            end_beat = self._seconds_to_beat(w_end, bpm, gap_s)

            # Prevent overlapping notes: push start forward if needed
            if start_beat < cursor_beat:
                start_beat = cursor_beat

            duration = max(1, end_beat - start_beat)

            parsed_word = word if word.startswith("-") else f" {word}"
            file_lines.append(
                f": {start_beat} {duration} {midi} {parsed_word}")
            cursor_beat = start_beat + duration

            if word_obj["line_break"]:
                file_lines.append(f"- {cursor_beat}")

        file_lines.append("E")

        with open(ultrastar_file, "w") as f:
            f.write("\n".join(file_lines))

        song["beats"].unlink()
        song["instrumental"].unlink()
        song["pitches"].unlink()
        song["vocals"].unlink()
        song["words"].unlink()
