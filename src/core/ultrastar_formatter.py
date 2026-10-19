import src.core.config as config
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

MIDI_MIN = 40       # E2 — lowest realistic singing note
MIDI_MAX = 84       # C6 — highest realistic singing note
MIDI_FALLBACK = 60  # Middle C — used when no valid pitch is detected

# UltraStar converts each file beat with ``15 / #BPM`` seconds (USDX
# multiplies the header BPM by 4 internally). Raising the header BPM — and
# therefore every beat position — by the same power of two keeps the timing
# identical while making the integer beat grid finer. The header alone gives
# ~100 ms on a 150 BPM song; a header of ~300 BPM gives ~50 ms, which matches
# the precision of the alignment without creating unhittable 25 ms notes.
# This mirrors the approach used by UltraSinger.
BPM_HEADER_MIN = 250.0

PITCH_CONFIDENCE_MIN = 0.10
MEDIAN_WINDOW_FRAMES = 5      # ~50 ms with pyin's 10 ms hop
MIN_STABLE_FRAMES = 6         # a pitch change must last ~60 ms
MIN_NOTE_DURATION_S = 0.08
EDGE_TRIM_S = 0.03
OCTAVE_JUMP_SEMITONES = 7
OCTAVE_TOLERANCE_SEMITONES = 2

# MFA's speech models compress sustained vowels: a word often ends when its
# vowel starts, leaving the voiced tail unlabeled. These thresholds decide
# whether that tail belongs to the previous syllable or to a following one
# that the aligner squeezed.
VOICED_GAP_MIN_S = 0.1
MAX_VOICED_GAP_S = 2.0
SHORT_SYLLABLE_S = 0.2
GAP_TOUCH_TOLERANCE_S = 0.08

# #LANGUAGE expects ISO 639-2 English names.
LANGUAGE_NAMES = {
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "en": "English",
    "en-us": "English",
    "es": "Spanish",
    "es-419": "Spanish",
    "es-es": "Spanish",
}


class UltraStarFormatter:
    def __init__(self, songs: dict):
        self.songs = songs

    # --------------------------------------------------------------- timing
    def _seconds_to_beat(self, seconds: float, bpm: float, gap_s: float = 0.0) -> int:
        return round((seconds - gap_s) * bpm * 4 / 60)

    def _interpolate_timings(self, result: list) -> None:
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
                # Guard against overlapping/inverted MFA boundaries.
                span = max(0.01 * count, next_start - prev_end)
                step = span / (count + 1)

                for k in range(count):
                    result[i + k]["start"] = round(prev_end + k * step, 4)
                    result[i + k]["end"] = round(prev_end + (k + 1) * step, 4)
                i = j
            else:
                i += 1

    # ---------------------------------------------------------------- pitch
    @staticmethod
    def _median_smooth(values: np.ndarray, window: int) -> np.ndarray:
        if len(values) < 3 or window < 3:
            return values.astype(float)
        half = window // 2
        padded = np.pad(values.astype(float), half, mode="edge")
        return np.array(
            [np.median(padded[i:i + window]) for i in range(len(values))]
        )

    def _pitch_frames(
        self, pitch_data: dict, start: float, end: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (times, midi) for the valid/voiced frames in a window."""
        times = np.asarray(pitch_data["times"], dtype=float)
        midi = np.asarray(pitch_data["midi_notes"], dtype=float)

        lo = int(np.searchsorted(times, start, side="left"))
        hi = int(np.searchsorted(times, end, side="right"))
        if hi <= lo:
            return np.array([]), np.array([])

        times = times[lo:hi]
        midi = midi[lo:hi]

        valid = (midi >= MIDI_MIN) & (midi <= MIDI_MAX)
        probabilities = pitch_data.get("voiced_prob")
        if probabilities is not None and len(probabilities) >= hi:
            probabilities = np.asarray(probabilities[lo:hi], dtype=float)
            confident = valid & (probabilities >= PITCH_CONFIDENCE_MIN)
            # Keep only confident frames. If the whole window is unvoiced/low
            # confidence, return no frames so the caller can fall back to a
            # neighboring syllable instead of using noise as pitch.
            if confident.sum() >= 2:
                valid = confident
            else:
                return np.array([]), np.array([])

        return times[valid], midi[valid]

    def _segment_pitch(self, times: np.ndarray, midi: np.ndarray) -> list[dict]:
        """Split a pitch contour into stable notes (melisma).

        Returns a list of {start, end, midi} in seconds, covering the whole
        syllable. Short noisy excursions (vibrato, scoops) are absorbed by the
        neighboring note.
        """
        if len(midi) == 0:
            return []
        if len(midi) < 4:
            return [{
                "start": float(times[0]),
                "end": float(times[-1]),
                "midi": float(np.median(midi)),
            }]

        hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
        smoothed = self._median_smooth(midi, MEDIAN_WINDOW_FRAMES)

        # Hysteresis: only accept a new pitch after it has persisted for a
        # few frames. This keeps vibrato from splitting a single note.
        segments: list[dict] = []
        seg_start = 0
        current_rounded = int(round(float(np.median(smoothed[:min(5, len(smoothed))]))))
        candidate_rounded = None
        candidate_start = 0
        candidate_count = 0

        for i in range(1, len(smoothed)):
            rounded = int(round(float(smoothed[i])))
            if rounded == current_rounded:
                candidate_rounded = None
                candidate_count = 0
                continue

            if candidate_rounded is None or rounded != candidate_rounded:
                candidate_rounded = rounded
                candidate_start = i
                candidate_count = 1
            else:
                candidate_count += 1

            if candidate_count >= MIN_STABLE_FRAMES:
                if candidate_start > seg_start:
                    segments.append({
                        "start_idx": seg_start,
                        "end_idx": candidate_start,
                        "midi": float(np.median(smoothed[seg_start:candidate_start])),
                    })
                seg_start = candidate_start
                current_rounded = int(round(float(np.median(
                    smoothed[candidate_start:i + 1]
                ))))
                candidate_rounded = None
                candidate_count = 0

        segments.append({
            "start_idx": seg_start,
            "end_idx": len(smoothed),
            "midi": float(np.median(smoothed[seg_start:])),
        })

        # Absorb short notes (noise/transition) into the nearest-pitch neighbor.
        min_frames = max(2, int(round(MIN_NOTE_DURATION_S / hop)))
        changed = True
        while changed and len(segments) > 1:
            changed = False
            for idx, seg in enumerate(segments):
                if seg["end_idx"] - seg["start_idx"] >= min_frames:
                    continue

                neighbors = []
                if idx > 0:
                    neighbors.append(idx - 1)
                if idx < len(segments) - 1:
                    neighbors.append(idx + 1)
                if not neighbors:
                    break

                target = min(
                    neighbors,
                    key=lambda j: (
                        abs(segments[j]["midi"] - seg["midi"]),
                        -(segments[j]["end_idx"] - segments[j]["start_idx"]),
                    ),
                )

                if target < idx:
                    segments[target]["end_idx"] = seg["end_idx"]
                else:
                    segments[target]["start_idx"] = seg["start_idx"]
                merged_start = segments[target]["start_idx"]
                merged_end = segments[target]["end_idx"]
                segments[target]["midi"] = float(np.median(
                    smoothed[merged_start:merged_end]
                ))
                del segments[idx]
                changed = True
                break

        # Merge neighboring segments that ended up on the same semitone.
        merged: list[dict] = []
        for seg in segments:
            if merged and round(merged[-1]["midi"]) == round(seg["midi"]):
                merged[-1]["end_idx"] = seg["end_idx"]
                merged[-1]["midi"] = float(np.median(
                    smoothed[merged[-1]["start_idx"]:merged[-1]["end_idx"]]
                ))
            else:
                merged.append(seg)

        # Convert frame indices to absolute times. The first note starts at the
        # beginning of the syllable and the last one ends at its end, ensuring
        # full coverage.
        boundaries = [float(times[merged[0]["start_idx"]])]
        for seg in merged[1:]:
            boundaries.append(float(times[seg["start_idx"]]))
        boundaries.append(float(times[-1]) + hop)

        notes = []
        for i, seg in enumerate(merged):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end <= start:
                continue
            notes.append({
                "start": start,
                "end": end,
                "midi": seg["midi"],
            })

        return notes or [{
            "start": float(times[0]),
            "end": float(times[-1]) + hop,
            "midi": float(np.median(midi)),
        }]

    def _syllable_notes(
        self, pitch_data: dict, start: float, end: float
    ) -> list[dict]:
        """Return the pitch notes for one syllable (possibly empty)."""
        if end <= start:
            end = start + 0.01

        times, midi = self._pitch_frames(pitch_data, start, end)
        if len(midi) == 0:
            return []

        # Trim attack/transition frames at the edges when there is enough
        # material. Only used when the syllable has a single note; melisma
        # segments need the full contour.
        hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
        trim = int(round(EDGE_TRIM_S / max(hop, 1e-6)))
        if len(midi) > 2 * trim + 2 and trim > 0:
            trimmed_midi = midi[trim:-trim]
        else:
            trimmed_midi = midi

        if len(midi) < 4:
            notes = [{
                "start": start,
                "end": end,
                "midi": float(np.median(trimmed_midi)),
            }]
        else:
            notes = self._segment_pitch(times, midi)
            if not notes:
                notes = [{
                    "start": start,
                    "end": end,
                    "midi": float(np.median(trimmed_midi)),
                }]
            elif len(notes) == 1:
                # Sustained note: the median without the edges keeps the
                # initial scoop from pulling the pitch.
                notes[0]["midi"] = float(np.median(trimmed_midi))

        # Ensure the whole syllable is covered.
        notes[0]["start"] = start
        notes[-1]["end"] = end
        for note in notes:
            note["start"] = max(start, min(end, note["start"]))
            note["end"] = max(note["start"] + 0.001, min(end, note["end"]))

        return notes

    def _redistribute_voiced_gaps(
        self, entries: list[dict], pitch_data: dict
    ) -> int:
        """Give voiced tails left between syllables to the right syllable.

        MFA's speech models compress sustained vowels, so a word often ends
        when its vowel starts and the rest of the hold stays unlabeled. When
        the following syllable was squeezed (e.g. a final stressed syllable
        such as "mar"), the hold belongs to it, not to the previous word.
        Otherwise the hold extends the previous syllable, and when both are
        short the gap is split between them.
        """
        adjusted = 0

        for previous, following in zip(entries, entries[1:]):
            gap_start = previous["end"]
            gap_end = following["start"]
            gap = gap_end - gap_start
            if gap < VOICED_GAP_MIN_S:
                continue

            runs = self._voiced_runs(pitch_data, gap_start, gap_end)
            if not runs:
                continue

            following_short = (
                following["end"] - following["start"] < SHORT_SYLLABLE_S
            )

            if gap > MAX_VOICED_GAP_S:
                # Long gaps separate phrases/instrumental sections and must
                # never be folded into one note. The exception is a short
                # first syllable right after a sung phrase: the aligner left
                # the beginning of that phrase unlabeled.
                adjacent = [
                    run for run in runs
                    if gap_end - run[1] <= GAP_TOUCH_TOLERANCE_S
                ]
                if following_short and adjacent:
                    run_start, run_end = max(
                        adjacent, key=lambda run: run[1] - run[0]
                    )
                    run_duration = run_end - run_start
                    # Only a short voiced onset, not a phrase that fills the
                    # whole long gap.
                    if 0.08 <= run_duration < 0.5 * gap:
                        following["start"] = run_start
                        adjusted += 1
                continue

            run_start, run_end = max(runs, key=lambda run: run[1] - run[0])
            if run_end - run_start < 0.08:
                continue

            # Ignore voiced islands that do not touch either syllable: they
            # are background noise inside a breath, not a sustained vowel.
            touches_previous = run_start - gap_start <= GAP_TOUCH_TOLERANCE_S
            touches_following = gap_end - run_end <= GAP_TOUCH_TOLERANCE_S
            if not (touches_previous or touches_following):
                continue

            previous_short = (
                previous["end"] - previous["start"] < SHORT_SYLLABLE_S
            )

            if following_short and not previous_short:
                # The aligner put the following word right after the hold.
                following["start"] = run_start
            elif previous_short and following_short:
                midpoint = max(
                    gap_start, min(gap_end, (run_start + run_end) / 2)
                )
                previous["end"] = midpoint
                following["start"] = midpoint
            else:
                previous["end"] = min(run_end, gap_end)

            adjusted += 1

        return adjusted

    def _voiced_runs(
        self, pitch_data: dict, start: float, end: float
    ) -> list[tuple[float, float]]:
        """Contiguous voiced runs (in seconds) inside a time window."""
        times, midi = self._pitch_frames(pitch_data, start, end)
        if len(midi) < 5:
            return []

        hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
        runs: list[tuple[float, float]] = []
        run_start = 0

        for i in range(1, len(times)):
            if times[i] - times[i - 1] > 3 * hop:
                runs.append((float(times[run_start]),
                             float(times[i - 1]) + hop))
                run_start = i
        runs.append((float(times[run_start]), float(times[-1]) + hop))

        return runs

    def _correct_octave_errors(self, notes: list[dict]) -> int:
        """Move octave-jumped notes to the register of their neighbors.

        Two passes are used:

        1. A longer context window catches short runs where the tracker locked
           onto the wrong octave, as long as the surrounding register matches
           the song's overall register. Genuine extended register changes (a
           long bridge, for example) are left alone because their context moves
           with them.
        2. A short window catches isolated octave spikes.
        """
        corrections = 0
        pitches = [note["midi"] for note in notes]
        if not pitches:
            return 0

        global_median = float(np.median(pitches))

        # Pass 1: wrong-octave runs.
        for i, note in enumerate(notes):
            lo = max(0, i - 10)
            hi = min(len(notes), i + 11)
            context = [pitches[j] for j in range(lo, hi) if j != i]
            if len(context) < 6:
                continue

            median = float(np.median(context))
            if abs(median - global_median) > 7:
                continue

            difference = note["midi"] - median
            if abs(difference) < 11:
                continue

            best = None
            for octaves in (-2, -1, 1, 2):
                candidate = note["midi"] + 12 * octaves
                if abs(candidate - median) >= abs(difference):
                    continue
                if best is None or abs(candidate - median) < abs(best - median):
                    best = candidate

            if best is not None and abs(best - median) <= 6:
                note["midi"] = best
                pitches[i] = best
                corrections += 1

        # Pass 2: isolated spikes.
        for i, note in enumerate(notes):
            lo = max(0, i - 3)
            hi = min(len(notes), i + 4)
            context = [pitches[j] for j in range(lo, hi) if j != i]
            if len(context) < 3:
                continue

            median = float(np.median(context))
            difference = note["midi"] - median
            if abs(difference) <= OCTAVE_JUMP_SEMITONES - 1:
                continue

            best = None
            for octaves in (-2, -1, 1, 2):
                candidate = note["midi"] + 12 * octaves
                if abs(candidate - median) >= abs(difference):
                    continue
                if best is None or abs(candidate - median) < abs(best - median):
                    best = candidate

            if best is not None and abs(best - median) <= OCTAVE_TOLERANCE_SEMITONES:
                note["midi"] = best
                pitches[i] = best
                corrections += 1

        return corrections

    def _fill_unvoiced_syllables(self, syllables: list[dict]) -> int:
        """Give unvoiced syllables the pitch of the closest voiced neighbor."""
        filled = 0

        voiced_pitches = [
            note["midi"] for entry in syllables for note in entry["notes"]
        ]
        register = (
            float(np.median(voiced_pitches))
            if voiced_pitches else float(MIDI_FALLBACK)
        )

        def neighbor_register(midi: float) -> float:
            """Keep a neighbor pitch in the song's main octave register."""
            if abs(midi - register) < 11:
                return midi
            candidates = [midi + 12 * octave for octave in (-2, -1, 1, 2)]
            best = min(candidates, key=lambda value: abs(value - register))
            if abs(best - register) <= 7:
                return best
            return midi

        next_pitch = None
        for syl in reversed(syllables):
            if syl["notes"]:
                next_pitch = syl["notes"][-1]["midi"]
            else:
                syl["_next_pitch"] = next_pitch

        previous_pitch = None
        for syl in syllables:
            if syl["notes"]:
                previous_pitch = syl["notes"][-1]["midi"]
                continue

            midi = previous_pitch
            if midi is None:
                midi = syl.get("_next_pitch")
            if midi is None:
                midi = MIDI_FALLBACK
            midi = neighbor_register(float(midi))

            syl["notes"] = [{
                "start": syl["start"],
                "end": syl["end"],
                "midi": float(midi),
            }]
            syl["fallback"] = True
            filled += 1

        for syl in syllables:
            syl.pop("_next_pitch", None)

        return filled

    def _merge_syllable_notes(self, entries: list[dict]) -> int:
        """Keep a single note per syllable, absorbing holds into that note.

        UltraStar marks a held syllable with extra ``~`` lines. When the chart
        should not use them, the syllable becomes one note spanning its whole
        duration, keeping the pitch of the note the singer holds the longest
        (the breath/sustain stays inside the syllable).
        """
        merged = 0

        for entry in entries:
            notes = entry["notes"]
            if len(notes) < 2:
                continue

            longest = max(notes, key=lambda note: note["end"] - note["start"])
            entry["notes"] = [{
                "start": notes[0]["start"],
                "end": notes[-1]["end"],
                "midi": longest["midi"],
            }]
            merged += 1

        return merged

    # ------------------------------------------------------------- output
    def format_ultrastar_song_folder(self, artist_song: str):
        base_directory = os.path.join(config.OUTPUT_DIR, artist_song)
        song = self.songs[artist_song]
        os.makedirs(base_directory, exist_ok=True)

        # Transcode the input audio to a real MP3 regardless of its original
        # container/codec (e.g., YouTube downloads saved with a ".mp3"
        # extension but actually containing AAC/MP4 data, which UltraStar
        # Play's audio loader rejects).
        new_full_track_file = f"{artist_song}.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(song["full_track"]),
             "-codec:a", "libmp3lame", "-b:a", "192k",
             os.path.join(base_directory, new_full_track_file)],
            check=True,
        )

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
        with open(song["syllables"], encoding="utf-8") as f:
            syllables = json.load(f)
        with open(song["pitches"], encoding="utf-8") as f:
            pitches = json.load(f)
        with open(song["beats"], encoding="utf-8") as f:
            beats = json.load(f)

        self._interpolate_timings(syllables)

        detected_bpm = float(beats.get("bpm", 0) or 0)
        if not math.isfinite(detected_bpm) or detected_bpm <= 0:
            detected_bpm = 120.0
        bpm_multiplier = 1
        while detected_bpm * bpm_multiplier < BPM_HEADER_MIN:
            bpm_multiplier *= 2
        bpm = detected_bpm * bpm_multiplier

        gap_s = float(syllables[0]["start"]) if syllables else 0.0
        if not math.isfinite(gap_s):
            gap_s = 0.0
        gap_ms = round(gap_s * 1000)

        # Generate the notes for every syllable before writing the file, so
        # octave errors can be corrected and unvoiced syllables can be filled.
        entries: list[dict] = []
        for syl_obj in syllables:
            entries.append({
                "syllable": syl_obj["syllable"],
                "start": float(syl_obj["start"]),
                "end": float(syl_obj["end"]),
                "notes": [],
                "line_break": syl_obj.get("line_break", False),
                "first_syllable": syl_obj.get("first_syllable", True),
            })

        # Move voiced tails that MFA left unlabeled to the syllable that was
        # actually held before generating the notes.
        adjusted_gaps = self._redistribute_voiced_gaps(entries, pitches)
        for entry in entries:
            entry["notes"] = self._syllable_notes(
                pitches, entry["start"], entry["end"]
            )

        fallback_notes = self._fill_unvoiced_syllables(entries)

        # Correct octaves after fallback filling so pitches inherited from
        # neighbors are corrected too.
        all_notes = [note for entry in entries for note in entry["notes"]]
        octave_corrections = self._correct_octave_errors(all_notes)
        merged_holds = self._merge_syllable_notes(entries)

        file_lines = [
            "#VERSION:1.1.0",
            f"#ARTIST:{song['artist']}",
            f"#TITLE:{song['title']}",
            f"#MP3:{new_full_track_file}",
            f"#AUDIO:{new_full_track_file}",
            f"#INSTRUMENTAL:{new_instrumental_file}",
            f"#VOCALS:{new_vocals_file}",
            f"#COVER:{new_cover_file}",
            f"#BPM:{self._format_number(bpm)}",
            f"#GAP:{gap_ms}",
        ]
        if gap_s > 0:
            # #PREVIEWSTART is expressed in seconds (v1 specification).
            file_lines.append(f"#PREVIEWSTART:{self._format_number(gap_s)}")
        language = (song.get("language") or "").lower().replace("_", "-")
        if language:
            base_language = language.split("-", 1)[0]
            display_language = LANGUAGE_NAMES.get(
                language, LANGUAGE_NAMES.get(base_language, language)
            )
            file_lines.append(f"#LANGUAGE:{display_language}")

        # Convert every note to integer beats first. Minimum-duration
        # adjustments are capped by the next note's start, so extending one
        # note can never push the rest of the song forward (no cumulative
        # drift).
        flat_notes: list[dict] = []
        for entry_idx, entry in enumerate(entries):
            for note_idx, note in enumerate(entry["notes"]):
                flat_notes.append({
                    "entry_idx": entry_idx,
                    "note_idx": note_idx,
                    "start": self._seconds_to_beat(
                        note["start"], bpm, gap_s
                    ),
                    "end": self._seconds_to_beat(note["end"], bpm, gap_s),
                    # The UltraStar format stores pitch as semitones relative
                    # to C4 (0 = C4), while pyin gives MIDI note numbers
                    # (60 = C4).
                    "pitch": int(round(note["midi"])) - 60,
                })

        previous_end = 0
        for note in flat_notes:
            if note["start"] < previous_end:
                note["start"] = previous_end
            if note["end"] <= note["start"]:
                note["end"] = note["start"] + 1
            previous_end = note["end"]

        min_note_beats = max(
            1, math.ceil(MIN_NOTE_DURATION_S * bpm * 4 / 60)
        )
        raised_to_min = 0
        for i, note in enumerate(flat_notes):
            if note["end"] - note["start"] >= min_note_beats:
                continue

            target_end = note["start"] + min_note_beats
            if i + 1 < len(flat_notes):
                target_end = min(target_end, flat_notes[i + 1]["start"])
            if target_end > note["end"]:
                note["end"] = target_end
                raised_to_min += 1

        note_count = len(flat_notes)
        cursor_beat = 0
        flat_idx = 0

        for entry in entries:
            for note_idx in range(len(entry["notes"])):
                note = flat_notes[flat_idx]
                flat_idx += 1

                if note_idx == 0:
                    syllable = entry["syllable"] or "~"
                    parsed_syllable = (
                        f" {syllable}" if entry["first_syllable"] else syllable
                    )
                else:
                    # Continuation of the same syllable (melisma).
                    parsed_syllable = "~"

                duration = note["end"] - note["start"]
                file_lines.append(
                    f": {note['start']} {duration} {note['pitch']} "
                    f"{parsed_syllable}"
                )
                cursor_beat = note["end"]

            if entry["line_break"]:
                file_lines.append(f"- {cursor_beat}")

        file_lines.append("E")

        with open(ultrastar_file, "w", encoding="utf-8") as f:
            f.write("\n".join(file_lines))

        print(
            f"[UltraStar] {len(entries)} syllables -> {note_count} notes | "
            f"BPM {self._format_number(detected_bpm)} (x{bpm_multiplier} in "
            f"the file) | {adjusted_gaps} voiced gap(s) redistributed | "
            f"{octave_corrections} octave correction(s) | "
            f"{raised_to_min} note(s) raised to the minimum duration | "
            f"{fallback_notes} unvoiced syllable(s) filled from neighbors | "
            f"{merged_holds} held syllable(s) merged into a single note"
        )

    @staticmethod
    def _format_number(value: float) -> str:
        """Format a number without trailing zeros (and without 'e' notation)."""
        return f"{value:.6f}".rstrip("0").rstrip(".")
