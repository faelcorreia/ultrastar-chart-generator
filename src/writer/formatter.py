"""UltraStar song folder writer."""
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import src.core.config as config
from src.model.lyrics import Lyrics
from src.writer import gaps, pitch, timing
from src.writer.constants import (
    BPM_HEADER_MIN, LANGUAGE_NAMES, MIN_NOTE_DURATION_S,
)
from src.writer.lyrics_input import lyrics_to_syllable_dicts


class UltraStarFormatter:
    def __init__(self, songs: dict):
        self.songs = songs

    # Thin delegates keep the old method names working.
    _seconds_to_beat = staticmethod(timing.seconds_to_beat)
    _interpolate_timings = staticmethod(timing.interpolate_timings)
    _format_number = staticmethod(timing.format_number)
    _median_smooth = staticmethod(pitch.median_smooth)
    _pitch_frames = staticmethod(pitch.pitch_frames)
    _segment_pitch = staticmethod(pitch.segment_pitch)
    _syllable_notes = staticmethod(pitch.syllable_notes)
    _voiced_runs = staticmethod(pitch.voiced_runs)
    _correct_octave_errors = staticmethod(pitch.correct_octave_errors)
    _fill_unvoiced_syllables = staticmethod(pitch.fill_unvoiced_syllables)
    _merge_syllable_notes = staticmethod(pitch.merge_syllable_notes)
    _redistribute_voiced_gaps = staticmethod(gaps.redistribute_voiced_gaps)

    def format_ultrastar_song_folder(self, artist_song: str,
                                     lyrics: Lyrics = None):
        """Write the song folder.

        `lyrics` (timed, in seconds) takes precedence; otherwise
        song["syllables"] is a Lyrics object or a path to the syllables JSON.
        """
        base_directory = os.path.join(
            getattr(self, "output_dir", config.OUTPUT_DIR), artist_song)
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

        # The cover is optional: the song works without one.
        new_cover_file = None
        if song.get("cover"):
            new_cover_file = f"{artist_song} [CO]{Path(song['cover']).suffix}"
            shutil.copy(song["cover"], os.path.join(
                base_directory, new_cover_file))

        # Define the final UltraStar file
        ultrastar_file = os.path.join(base_directory, f"{artist_song}.txt")

        # Read relevant data
        if lyrics is None and isinstance(song.get("syllables"), Lyrics):
            lyrics = song["syllables"]
        if lyrics is not None:
            syllables = lyrics_to_syllable_dicts(lyrics)
        else:
            with open(song["syllables"], encoding="utf-8") as f:
                syllables = json.load(f)
        with open(song["pitches"], encoding="utf-8") as f:
            pitches = json.load(f)
        with open(song["beats"], encoding="utf-8") as f:
            beats = json.load(f)

        timing.interpolate_timings(syllables)

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
        adjusted_gaps = gaps.redistribute_voiced_gaps(entries, pitches)
        for entry in entries:
            entry["notes"] = pitch.syllable_notes(
                pitches, entry["start"], entry["end"]
            )

        fallback_notes = pitch.fill_unvoiced_syllables(entries)

        # Correct octaves after fallback filling so pitches inherited from
        # neighbors are corrected too.
        all_notes = [note for entry in entries for note in entry["notes"]]
        octave_corrections = pitch.correct_octave_errors(all_notes)
        merged_holds = pitch.merge_syllable_notes(entries)

        file_lines = [
            "#VERSION:1.1.0",
            f"#ARTIST:{song['artist']}",
            f"#TITLE:{song['title']}",
            f"#MP3:{new_full_track_file}",
            f"#AUDIO:{new_full_track_file}",
            f"#INSTRUMENTAL:{new_instrumental_file}",
            f"#VOCALS:{new_vocals_file}",
            *([f"#COVER:{new_cover_file}"] if new_cover_file else []),
            f"#BPM:{timing.format_number(bpm)}",
            f"#GAP:{gap_ms}",
        ]
        if gap_s > 0:
            # #PREVIEWSTART is expressed in seconds (v1 specification).
            file_lines.append(f"#PREVIEWSTART:{timing.format_number(gap_s)}")
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
                    "start": timing.seconds_to_beat(
                        note["start"], bpm, gap_s
                    ),
                    "end": timing.seconds_to_beat(note["end"], bpm, gap_s),
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
            f"BPM {timing.format_number(detected_bpm)} (x{bpm_multiplier} in "
            f"the file) | {adjusted_gaps} voiced gap(s) redistributed | "
            f"{octave_corrections} octave correction(s) | "
            f"{raised_to_min} note(s) raised to the minimum duration | "
            f"{fallback_notes} unvoiced syllable(s) filled from neighbors | "
            f"{merged_holds} held syllable(s) merged into a single note"
        )
