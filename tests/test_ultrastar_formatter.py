import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

import src.core.config as config  # noqa: E402
from src.core.ultrastar_formatter import UltraStarFormatter  # noqa: E402


class SegmentPitchTest(unittest.TestCase):
    def setUp(self):
        self.formatter = UltraStarFormatter({})
        self.times = np.arange(0.0, 2.0, 0.01)

    def test_keeps_vibrato_as_one_note(self):
        midi = 60.0 + 0.6 * np.sin(2 * np.pi * 5 * self.times)
        notes = self.formatter._segment_pitch(self.times, midi)

        self.assertEqual(len(notes), 1)
        self.assertAlmostEqual(notes[0]["midi"], 60.0, delta=1.0)

    def test_splits_melisma_into_stable_notes(self):
        midi = np.concatenate([
            np.full(80, 60.0),
            np.full(80, 62.0),
            np.full(40, 60.0),
        ])
        notes = self.formatter._segment_pitch(
            self.times[:len(midi)], midi
        )

        self.assertEqual(len(notes), 3)
        self.assertAlmostEqual(notes[0]["midi"], 60.0, delta=1.0)
        self.assertAlmostEqual(notes[1]["midi"], 62.0, delta=1.0)
        self.assertAlmostEqual(notes[2]["midi"], 60.0, delta=1.0)

    def test_note_boundaries_are_contiguous(self):
        midi = np.concatenate([np.full(50, 57.0), np.full(50, 62.0)])
        notes = self.formatter._segment_pitch(self.times[:len(midi)], midi)

        for previous, current in zip(notes, notes[1:]):
            self.assertAlmostEqual(previous["end"], current["start"], places=2)


class OctaveCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.formatter = UltraStarFormatter({})

    def test_corrects_isolated_octave_spike(self):
        notes = [{"midi": 60.0} for _ in range(8)]
        notes[4]["midi"] = 72.0

        corrections = self.formatter._correct_octave_errors(notes)

        self.assertEqual(corrections, 1)
        self.assertEqual(notes[4]["midi"], 60.0)

    def test_corrects_two_octave_spike(self):
        notes = [{"midi": 60.0} for _ in range(8)]
        notes[4]["midi"] = 36.0

        self.formatter._correct_octave_errors(notes)

        self.assertEqual(notes[4]["midi"], 60.0)

    def test_keeps_sustained_register_change(self):
        notes = (
            [{"midi": 60.0} for _ in range(10)]
            + [{"midi": 72.0} for _ in range(20)]
        )

        self.formatter._correct_octave_errors(notes)

        self.assertTrue(all(note["midi"] == 72.0 for note in notes[10:]))


class FillUnvoicedTest(unittest.TestCase):
    def test_uses_previous_pitch_then_next_pitch(self):
        formatter = UltraStarFormatter({})
        entries = [
            {"start": 0.0, "end": 0.5,
             "notes": [{"start": 0.0, "end": 0.5, "midi": 57.0}]},
            {"start": 0.5, "end": 1.0, "notes": []},
            {"start": 1.0, "end": 1.5,
             "notes": [{"start": 1.0, "end": 1.5, "midi": 62.0}]},
            {"start": 1.5, "end": 2.0, "notes": []},
            {"start": 2.0, "end": 2.5, "notes": []},
        ]

        filled = formatter._fill_unvoiced_syllables(entries)

        self.assertEqual(filled, 3)
        self.assertEqual(entries[1]["notes"][0]["midi"], 57.0)
        self.assertEqual(entries[3]["notes"][0]["midi"], 62.0)
        self.assertEqual(entries[4]["notes"][0]["midi"], 62.0)

    def test_fallback_stays_in_the_main_register(self):
        formatter = UltraStarFormatter({})
        entries = [
            {"start": float(i), "end": float(i + 1),
             "notes": [{"start": float(i), "end": float(i + 1),
                        "midi": pitch}]}
            for i, pitch in enumerate([59.0, 59.0, 59.0, 59.0, 43.0])
        ]
        entries.append({
            "start": 5.0, "end": 6.0, "notes": [],
        })

        formatter._fill_unvoiced_syllables(entries)

        # 43 is an octave error in a 59-register song; the fallback is moved
        # to the nearest octave (55) instead of copying 43.
        self.assertEqual(entries[-1]["notes"][0]["midi"], 55.0)


class RedistributeVoicedGapsTest(unittest.TestCase):
    def setUp(self):
        self.formatter = UltraStarFormatter({})
        times = np.arange(0.0, 3.0, 0.01)
        self.pitch = {
            "times": times,
            "midi_notes": np.full(len(times), 60.0),
        }

    @staticmethod
    def _entries(previous, following):
        return [
            {"start": previous[0], "end": previous[1], "notes": []},
            {"start": following[0], "end": following[1], "notes": []},
        ]

    def test_hold_goes_to_the_squeezed_following_syllable(self):
        entries = self._entries((0.0, 1.0), (2.0, 2.1))

        adjusted = self.formatter._redistribute_voiced_gaps(
            entries, self.pitch
        )

        self.assertEqual(adjusted, 1)
        self.assertEqual(entries[0]["end"], 1.0)
        self.assertEqual(entries[1]["start"], 1.0)

    def test_hold_extends_previous_when_following_is_not_squeezed(self):
        entries = self._entries((0.0, 1.0), (2.0, 2.5))

        adjusted = self.formatter._redistribute_voiced_gaps(
            entries, self.pitch
        )

        self.assertEqual(adjusted, 1)
        self.assertEqual(entries[0]["end"], 2.0)
        self.assertEqual(entries[1]["start"], 2.0)

    def test_hold_is_split_when_both_syllables_are_squeezed(self):
        entries = self._entries((0.9, 1.0), (2.0, 2.1))

        adjusted = self.formatter._redistribute_voiced_gaps(
            entries, self.pitch
        )

        self.assertEqual(adjusted, 1)
        self.assertAlmostEqual(entries[0]["end"], 1.5, places=2)
        self.assertEqual(entries[0]["end"], entries[1]["start"])

    def test_does_not_span_long_phrase_gaps(self):
        entries = self._entries((0.0, 0.5), (3.0, 3.1))

        adjusted = self.formatter._redistribute_voiced_gaps(
            entries, self.pitch
        )

        self.assertEqual(adjusted, 0)
        self.assertEqual(entries[0]["end"], 0.5)
        self.assertEqual(entries[1]["start"], 3.0)

    def test_short_first_syllable_after_long_gap_starts_with_the_phrase(self):
        entries = self._entries((0.0, 0.5), (5.0, 5.1))
        times = np.arange(0.0, 6.0, 0.01)
        midi = np.zeros(len(times))
        phrase = (times >= 4.5) & (times <= 5.0)
        midi[phrase] = 60.0
        pitch = {"times": times, "midi_notes": midi}

        adjusted = self.formatter._redistribute_voiced_gaps(entries, pitch)

        self.assertEqual(adjusted, 1)
        self.assertEqual(entries[0]["end"], 0.5)
        self.assertAlmostEqual(entries[1]["start"], 4.5, places=2)

    def test_ignores_voiced_island_away_from_both_syllables(self):
        entries = self._entries((0.0, 1.0), (2.0, 2.5))
        times = np.arange(0.0, 3.0, 0.01)
        midi = np.zeros(len(times))
        island = (times >= 1.4) & (times <= 1.6)
        midi[island] = 60.0
        pitch = {"times": times, "midi_notes": midi}

        adjusted = self.formatter._redistribute_voiced_gaps(entries, pitch)

        self.assertEqual(adjusted, 0)
        self.assertEqual(entries[0]["end"], 1.0)
        self.assertEqual(entries[1]["start"], 2.0)


class MergeSyllableNotesTest(unittest.TestCase):
    def test_merges_hold_into_a_single_note(self):
        formatter = UltraStarFormatter({})
        entries = [{
            "start": 0.0,
            "end": 2.0,
            "notes": [
                {"start": 0.0, "end": 0.5, "midi": 60.0},
                {"start": 0.5, "end": 2.0, "midi": 62.0},
            ],
        }]

        merged = formatter._merge_syllable_notes(entries)

        self.assertEqual(merged, 1)
        self.assertEqual(len(entries[0]["notes"]), 1)
        note = entries[0]["notes"][0]
        self.assertEqual(note["start"], 0.0)
        self.assertEqual(note["end"], 2.0)
        # The longest note is the pitch the singer holds.
        self.assertEqual(note["midi"], 62.0)

    def test_leaves_single_notes_untouched(self):
        formatter = UltraStarFormatter({})
        note = {"start": 0.0, "end": 0.5, "midi": 60.0}
        entries = [{"start": 0.0, "end": 0.5, "notes": [note]}]

        self.assertEqual(formatter._merge_syllable_notes(entries), 0)
        self.assertEqual(entries[0]["notes"], [note])


class FormatNumberTest(unittest.TestCase):
    def test_removes_trailing_zeros(self):
        self.assertEqual(UltraStarFormatter._format_number(590.625), "590.625")
        self.assertEqual(UltraStarFormatter._format_number(14.520000), "14.52")


class SongFolderOutputTest(unittest.TestCase):
    def test_writes_separated_vocals_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song_name = "Artist - Song"

            sources = {
                "full_track": root / "full.mp3",
                "instrumental": root / "instrumental.mp3",
                "vocals": root / "vocals.mp3",
                "cover": root / "cover.jpg",
            }
            for path in sources.values():
                path.write_bytes(b"data:" + path.stem.encode())

            syllables = root / "syllables.json"
            syllables.write_text(json.dumps([{
                "syllable": "la",
                "start": 0.5,
                "end": 0.9,
                "line_break": True,
                "first_syllable": True,
            }]), encoding="utf-8")

            pitches = root / "pitches.json"
            pitches.write_text(json.dumps({
                "times": [0.5, 0.6, 0.7, 0.8],
                "midi_notes": [60.0, 60.0, 60.0, 60.0],
            }), encoding="utf-8")

            beats = root / "beats.json"
            beats.write_text(json.dumps({"bpm": 120}), encoding="utf-8")

            song = {
                **sources,
                "artist": "Artist",
                "title": "Song",
                "syllables": syllables,
                "pitches": pitches,
                "beats": beats,
            }

            output_dir = root / "output"
            formatter = UltraStarFormatter({song_name: song})
            # ffmpeg is only responsible for the full-track transcode here.
            with mock.patch.object(config, "OUTPUT_DIR", output_dir), \
                    mock.patch("src.core.ultrastar_formatter.subprocess.run"):
                formatter.format_ultrastar_song_folder(song_name)

            song_dir = output_dir / song_name
            chart = (song_dir / f"{song_name}.txt").read_text(
                encoding="utf-8"
            ).splitlines()

            vocals_file = song_dir / f"{song_name} [VOC].mp3"
            instrumental_file = song_dir / f"{song_name} [INSTR].mp3"
            self.assertTrue(vocals_file.is_file())
            self.assertTrue(instrumental_file.is_file())
            self.assertEqual(
                vocals_file.read_bytes(), sources["vocals"].read_bytes()
            )
            self.assertIn(f"#VOCALS:{vocals_file.name}", chart)
            self.assertIn(f"#INSTRUMENTAL:{instrumental_file.name}", chart)


if __name__ == "__main__":
    unittest.main()
