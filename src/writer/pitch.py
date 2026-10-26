"""Pitch contour handling: frames, notes, octave fixes, unvoiced fill."""
import numpy as np

from src.writer.constants import (
    EDGE_TRIM_S, MEDIAN_WINDOW_FRAMES, MIDI_FALLBACK, MIDI_MAX, MIDI_MIN,
    MIN_HOLD_NOTE_S, MIN_NOTE_DURATION_S, MIN_STABLE_FRAMES, OCTAVE_JUMP_SEMITONES,
    OCTAVE_TOLERANCE_SEMITONES, PITCH_CONFIDENCE_MIN,
)

def median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 3 or window < 3:
        return values.astype(float)
    half = window // 2
    padded = np.pad(values.astype(float), half, mode="edge")
    return np.array(
        [np.median(padded[i:i + window]) for i in range(len(values))]
    )


def pitch_frames(pitch_data: dict, start: float, end: float
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


def segment_pitch(times: np.ndarray, midi: np.ndarray) -> list[dict]:
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
    smoothed = median_smooth(midi, MEDIAN_WINDOW_FRAMES)

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


def syllable_notes(pitch_data: dict, start: float, end: float
) -> list[dict]:
    """Return the pitch notes for one syllable (possibly empty)."""
    if end <= start:
        end = start + 0.01

    times, midi = pitch_frames(pitch_data, start, end)
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
        notes = segment_pitch(times, midi)
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


def voiced_runs(pitch_data: dict, start: float, end: float
) -> list[tuple[float, float]]:
    """Contiguous voiced runs (in seconds) inside a time window."""
    times, midi = pitch_frames(pitch_data, start, end)
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


def correct_octave_errors(notes: list[dict]) -> int:
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


def fill_unvoiced_syllables(syllables: list[dict]) -> int:
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


def merge_syllable_notes(entries: list[dict],
                         min_hold_s: float = MIN_HOLD_NOTE_S) -> int:
    """Keep real holds as ``~`` continuations, absorb the noisy fragments.

    When the singer holds a vowel and the pitch changes, the syllable keeps
    one note per stable pitch; the writer emits the later ones as
    ``: beat dur pitch ~`` lines. Fragments shorter than ``min_hold_s`` and
    neighbors that share the same semitone are merged into the longer
    neighbor, so a syllable only has several notes when each one is audible.
    Returns the number of syllables whose notes were consolidated.
    """
    changed = 0

    for entry in entries:
        notes = [dict(note) for note in entry["notes"]]
        if len(notes) < 2:
            continue

        def duration(note: dict) -> float:
            return note["end"] - note["start"]

        def absorb(keep: int, drop: int) -> None:
            a, b = notes[min(keep, drop)], notes[max(keep, drop)]
            winner = notes[keep]
            notes[min(keep, drop)] = {
                "start": a["start"],
                "end": b["end"],
                "midi": winner["midi"],
            }
            del notes[max(keep, drop)]

        while len(notes) > 1:
            same = next(
                (i for i in range(len(notes) - 1)
                 if round(notes[i]["midi"]) == round(notes[i + 1]["midi"])),
                None,
            )
            if same is not None:
                longer = max((same, same + 1),
                             key=lambda i: duration(notes[i]))
                absorb(longer, same + (same + 1 - longer))
                continue

            shortest = min(range(len(notes)),
                           key=lambda i: duration(notes[i]))
            if duration(notes[shortest]) >= min_hold_s:
                break
            neighbors = [i for i in (shortest - 1, shortest + 1)
                         if 0 <= i < len(notes)]
            keep = max(neighbors, key=lambda i: duration(notes[i]))
            absorb(keep, shortest)

        if len(notes) != len(entry["notes"]):
            changed += 1
        entry["notes"] = notes

    return changed
