"""Redistribution of voiced tails the aligner left between syllables."""
from src.writer.constants import (
    GAP_TOUCH_TOLERANCE_S, MAX_VOICED_GAP_S, SHORT_SYLLABLE_S, VOICED_GAP_MIN_S,
)
from src.writer.pitch import voiced_runs

def redistribute_voiced_gaps(entries: list[dict], pitch_data: dict
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

        runs = voiced_runs(pitch_data, gap_start, gap_end)
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
