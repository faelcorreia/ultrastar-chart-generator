"""Seconds-to-beat conversion, timing interpolation and number format."""


def seconds_to_beat(seconds: float, bpm: float, gap_s: float = 0.0) -> int:
    return round((seconds - gap_s) * bpm * 4 / 60)


def interpolate_timings(result: list) -> None:
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


def format_number(value: float) -> str:
    """Format a number without trailing zeros (and without 'e' notation)."""
    return f"{value:.6f}".rstrip("0").rstrip(".")
