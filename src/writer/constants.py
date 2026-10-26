"""Tuning constants and language names for the UltraStar writer."""

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
# A pitch change inside a syllable only becomes a ``~`` continuation note
# when each part lasts at least this long.
MIN_HOLD_NOTE_S = 0.2
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
