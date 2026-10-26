# I4: real end-to-end runs (en / pt / es)

Date: 2026-10-26. Machine: 8 cores, 39 GB RAM, no GPU (everything on CPU,
Whisper `small` int8, Demucs htdemucs, MMS CTC via onnxruntime). LRCLIB and
MusicBrainz/Cover Art Archive were queried online.

Inputs: only the MP3 of six finished songs from the read-only library
(`/run/media/ghost/ROMS/ultrastar-songs`), copied to a scratch work dir. The
library `.txt` is the ground truth. Command (same for all):

```
python -m src.batch.cli list.csv --audio-dir audio --output-dir out --force-existing --force
python scripts/i4_compare.py out/<song>/<song>.txt <library>/<song>/<song>.txt
```

Wall time: about 3.5 min per song (separation ~1.5 min, Whisper + CTC the rest),
well under the 15 min/song limit, so the batch was not reduced.

## Metrics

`i4_compare.py` takes, for each reference line, the nearest generated line
start (recall side) and for each generated line the nearest reference line start
(precision side). A line is OK when the error is at most 0.5 s. Line-count
differences come from different segmentation, which hurts the recall side more
than the precision side.

## Final results (after the fixes below)

| Song | Lang | Path | Status | Lines gen/ref | Median err (recall) | % ref lines OK | % gen lines OK |
|---|---|---|---|---|---|---|---|
| Avril Lavigne - Knocking on Heaven's Door | en | LRCLIB (plain+synced) -> reconcile -> sync_validate (CTC only) -> align_window | review | 22 / 25 | 0.76 s | 44 % | 50 % |
| Green Day - She | en | same | review | 20 / 37 | 1.41 s | 5 % | 10 % |
| Tribalistas - Já sei namorar | pt | LRCLIB miss -> Whisper on vocals | review | 22 / 86 | 2.68 s | 10 % | 36 % |
| Jota Quest - O vento | pt | LRCLIB miss -> Whisper | failed (clear message) | - / 69 | - | - | - |
| Manu Chao - Clandestino | es | LRCLIB synced -> reconcile -> sync_validate -> align_window | review | 43 / 40 | 0.11 s | 87.5 % | 81 % |
| Mauren Mendoza - Pegasus Fantasy | es | LRCLIB miss -> Whisper on vocals | review | 18 / 23 | 0.56 s | 48 % | 61 % |

Before the fixes: nothing ran at all (see bugs). Intermediate numbers: Avril 16 %
before the CTC-only fallback (44 % after); Green Day 13.5 % recall before and
5 % after (median error 1.69 s to 1.41 s, so timings moved closer on average but
fewer lines landed inside 0.5 s; this song needs a closer look); Pegasus 0 %
(1 line) with VAD on, 48 % with VAD off and vocals as input.

## Bugs found and fixed (all in separate commits)

1. Batch CLI never configured LRCLIB: every song failed with "no LRCLIB client
   configured". Now it builds a real `LrclibClient` (disk cache).
2. Batch CLI did not wire MusicBrainz, and the writer required a cover, so any MP3
   without embedded art failed at the last stage. MusicBrainz/CAA is wired in, and
   the cover is now optional in the writer and the write stage.
3. The CTC aligner aborts the whole process (`std::bad_alloc`, no Python
   exception) when a window has fewer frames than tokens (a 0.05 s window came
   out of a bad reconcile/sync result). Now raises, and the line falls back to
   even timing.
4. `sync_validate` applied a global offset computed from a single noisy anchor
   (shifted a whole song by 6 s). A global offset/drift now needs 3+ anchors and
   at least 50 % agreement.
5. Whisper ran with `vad_filter=True`, which dropped most sung phrases (Pegasus:
   4 words vs 48 without VAD on the mix vs 71 on the vocals) and made the fallback
   say "no words" on two Portuguese songs. VAD is now off and, when LRCLIB has
   nothing, the planner separates the vocals before the lyrics step and Whisper
   transcribes the vocals stem.
6. Whisper small hears few lines of sung English (0-10 % of the lines matched on
   the two English songs), so every line was marked "uncertain" and no correction
   was applied although CTC alone was clearly better than the LRCLIB time (Avril:
   12 of 22 CTC lines within 0.7 s of the reference vs 1 of 22 for the LRCLIB
   times). When Whisper covers less than 30 % of the lines, CTC alone now judges
   the sync and the report says so.
7. Degenerate Whisper output ("e e e e ...", Jota Quest) is rejected with a clear
   message instead of producing a nonsense song.

Environment note (not a code bug): the venv lacked `ctc_forced_aligner` and
`Unidecode`, which are in `requirements.txt`; installed with
`pip install -r requirements.txt`.

## Parameter variations tried (defaults NOT changed except the VAD fix above)

Whisper on the separated vocals, lines of the LRCLIB lyrics found by the D5 3-word
estimator (higher is better). "good" = estimate within 1 s of a reference line.

| Song | Model | Settings | Lines found | good |
|---|---|---|---|---|
| Avril | small | default (repeat 1 / 2) | 2 / 22, 2 / 22 | 2, 2 |
| Avril | small | temperature 0 | 6 / 22 | 1 |
| Avril | small | temperature 0, no condition on previous | 8 / 22 | 5 |
| Avril | small | beam 5, same | 8 / 22 | 5 |
| Avril | medium | default | 2 / 22 | 1 |
| Avril | medium | temperature 0 | 6 / 22 | 5 |
| Green Day | small | any of the above | 0 / 20 | 0 |
| Pegasus (transcript word count) | small | VAD on, mix | 4 words | - |
| Pegasus | small | VAD off, mix / vocals | 48 / 71 words | - |
| Pegasus | medium | VAD off, mix | 47 words | - |
| Jota Quest | small / medium | VAD off, vocals | "e e e ..." / 0 words | - |

Takeaways: Whisper output on sung audio is not deterministic (same file gave 103
and 112 words), `medium` is not better than `small` here (2x slower),
`temperature=0` without conditioning gives a little more coverage but not enough
to change the conclusion. The 3-word exact-prefix estimator is too strict for
Whisper's misheard lyrics ("knocking on the heaven door"). A fuzzy matcher would
be the next improvement. Defaults (model small, estimator) left as they are.

## Problems still open

- Whisper cannot confirm timing on sung English with our estimator; CTC alone
  carries these songs and is right about half of the lines at 0.5 s (Avril),
  but only 10 % on Green Day. Worth investigating whether reference timing for
  Green Day differs systematically (median signed error +0.54 s) or CTC is
  locking onto the wrong repetitions.
- LRCLIB: no match for two Portuguese songs. Tribalistas exists in LRCLIB as
  "12 - Tribalistas - Já Sei Namorar" (plain only, 197 s), which the exact lookup
  and the duration check rightly do not accept. The fallback produced 22 lines
  where the reference has 86: Whisper small on a dense Portuguese song gives
  fragments, and the transcribed text is unreliable (always flagged "review").
- Jota Quest - O vento fails: Whisper small/medium produce only noise on this
  mix. A lyrics-only (plain text) input would be the way to handle it.
- D4 reconcile inserted a copied chorus line with a wrong time in Avril
  (`missing_chorus`, not confirmed by CTC but still flagged and kept); later CTC
  windows had to be repaired. Worth a look in D4.
- Reference files are not uniform (e.g. Jota Quest uses BPM 120 / GAP 0), so
  metrics on those songs are rougher.
