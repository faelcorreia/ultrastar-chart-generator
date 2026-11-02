import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from app import _main, _parse_input_file  # noqa: E402
from src import planner as P  # noqa: E402

SONGS = {
    "en": "The Beatles - Let It Be",
    "pt": "Caetano Veloso - Força Estranha",
    "es": "Shakira - Sueño Mañana",
}


class InputFileParsingTest(unittest.TestCase):
    def test_full_track(self):
        parsed = _parse_input_file(Path(
            "Example Artist - Example Song - full track.mp3"
        ))

        self.assertEqual(
            parsed["artist_song"], "Example Artist - Example Song"
        )
        self.assertEqual(parsed["artist"], "Example Artist")
        self.assertEqual(parsed["title"], "Example Song")
        self.assertEqual(parsed["role"], "full_track")
        self.assertIsNone(parsed["language"])

    def test_lyrics_with_language(self):
        parsed = _parse_input_file(
            Path("Artist-Name - Song Title - en lyrics.txt")
        )

        self.assertEqual(parsed["artist"], "Artist-Name")
        self.assertEqual(parsed["title"], "Song Title")
        self.assertEqual(parsed["role"], "lyrics")
        self.assertEqual(parsed["language"], "en")

    def test_lyrics_without_language_stays_unresolved(self):
        parsed = _parse_input_file(Path("Artist - Song - lyrics.txt"))

        self.assertEqual(parsed["role"], "lyrics")
        self.assertIsNone(parsed["language"])

    def test_cover(self):
        parsed = _parse_input_file(Path("Artist - Song - cover.jpg"))

        self.assertEqual(parsed["role"], "cover")

    def test_unrelated_file_is_ignored(self):
        self.assertIsNone(_parse_input_file(Path("notes.txt")))

    def test_missing_artist_title_separator_is_ignored(self):
        self.assertIsNone(_parse_input_file(Path("Song - full track.mp3")))


class MainPlannerTest(unittest.TestCase):
    """app._main with fake planner stages: pass, fail, --force, name filter."""

    STAGE_NAMES = (P.NORMALIZE, P.METADATA, P.LANGUAGE, P.VOCALS, P.ALIGN,
                   P.ALIGN_PLAIN, P.ALIGN_WINDOW, P.RECONCILE,
                   P.SYNC_VALIDATE, P.PITCHES, P.BEATS, P.SYLLABLES, P.WRITE)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.input = self.tmp / "input"
        self.output = self.tmp / "output"
        self.input.mkdir()
        self.output.mkdir()
        for lang, name in SONGS.items():
            (self.input / f"{name} - full track.mp3").write_bytes(b"a")
            (self.input / f"{name} - cover.jpg").write_bytes(b"c")
            (self.input / f"{name} - {lang} lyrics.txt").write_text(
                "primeira linha\nsegunda linha\n", encoding="utf-8")
        self.calls: list[tuple[str, str]] = []

    def tearDown(self):
        self._tmp.cleanup()

    def stages(self, fail_for=None):
        calls = self.calls

        def fake(stage):
            def run(job):
                calls.append((stage, job.artist))
                if stage == P.PITCHES and job.artist == fail_for:
                    raise RuntimeError("pitch boom")
                # Outputs must exist on disk: the stage cache only hits when
                # every file of the stored result is still there.
                def out(kind):
                    path = self.output / f"{job.artist} - {kind}"
                    path.write_text("{}", encoding="utf-8")
                    return path

                if stage == P.VOCALS:
                    job.vocals_path = job.audio_path
                elif stage == P.PITCHES:
                    job.extra["pitches"] = out("pitches.json")
                elif stage == P.BEATS:
                    job.extra["beats"] = out("beats.json")
                elif stage == P.SYLLABLES:
                    job.extra["syllables"] = out("syllables.json")
                elif stage == P.WRITE:
                    folder = self.output / f"{job.artist} - {job.title}"
                    folder.mkdir(exist_ok=True)
                    job.extra["output_folder"] = folder
            return run

        return {name: fake(name) for name in self.STAGE_NAMES}

    def run_main(self, argv, fail_for=None, force=False):
        args = list(argv) + (["--force"] if force else [])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = _main(args, input_dir=self.input, output_dir=self.output,
                         stages=self.stages(fail_for))
        return code, out.getvalue()

    def written(self):
        return {artist for stage, artist in self.calls if stage == P.WRITE}

    def test_failing_song_does_not_stop_the_others(self):
        code, out = self.run_main([], fail_for="Caetano Veloso")

        self.assertEqual(code, 1)
        self.assertEqual(self.written(), {"The Beatles", "Shakira"})
        self.assertIn("=> failed: pitches: pitch boom", out)
        self.assertIn("1 song(s) failed: Caetano Veloso - Força Estranha", out)

    def test_all_songs_ok_exits_zero(self):
        code, _ = self.run_main([])

        self.assertEqual(code, 0)
        self.assertEqual(self.written(), {"The Beatles", "Caetano Veloso",
                                          "Shakira"})

    def test_name_filter_runs_only_named_song(self):
        code, _ = self.run_main([SONGS["es"]])

        self.assertEqual(code, 0)
        self.assertEqual({artist for _, artist in self.calls}, {"Shakira"})

    def test_unknown_name_is_an_error(self):
        code, out = self.run_main(["Nobody - Nothing"])

        self.assertEqual(code, 1)
        self.assertEqual(self.calls, [])
        self.assertIn("Not found", out)

    def test_force_recomputes_cached_stages(self):
        self.run_main([])
        first = len(self.calls)

        self.run_main([])
        self.assertEqual(len(self.calls), first, "second run should use cache")

        self.run_main([], force=True)
        self.assertEqual(len(self.calls), first * 2)


if __name__ == "__main__":
    unittest.main()
