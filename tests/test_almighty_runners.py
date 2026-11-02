import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from src.model.song_job import SongJob  # noqa: E402
from src.stages import almighty_runners as ar  # noqa: E402
from src.stages import audio_analysis as aa  # noqa: E402
from tests.test_audio_analysis import SONGS  # noqa: E402

PITCH_FRAMES = {"schema_version": 1, "frames": [
    {"time": 0.0, "f0_hz": None, "confidence": 0.1},
    {"time": 0.01, "f0_hz": 440.0, "confidence": 0.9},
    {"time": 0.02, "f0_hz": 261.6255653005986, "confidence": 0.5},
]}

BEATS_DOC = {"schema_version": 1, "bpm": 120.0, "engine": "librosa", "meter": 4,
             "beats": [0.5, 1.0, 1.5], "downbeats": [0.5]}


class FakeResult:
    """Stand-in for almighty_listener.api.Result: get() devolve o arquivo pronto."""

    def __init__(self, files):
        self._files = files

    def get(self, artifact):
        if artifact not in self._files:
            raise KeyError(f"artefato {artifact!r} não foi produzido")
        return self._files[artifact]


class FakeListen:
    """listen falso e injetável: grava as chamadas e devolve os arquivos dados."""

    def __init__(self, files, error=None):
        self.files = files
        self.error = error
        self.calls = []

    def __call__(self, path, want=None, *, device="cpu", options=None,
                 cache_dir=None, work_dir=None):
        self.calls.append({"path": Path(path), "want": list(want),
                           "cache_dir": Path(cache_dir),
                           "work_dir": Path(work_dir)})
        if self.error is not None:
            raise self.error
        return FakeResult(self.files)


class AlmightyRunnersTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.media = self.tmp / "media"
        self.media.mkdir()
        self.files = {}
        self._write("stems.vocals", b"VOCALS")
        self._write("stems.instrumental", b"INSTR")
        self._write_json("pitch", PITCH_FRAMES)
        self._write_json("beats", BEATS_DOC)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, artifact, data: bytes):
        path = self.media / artifact.replace(".", "_")
        path.write_bytes(data)
        self.files[artifact] = path

    def _write_json(self, artifact, doc):
        path = self.media / f"{artifact}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        self.files[artifact] = path

    def song_dirs(self, lang, artist, title):
        name = f"{artist} - {title}"
        audio = self.tmp / lang / f"{name}.mp3"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"a")
        work = self.tmp / lang / name / ".work"
        work.mkdir(parents=True, exist_ok=True)
        return name, audio, work

    def test_separate_writes_wav_with_old_names(self):
        for lang, (artist, title) in SONGS.items():
            with self.subTest(lang=lang):
                name, audio, work = self.song_dirs(lang, artist, title)
                fake = FakeListen(self.files)
                out = ar.almighty_separate({"full_track": audio}, name, work,
                                           listen=fake)
                self.assertEqual(out["vocals"], work / f"{name} - vocals.wav")
                self.assertEqual(out["instrumental"],
                                 work / f"{name} - instrumental.wav")
                self.assertEqual(out["vocals"].read_bytes(), b"VOCALS")
                self.assertEqual(out["instrumental"].read_bytes(), b"INSTR")
                call = fake.calls[0]
                self.assertEqual(call["path"], audio)
                self.assertEqual(call["want"], ["stems"])
                self.assertEqual(call["cache_dir"], self.tmp / lang / ".almighty-cache")

    def test_pitches_unvoiced_frames_are_zero_not_null(self):
        for lang, (artist, title) in SONGS.items():
            with self.subTest(lang=lang):
                name, audio, work = self.song_dirs(lang, artist, title)
                fake = FakeListen(self.files)
                out = ar.almighty_pitches({"full_track": audio}, name, work,
                                          listen=fake)
                self.assertEqual(out["pitches"], work / f"{name} - pitches.json")
                data = json.loads(out["pitches"].read_text(encoding="utf-8"))
                self.assertEqual(set(data), {"times", "frequencies", "midi_notes",
                                             "voiced_prob"})
                self.assertEqual(data["times"], [0.0, 0.01, 0.02])
                self.assertEqual(data["frequencies"][0], 0.0)
                self.assertEqual(data["midi_notes"][0], 0.0)
                self.assertEqual(data["midi_notes"][1], 69.0)
                self.assertEqual(data["midi_notes"][2], 60.0)
                self.assertEqual(data["voiced_prob"], [0.1, 0.9, 0.5])
                self.assertNotIn(None, data["frequencies"] + data["midi_notes"])
                self.assertEqual(fake.calls[0]["want"], ["pitch"])

    def test_beats_read_the_instrumental(self):
        for lang, (artist, title) in SONGS.items():
            with self.subTest(lang=lang):
                name, audio, work = self.song_dirs(lang, artist, title)
                instrumental = work / f"{name} - instrumental.wav"
                instrumental.write_bytes(b"INSTR")
                fake = FakeListen(self.files)
                out = ar.almighty_beats({"full_track": audio,
                                         "instrumental": instrumental},
                                        name, work, listen=fake)
                self.assertEqual(out["beats"], work / f"{name} - beats.json")
                data = json.loads(out["beats"].read_text(encoding="utf-8"))
                self.assertEqual(data["bpm"], 120.0)
                self.assertEqual(data["beats"], [0.5, 1.0, 1.5])
                self.assertEqual(fake.calls[0]["path"], instrumental)
                self.assertEqual(fake.calls[0]["want"], ["beats"])

    def test_beats_without_instrumental_uses_full_track(self):
        name, audio, work = self.song_dirs("en", *SONGS["en"])
        fake = FakeListen(self.files)
        ar.almighty_beats({"full_track": audio}, name, work, listen=fake)
        self.assertEqual(fake.calls[0]["path"], audio)

    def test_almighty_failure_names_song_and_stage(self):
        name, audio, work = self.song_dirs("pt", *SONGS["pt"])
        fake = FakeListen(self.files, error=RuntimeError("boom na GPU"))
        with self.assertRaises(ar.AlmightyError) as ctx:
            ar.almighty_separate({"full_track": audio}, name, work, listen=fake)
        message = str(ctx.exception)
        self.assertIn(name, message)
        self.assertIn("os stems", message)
        self.assertIn("boom na GPU", message)

    def test_missing_artifact_fails_clearly(self):
        name, audio, work = self.song_dirs("es", *SONGS["es"])
        fake = FakeListen({"stems.vocals": self.files["stems.vocals"]})
        with self.assertRaises(ar.AlmightyError) as ctx:
            ar.almighty_separate({"full_track": audio}, name, work, listen=fake)
        self.assertIn("stems.instrumental", str(ctx.exception))
        self.assertIn(name, str(ctx.exception))

    def test_pitch_outside_contract_fails_clearly(self):
        name, audio, work = self.song_dirs("en", *SONGS["en"])
        self._write_json("pitch", {"frames": [{"f0_hz": 220.0, "confidence": 0.4}]})
        with self.assertRaises(ar.AlmightyError) as ctx:
            ar.almighty_pitches({"full_track": audio}, name, work,
                                listen=FakeListen(self.files))
        self.assertIn("fora do contrato", str(ctx.exception))

    def test_missing_package_fails_with_install_hint(self):
        name, audio, work = self.song_dirs("en", *SONGS["en"])
        with mock.patch.dict(sys.modules, {"almighty_listener.api": None}):
            with self.assertRaises(ar.AlmightyError) as ctx:
                ar.almighty_beats({"full_track": audio}, name, work)
        self.assertIn("não instalado", str(ctx.exception))

    def test_default_stages_run_through_almighty(self):
        fake = FakeListen(self.files)
        for lang, (artist, title) in SONGS.items():
            with self.subTest(lang=lang):
                name, audio, _ = self.song_dirs(lang, artist, title)
                job = SongJob(artist=artist, title=title, language=lang,
                              audio_path=audio)
                with mock.patch.object(ar, "_default_listen", return_value=fake):
                    stages = aa.make_stages(self.tmp / lang)
                    for stage in ("vocals", "pitches", "beats"):
                        stages[stage](job)
                work = self.tmp / lang / name / ".work"
                self.assertEqual(job.vocals_path, work / f"{name} - vocals.wav")
                self.assertEqual(job.extra["instrumental"],
                                 work / f"{name} - instrumental.wav")
                self.assertTrue(job.extra["pitches"].is_file())
                self.assertTrue(job.extra["beats"].is_file())
                self.assertNotIn("review", job.extra)
        wants = [call["want"] for call in fake.calls]
        self.assertEqual(wants, [["stems"], ["pitch"], ["beats"]] * 3)

    def test_all_unvoiced_pitch_flags_review(self):
        self._write_json("pitch", {"frames": [
            {"time": 0.0, "f0_hz": None, "confidence": 0.0},
            {"time": 0.01, "f0_hz": None, "confidence": 0.01},
        ]})
        name, audio, _ = self.song_dirs("en", *SONGS["en"])
        job = SongJob(artist=SONGS["en"][0], title=SONGS["en"][1], language="en",
                      audio_path=audio)
        with mock.patch.object(ar, "_default_listen",
                               return_value=FakeListen(self.files)):
            stages = aa.make_stages(self.tmp / "en")
            stages["vocals"](job)
            stages["pitches"](job)
        self.assertIn("no melody", job.extra["review"])


if __name__ == "__main__":
    unittest.main()
