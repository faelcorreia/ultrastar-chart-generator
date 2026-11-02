import io
import os
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

os.environ.setdefault("INPUT_DIR", tempfile.gettempdir())
os.environ.setdefault("OUTPUT_DIR", tempfile.gettempdir())

from fastapi.testclient import TestClient  # noqa: E402

from src.api.server import create_app  # noqa: E402
from src.api.store import JobStore, Worker  # noqa: E402
from src.model.lyrics import Line, Lyrics, Syllable  # noqa: E402

NAMES = ["normalize", "metadata", "lyrics", "language", "vocals", "align",
         "reconcile", "sync_validate", "align_window", "align_plain",
         "pitches", "beats", "syllables", "write"]
SONGS = [("Beyoncé", "Halo", "en"), ("João Gilberto", "Chega de Saudade", "pt"),
         ("Peso Pluma", "Ella Baila Sola", "es")]


def fake_stages(out: Path, fail_on=None):
    def mk(name):
        def stage(job):
            if name == fail_on:
                raise RuntimeError("boom")
            if name == "language" and not job.language:
                job.language = "en"
            if name == "lyrics":
                job.lyrics = Lyrics(lines=[Line(text="la")],
                                    language=job.language)
            if name == "align_plain":
                job.lyrics = Lyrics(lines=[Line(syllables=[
                    Syllable("la", 0.0, 1.0)])], language=job.language)
            if name == "write":
                folder = out / f"{job.artist} - {job.title}"
                (folder / ".work").mkdir(parents=True, exist_ok=True)
                (folder / ".work" / "cache.json").write_text("{}")
                (folder / "song.txt").write_text(f"#TITLE:{job.title}\n",
                                                 encoding="utf-8")
                job.extra["output_folder"] = folder
        return stage
    return {n: mk(n) for n in NAMES}


def no_probe(path):
    return {}


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())

    def client(self, **kw):
        stages = kw.pop("stages", fake_stages(self.out))
        app = create_app(self.out, stages=stages, probe=no_probe, **kw)
        return TestClient(app)

    def upload(self, c, name="a.mp3", data=b"ID3fake", **form):
        return c.post("/jobs", files={"file": (name, data, "audio/mpeg")},
                      data=form)

    def wait(self, c, job_id, until=("done", "failed", "review")):
        for _ in range(200):
            rec = c.get(f"/jobs/{job_id}").json()
            if rec["status"] in until:
                return rec
            time.sleep(0.02)
        self.fail(f"job stuck: {rec}")

    def test_full_flow_three_languages(self):
        with self.client() as c:
            for artist, title, lang in SONGS:
                r = self.upload(c, artist=artist, title=title, language=lang,
                                year="2000")
                self.assertEqual(r.status_code, 202)
                body = r.json()
                self.assertEqual((body["artist"], body["title"],
                                  body["language"]), (artist, title, lang))
                self.assertNotIn("folder", body)
                rec = self.wait(c, body["id"])
                self.assertEqual(rec["status"], "done", rec)
                self.assertTrue(rec["chart_ready"])
                z = c.get(f"/jobs/{body['id']}/chart")
                self.assertEqual(z.status_code, 200)
                names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
                self.assertEqual(names, [f"{artist} - {title}/song.txt"])

    def test_identity_from_filename(self):
        with self.client() as c:
            r = self.upload(c, name="Mägo de Oz - La Danza del Fuego.mp3")
            self.assertEqual((r.json()["artist"], r.json()["title"]),
                             ("Mägo de Oz", "La Danza del Fuego"))

    def test_missing_identity_is_422(self):
        with self.client() as c:
            self.assertEqual(self.upload(c, name="x.mp3").status_code, 422)
            self.assertEqual(c.get("/jobs").json(), [])

    def test_rejects_bad_type_empty_and_oversize(self):
        with self.client(max_upload_mb=1) as c:
            self.assertEqual(
                self.upload(c, name="A - B.txt").status_code, 415)
            self.assertEqual(
                self.upload(c, name="A - B.mp3", data=b"").status_code, 422)
            big = b"0" * (1024 * 1024 + 1)
            self.assertEqual(
                self.upload(c, name="A - B.mp3", data=big).status_code, 413)

    def test_failed_job_reports_stage_reason_and_no_chart(self):
        with self.client(stages=fake_stages(self.out, "pitches")) as c:
            jid = self.upload(c, name="A - B.mp3").json()["id"]
            rec = self.wait(c, jid)
            self.assertEqual(rec["status"], "failed")
            self.assertIn("pitches", rec["reason"])
            self.assertEqual(c.get(f"/jobs/{jid}/chart").status_code, 409)

    def test_unknown_job_404_and_delete(self):
        with self.client() as c:
            self.assertEqual(c.get("/jobs/nope").status_code, 404)
            jid = self.upload(c, name="A - B.mp3").json()["id"]
            self.wait(c, jid)
            self.assertEqual(c.delete(f"/jobs/{jid}").status_code, 200)
            self.assertEqual(c.get(f"/jobs/{jid}").status_code, 404)

    def test_jobs_run_one_at_a_time_in_order(self):
        order = []

        def runner(job, out, stages):
            order.append(("start", job.title))
            time.sleep(0.05)
            order.append(("end", job.title))
            from src.planner import JobResult
            return JobResult("done", "ok", job)

        with self.client(runner=runner) as c:
            ids = [self.upload(c, name=f"A - T{i}.mp3").json()["id"]
                   for i in range(3)]
            for jid in ids:
                self.wait(c, jid)
        self.assertEqual(order, [(k, f"T{i}") for i in range(3)
                                 for k in ("start", "end")])

    def test_queued_job_can_be_cancelled(self):
        import threading
        gate = threading.Event()

        def runner(job, out, stages):
            gate.wait(5)
            from src.planner import JobResult
            return JobResult("done", "ok", job)

        with self.client(runner=runner) as c:
            first = self.upload(c, name="A - One.mp3").json()["id"]
            second = self.upload(c, name="A - Two.mp3").json()["id"]
            self.wait(c, first, until=("running",))
            self.assertEqual(c.delete(f"/jobs/{first}").status_code, 409)
            self.assertEqual(c.delete(f"/jobs/{second}").json()["status"],
                             "cancelled")
            gate.set()
            self.assertEqual(self.wait(c, first)["status"], "done")
            self.assertEqual(c.get(f"/jobs/{second}").json()["status"],
                             "cancelled")

    def test_queue_survives_restart(self):
        store = JobStore(self.out / ".api")
        jid = store.new_id()
        store.job_dir(jid).mkdir(parents=True)
        (store.job_dir(jid) / "audio.mp3").write_bytes(b"x")
        store.create({"id": jid, "artist": "A", "title": "B",
                      "language": "pt", "year": None, "filename": "x.mp3"})
        store.update(jid, status="running", stage="vocals")
        with self.client() as c:  # new process: recovers and runs it
            self.assertEqual(self.wait(c, jid)["status"], "done")

    def test_stage_progress_is_tracked(self):
        seen = []
        stages = fake_stages(self.out)
        store_ref = {}
        orig = stages["pitches"]

        def pitches(job):
            seen.append(store_ref["c"].get("/jobs").json()[0]["stage"])
            orig(job)
        stages["pitches"] = pitches
        with self.client(stages=stages) as c:
            store_ref["c"] = c
            self.wait(c, self.upload(c, name="A - B.mp3").json()["id"])
        self.assertEqual(seen, ["pitches"])


if __name__ == "__main__":
    unittest.main()
