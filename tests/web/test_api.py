from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from distributed_llm_simulation.application import run_experiment
from distributed_llm_simulation.web.app import create_app
from distributed_llm_simulation.web.drafts import (
    Draft,
    TemplateRequest,
    build_template,
    parse_draft,
    profile_directory,
)
from distributed_llm_simulation.web.results import RunStore, envelope


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "runs")) as c:
        yield c


def template(client, name="pipeline", **parameters):
    response = client.post("/api/templates", json={"template": name, "parameters": parameters})
    assert response.status_code == 200, response.text
    return response.json()


def wait_for(client, jid):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        job = client.get("/api/jobs/" + jid).json()
        if job["state"] not in ("queued", "running"):
            return job
        time.sleep(0.03)
    pytest.fail("worker did not finish")


def test_catalog_preview_and_exact_patches(client):
    catalog = client.get("/api/catalog").json()
    assert len(catalog["templates"]) == 5 and len(catalog["profiles"]) == 6
    draft = template(client)
    response = client.post("/api/preview", json=draft)
    assert response.status_code == 200
    assert response.json()["counts"]["activities"] == 6
    edited = client.post(
        "/api/draft/patch",
        json={
            "draft": draft,
            "edits": [
                {
                    "section": "hardware",
                    "path": ["resources", 3, "capacity"],
                    "value": "9007199254740993",
                    "kind": "integer",
                }
            ],
        },
    ).json()
    description = client.post("/api/describe", json=edited)
    assert description.status_code == 200
    assert description.json()["hardware"]["resources"][3]["capacity"] == "9007199254740993"
    assert "9007199254740993" in edited["hardware"]


def test_validation_missing_routes_and_invalid_yaml(client):
    draft = template(client)
    draft["hardware"] = "nodes: [invalid"
    response = client.post("/api/preview", json=draft)
    assert response.status_code == 422 and response.json()["detail"]
    draft = template(client)
    edited = client.post(
        "/api/draft/patch",
        json={
            "draft": draft,
            "edits": [{"section": "hardware", "path": ["routes"], "kind": "json", "value": "[]"}],
        },
    ).json()
    response = client.post("/api/preview", json=edited)
    assert response.status_code == 422 and "NO_ROUTE" in response.text
    assert (
        client.post(
            "/api/templates", json={"template": "pipeline", "profile": "../secret"}
        ).status_code
        == 422
    )
    assert client.get("/api/not-an-endpoint").status_code == 404
    assert (
        client.post("/api/jobs", json=draft, headers={"Origin": "https://example.org"}).status_code
        == 403
    )


def test_worker_cli_equivalence_and_artifacts(client, tmp_path):
    draft = template(client)
    response = client.post("/api/jobs", json=draft)
    assert response.status_code == 202
    job = wait_for(client, response.json()["id"])
    assert job["state"] == "finished", job.get("error")
    assert job["result_status"] == "SUCCESS"
    runs = client.get("/api/runs").json()
    assert len(runs) == 1 and runs[0]["makespan_s"] == pytest.approx(0.0091)
    rid = runs[0]["id"]
    detail = client.get("/api/runs/" + rid).json()
    assert "bounds.json" in detail["artifacts"]
    assert "llm_metrics.json" not in detail["artifacts"]
    assert client.get(f"/api/runs/{rid}/draft").status_code == 200
    assert client.get("/api/jobs/" + job["id"] + "/draft").json()["hardware"]
    assert client.get(f"/api/runs/{rid}/specs").json()["execution"]["activities"]
    assert len(client.get(f"/api/runs/{rid}/timeline?request=r0").json()) == 3
    trace = client.get(f"/api/runs/{rid}/trace?offset=1&limit=2").json()
    assert len(trace["rows"]) == 2 and trace["total"] > 2
    assert client.get(f"/api/runs/{rid}/artifacts/summary.json").status_code == 200
    assert client.get(f"/api/runs/{rid}/artifacts/worker.log").status_code == 404
    assert client.get("/api/runs/missing").status_code == 404
    assert client.get(f"/api/runs/{rid}/trace?limit=0").status_code == 422
    result = run_experiment(*parse_draft(Draft.model_validate(draft)), tmp_path / "reference")
    assert detail["summary"]["metrics"] == result.metrics
    recorded = client.get(f"/api/runs/{rid}/artifacts/trace.jsonl").text
    assert recorded == (tmp_path / "reference/trace.jsonl").read_text()
    curves = client.get(f"/api/runs/{rid}/series").json()
    assert curves["memory_available"] and curves["rates_available"]
    for series in curves["series"]:
        if series["kind"] == "memory":
            assert (
                max(p[1] for p in series["points"])
                == result.metrics["peak_memory_bytes"][series["id"]]
            )
    d = client.get(f"/api/runs/{rid}/draft").json()
    d["config"] = d["config"].replace("resource_changes", "summary")
    short = client.post("/api/jobs", json=d).json()
    assert wait_for(client, short["id"])["state"] == "finished"
    row = next(
        r for r in client.get("/api/runs").json() if r["directory"] == short["run_directory"]
    )
    unavailable = client.get("/api/runs/" + row["id"] + "/series").json()
    assert not unavailable["memory_available"] and not unavailable["rates_available"]
    assert not unavailable["series"]


def test_cancel_queued_and_running_jobs(client):
    draft = template(client, requests=100)
    first = client.post("/api/jobs", json=draft).json()
    second = client.post("/api/jobs", json=draft).json()
    assert (
        client.post("/api/jobs/" + second["id"] + "/cancel", json={}).json()["state"] == "cancelled"
    )
    assert (
        client.post("/api/jobs/" + first["id"] + "/cancel", json={}).json()["state"] == "cancelled"
    )
    assert wait_for(client, first["id"])["state"] == "cancelled"
    time.sleep(0.15)
    assert client.get("/api/jobs/" + second["id"]).json()["state"] == "cancelled"


def test_restart_marks_tasks_interrupted(tmp_path):
    root = tmp_path / "runs"
    folder = root / ".ui/jobs/old"
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(
        json.dumps(
            {"id": "old", "state": "running", "created_at": "2026", "run_directory": "ui-old"}
        )
    )
    with TestClient(create_app(root)) as client:
        assert client.get("/api/jobs/old").json()["state"] == "interrupted"


def test_legacy_partial_and_symlink_safety(tmp_path):
    root = tmp_path / "runs"
    draft = build_template(TemplateRequest(template="pipeline"), profile_directory())
    result = run_experiment(*parse_draft(draft), root / "legacy")
    for name in ("hardware.graph.json", "workload.graph.json"):
        (root / "legacy" / name).unlink()
    store = RunStore(root)
    row = store.scan()[0]
    assert store.specs(row["id"])["hardware"]["name"]
    assert store.detail(row["id"])["summary"]["makespan_s"] == result.makespan_s
    (root / "partial").mkdir()
    (root / "partial/manifest.json").write_text("{}")
    assert any(r["status"] == "INCOMPLETE" for r in store.scan())
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "summary.json").write_text("{}")
    (root / "escape").symlink_to(outside, target_is_directory=True)
    (root / "legacy/manifest.json").unlink()
    (root / "legacy/manifest.json").symlink_to(outside / "summary.json")
    assert all(r["directory"] not in ("legacy", "escape") for r in store.scan())


def test_envelope_preserves_peaks_and_order():
    points = [[i, 999 if i == 333 else i % 20] for i in range(1000)]
    sampled = envelope(points, 100)
    assert max(p[1] for p in sampled) == 999
    assert sampled[0] == points[0] and sampled[-1] == points[-1]
    assert sampled == sorted(sampled)


def test_preview_validates_runtime_config_without_executing(client):
    draft = template(client, name="image")
    edited = client.post(
        "/api/draft/patch",
        json={
            "draft": draft,
            "edits": [
                {
                    "section": "config",
                    "path": ["capacity_processes"],
                    "kind": "json",
                    "value": json.dumps([{"resource_id": "hbm_bw", "points": [[0, 100], [1, 50]]}]),
                }
            ],
        },
    ).json()
    response = client.post("/api/preview", json=edited)
    assert response.status_code == 422
    assert "UNSUPPORTED_DYNAMIC_COMPUTE" in response.text
    assert client.get("/api/runs").json() == []


def test_shutdown_during_subprocess_creation_cleans_up(tmp_path, monkeypatch):
    import asyncio

    from distributed_llm_simulation.web.jobs import JobManager

    async def scenario():
        creating, allow_spawn, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Process:
            returncode = None

            def terminate(self):
                self.returncode = -15
                exited.set()

            def kill(self):
                self.terminate()

            async def wait(self):
                await exited.wait()
                return self.returncode

        process = Process()

        async def spawn(*args, **kwargs):
            creating.set()
            await allow_spawn.wait()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        manager = JobManager(tmp_path / "runs")
        draft = build_template(TemplateRequest(template="pipeline"), profile_directory())
        job = manager.submit(draft)
        await manager.start()
        await asyncio.wait_for(creating.wait(), 2)
        stopping = asyncio.create_task(manager.stop())
        await asyncio.sleep(0)
        allow_spawn.set()
        await asyncio.wait_for(stopping, 2)
        assert process.returncode == -15
        assert manager.jobs[job["id"]]["state"] == "interrupted"
        assert not manager.processes

    asyncio.run(scenario())
