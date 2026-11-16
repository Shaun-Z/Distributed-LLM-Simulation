from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path


def now():
    return datetime.now(UTC).isoformat()


class JobManager:
    """Single-worker persistent queue. Cancellation and completion share one event loop."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.directory = self.root / ".ui" / "jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.processes = {}
        self.queue = asyncio.Queue()
        self.task = None
        for path in self.directory.glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
                if job.get("id") != path.parent.name:
                    continue
                job.setdefault("label", job["id"])
                for key in ("started_at", "finished_at", "result_status", "error"):
                    job.setdefault(key, None)
                if job["state"] in ("queued", "running"):
                    job.update(
                        state="interrupted",
                        finished_at=now(),
                        error="The workbench stopped before this job finished.",
                    )
                self.jobs[job["id"]] = job
                self.save(job)
            except (ValueError, OSError, KeyError):
                continue

    def save(self, job):
        directory = self.directory / job["id"]
        directory.mkdir(exist_ok=True)
        temporary = directory / "job.tmp"
        temporary.write_text(json.dumps(job, indent=2))
        temporary.replace(directory / "job.json")

    async def start(self):
        self.task = asyncio.create_task(self.work())

    async def stop(self):
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        for process in list(self.processes.values()):
            await self.terminate(process)
        for job in self.jobs.values():
            if job["state"] in ("queued", "running"):
                job.update(state="interrupted", finished_at=now())
                self.save(job)

    def submit(self, draft):
        jid = uuid.uuid4().hex
        job = {
            "id": jid,
            "label": draft.label,
            "state": "queued",
            "created_at": now(),
            "started_at": None,
            "finished_at": None,
            "result_status": None,
            "error": None,
            "run_directory": f"ui-{jid}",
        }
        self.jobs[jid] = job
        self.save(job)
        (self.directory / jid / "draft.json").write_text(draft.model_dump_json(indent=2))
        self.queue.put_nowait(jid)
        return dict(job)

    async def terminate(self, process):
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()

    async def cancel(self, jid):
        job = self.jobs[jid]
        if job["state"] in ("queued", "running"):
            job.update(state="cancelled", finished_at=now())
            self.save(job)
            if jid in self.processes:
                await self.terminate(self.processes[jid])
        return dict(job)

    async def work(self):
        while True:
            jid = await self.queue.get()
            job = self.jobs[jid]
            try:
                if job["state"] != "queued":
                    continue
                job.update(state="running", started_at=now())
                self.save(job)
                directory = self.directory / jid
                with (directory / "worker.log").open("w") as log:
                    creation = asyncio.create_task(
                        asyncio.create_subprocess_exec(
                            sys.executable,
                            "-m",
                            "distributed_llm_simulation.web.worker",
                            str(directory / "draft.json"),
                            str(self.root / job["run_directory"]),
                            stdout=log,
                            stderr=log,
                        )
                    )
                    try:
                        process = await asyncio.shield(creation)
                    except asyncio.CancelledError:
                        process = await creation
                        await self.terminate(process)
                        raise
                    self.processes[jid] = process
                    if job["state"] != "running":
                        await self.terminate(process)
                    await process.wait()
                if job["state"] == "running":
                    summary = self.root / job["run_directory"] / "summary.json"
                    if process.returncode == 0 and summary.is_file():
                        job.update(
                            state="finished",
                            result_status=json.loads(summary.read_text())["status"],
                        )
                    else:
                        job.update(
                            state="failed", error=(directory / "worker.log").read_text()[-16000:]
                        )
                    job["finished_at"] = now()
                    self.save(job)
            except asyncio.CancelledError:
                if jid in self.processes:
                    await self.terminate(self.processes[jid])
                raise
            except Exception as error:
                job.update(state="failed", error=str(error), finished_at=now())
                self.save(job)
            finally:
                self.processes.pop(jid, None)
                self.queue.task_done()
