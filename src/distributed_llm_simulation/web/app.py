from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .contracts import (
    CatalogResponse,
    JobResponse,
    PreviewResponse,
    RunResponse,
    RunSummaryResponse,
    SeriesResponse,
    TraceResponse,
)
from .drafts import (
    SECTIONS,
    Draft,
    PatchRequest,
    TemplateRequest,
    build_template,
    catalog,
    parse_draft,
    patch,
    preview,
    profile_directory,
    wire,
)
from .jobs import JobManager
from .results import RunStore


def create_app(runs_dir="runs", profiles_dir=None):
    root = Path(runs_dir).resolve()
    profiles = profile_directory(profiles_dir)
    manager = JobManager(root)
    store = RunStore(root)

    @asynccontextmanager
    async def lifespan(app):
        await manager.start()
        yield
        await manager.stop()

    app = FastAPI(title="Heterosim Workbench", version="1", lifespan=lifespan)
    app.state.jobs, app.state.store = manager, store

    @app.middleware("http")
    async def same_origin_mutations(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method in ("POST", "PUT", "DELETE", "PATCH") and origin:
            from urllib.parse import urlsplit

            if urlsplit(origin).netloc != request.headers.get("host"):
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    {"detail": "Cross-origin mutations are not allowed."}, status_code=403
                )
        return await call_next(request)

    @app.exception_handler(FileNotFoundError)
    @app.exception_handler(KeyError)
    async def missing(request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            {
                "detail": "A required artifact is missing."
                if isinstance(error, FileNotFoundError)
                else str(error)
            },
            status_code=404,
        )

    @app.exception_handler(ValueError)
    @app.exception_handler(TypeError)
    @app.exception_handler(yaml.YAMLError)
    async def invalid(request, error):
        from fastapi.responses import JSONResponse

        details = (
            error.errors(include_context=False, include_input=False)
            if isinstance(error, ValidationError)
            else [{"loc": [getattr(error, "path", "draft")], "msg": str(error)}]
        )
        return JSONResponse({"detail": details}, status_code=422)

    @app.get("/api/catalog", response_model=CatalogResponse)
    def get_catalog():
        return wire(catalog(profiles))

    @app.post("/api/templates", response_model=Draft)
    def template(body: TemplateRequest):
        return build_template(body, profiles)

    @app.post("/api/describe")
    def describe(body: Draft):
        return wire(
            {
                key: spec.model_dump(mode="json")
                for key, spec in zip(SECTIONS, parse_draft(body), strict=True)
            }
        )

    @app.post("/api/preview", response_model=PreviewResponse)
    def validate(body: Draft):
        return preview(body)

    @app.post("/api/draft/patch", response_model=Draft)
    def edit(body: PatchRequest):
        return patch(body)

    @app.get("/api/jobs", response_model=list[JobResponse])
    async def jobs():
        return sorted(manager.jobs.values(), key=lambda j: j["created_at"], reverse=True)

    @app.post("/api/jobs", status_code=202, response_model=JobResponse)
    async def submit(body: Draft):
        # Validation runs in a thread so API status/cancellation remain responsive.
        result = await asyncio.to_thread(preview, body)
        return manager.submit(Draft.model_validate(result["draft"]))

    @app.get("/api/jobs/{jid}", response_model=JobResponse)
    async def job(jid: str):
        return manager.jobs[jid]

    @app.get("/api/jobs/{jid}/draft")
    async def job_draft(jid: str):
        job = manager.jobs[jid]
        return Draft.model_validate_json((manager.directory / job["id"] / "draft.json").read_text())

    @app.post("/api/jobs/{jid}/cancel", response_model=JobResponse)
    async def cancel(jid: str):
        return await manager.cancel(jid)

    @app.get("/api/runs", response_model=list[RunSummaryResponse])
    def runs():
        rows = store.scan()
        labels = {j["run_directory"]: j["label"] for j in manager.jobs.values()}
        return wire([{**row, "label": labels.get(row["directory"], row["label"])} for row in rows])

    @app.get("/api/runs/{rid}", response_model=RunResponse)
    def run(rid: str):
        return wire(store.detail(rid))

    @app.get("/api/runs/{rid}/draft")
    def run_draft(rid: str):
        return store.draft(rid)

    @app.get("/api/runs/{rid}/specs")
    def specs(rid: str):
        return wire(store.specs(rid))

    @app.get("/api/runs/{rid}/timeline")
    def timeline(
        rid: str,
        start: float | None = Query(None, ge=0, allow_inf_nan=False),
        end: float | None = Query(None, ge=0, allow_inf_nan=False),
        request: str | None = None,
        search: str = "",
    ):
        if start is not None and end is not None and start > end:
            raise ValueError("End time must not precede start time.")
        return wire(store.timeline(rid, start, end, request, search))

    @app.get("/api/runs/{rid}/series", response_model=SeriesResponse)
    def series(
        rid: str,
        start: float | None = Query(None, ge=0, allow_inf_nan=False),
        end: float | None = Query(None, ge=0, allow_inf_nan=False),
        max_points: int = Query(2000, ge=20, le=20000),
    ):
        if start is not None and end is not None and start > end:
            raise ValueError("End time must not precede start time.")
        return wire(store.series(rid, start, end, max_points))

    @app.get("/api/runs/{rid}/trace", response_model=TraceResponse)
    def trace(
        rid: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
        event_type: str | None = None,
        activity: str | None = None,
        start: float | None = Query(None, ge=0, allow_inf_nan=False),
        end: float | None = Query(None, ge=0, allow_inf_nan=False),
    ):
        if start is not None and end is not None and start > end:
            raise ValueError("End time must not precede start time.")
        return wire(store.trace(rid, offset, limit, event_type, activity, start, end))

    @app.get("/api/runs/{rid}/artifacts/{name}")
    def artifact(rid: str, name: str):
        return FileResponse(store.artifact(rid, name), filename=name)

    static = Path(__file__).parent / "static"
    if (static / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path.startswith("api/") or path == "api":
            raise HTTPException(404, "Unknown API endpoint")
        if not (static / "index.html").is_file():
            raise HTTPException(
                503,
                "Build the frontend first: npm --prefix frontend ci && npm --prefix frontend run build",
            )
        return FileResponse(static / "index.html")

    return app
