"""Versioned Web response shapes; domain specs remain the core's source of truth."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .drafts import Draft


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PreviewResponse(Payload):
    draft: Draft
    specs: dict[str, Any]
    counts: dict[str, int]
    hashes: dict[str, str]
    execution: dict[str, Any]


class CatalogResponse(Payload):
    profiles: list[dict[str, Any]]
    templates: list[dict[str, Any]]


class JobResponse(Payload):
    id: str
    label: str
    state: Literal["queued", "running", "finished", "failed", "cancelled", "interrupted"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result_status: str | None = None
    error: str | None = None
    run_directory: str


class RunSummaryResponse(Payload):
    id: str
    directory: str
    label: str
    workload: str
    status: str
    makespan_s: float | None = None
    metrics: dict[str, Any]
    model_level: str | None = None
    scheduler: str | None = None
    network_mode: str | None = None
    modified_at: float


class RunResponse(Payload):
    id: str
    directory: str
    summary: dict[str, Any]
    manifest: dict[str, Any]
    critical_path: Any = None
    bounds: Any = None
    llm_metrics: Any = None
    failure: Any = None
    invalid: Any = None
    resources: list[dict[str, str]]
    requests: list[dict[str, str]]
    artifacts: list[str]


class SeriesResponse(Payload):
    series: list[dict[str, Any]]
    memory_available: bool
    rates_available: bool
    reason: str | None = None
    downsampled: bool = False


class TraceResponse(Payload):
    rows: list[dict[str, Any]]
    total: int
    offset: int
    limit: int
