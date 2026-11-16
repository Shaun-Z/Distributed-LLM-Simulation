"""Measurement exchange and held-out validation; never claim synthetic data is real."""

from __future__ import annotations

import math
import statistics

from pydantic import model_validator

from ..analysis.metrics import nearest_rank
from ..spec.base import Bytes, NonNegative, Positive, Spec, require


class Measurement(Spec):
    id: str
    device_identity: str
    software_version: str
    measured_at: str
    methodology: str
    source_uri: str
    synthetic: bool
    split: str  # calibration | validation
    operation: str
    size_bytes: Bytes = 0
    duration_s: Positive
    shape: tuple[Bytes, ...] = ()
    dtype: str = "fp32"
    concurrency: Positive = 1

    @model_validator(mode="after")
    def check(self):
        require(
            self.split in ("calibration", "validation"),
            "INVALID_MEASUREMENT",
            self.id,
            "unknown split",
        )
        require(
            all(
                v.strip()
                for v in (
                    self.device_identity,
                    self.software_version,
                    self.measured_at,
                    self.methodology,
                    self.source_uri,
                )
            ),
            "INVALID_MEASUREMENT",
            self.id,
            "provenance must be populated",
        )
        if not self.synthetic:
            require(
                not any(
                    v.lower() in ("todo", "unknown", "placeholder", "synthetic", "example")
                    for v in (self.device_identity, self.software_version, self.methodology)
                ),
                "INVALID_MEASUREMENT",
                self.id,
                "placeholder is not a real measurement",
            )
        return self


class CopyCalibration(Spec):
    setup_s: NonNegative
    payload_Bps: Positive
    calibration_ids: tuple[str, ...]
    synthetic: bool
    signature: tuple[str, str, str, float, str]

    def predict(self, size_bytes):
        return self.setup_s + size_bytes / self.payload_Bps


def fit_copy_profile(records):
    records = tuple(records)
    require(
        len(records) >= 2
        and len({r.id for r in records}) == len(records)
        and all(r.split == "calibration" for r in records),
        "INVALID_CALIBRATION",
        "records",
        "at least two calibration-only measurements required",
    )
    signatures = {
        (r.device_identity, r.software_version, r.operation, r.concurrency, r.dtype)
        for r in records
    }
    require(
        len(signatures) == 1,
        "INVALID_CALIBRATION",
        "records",
        "fit one operation/device/concurrency at a time",
    )
    xs, ys = [r.size_bytes for r in records], [r.duration_s for r in records]
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - xm) ** 2 for x in xs)
    require(denominator > 0, "INVALID_CALIBRATION", "size_bytes", "distinct sizes required")
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys, strict=False)) / denominator
    require(slope > 0 and math.isfinite(slope), "INVALID_CALIBRATION", "fit", "nonpositive slope")
    intercept = ym - slope * xm
    require(
        intercept >= -1e-12,
        "INVALID_CALIBRATION",
        "fit",
        "negative setup; linear model is unsuitable",
    )
    return CopyCalibration(
        setup_s=max(0, intercept),
        payload_Bps=1 / slope,
        calibration_ids=tuple(r.id for r in records),
        synthetic=any(r.synthetic for r in records),
        signature=next(iter(signatures)),
    )


def validate_copy_profile(profile, held_out):
    records = tuple(held_out)
    require(
        bool(records)
        and len({r.id for r in records}) == len(records)
        and all(
            r.split == "validation"
            and r.id not in profile.calibration_ids
            and (r.device_identity, r.software_version, r.operation, r.concurrency, r.dtype)
            == profile.signature
            for r in records
        ),
        "INVALID_VALIDATION",
        "records",
        "independent held-out records required",
    )
    rows = [
        {
            "id": r.id,
            "size_bytes": r.size_bytes,
            "measured_s": r.duration_s,
            "predicted_s": profile.predict(r.size_bytes),
            "absolute_error_s": abs(profile.predict(r.size_bytes) - r.duration_s),
            "relative_error": abs(profile.predict(r.size_bytes) - r.duration_s) / r.duration_s,
        }
        for r in records
    ]
    return {
        "samples": rows,
        "mean_relative_error": statistics.mean(r["relative_error"] for r in rows),
        "p95_relative_error": nearest_rank([r["relative_error"] for r in rows], 0.95),
        "synthetic": profile.synthetic or any(r.synthetic for r in records),
    }
