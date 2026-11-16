"""Immutable, canonical specifications. Runtime state never lives in these models."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False, strict=True)]
Bytes = Annotated[int, Field(ge=0, strict=True)]
Count = Annotated[int, Field(gt=0, strict=True)]


class Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    @model_validator(mode="after")
    def nonempty_identifiers(self):
        for field in ("id", "name"):
            if hasattr(self, field) and not getattr(self, field).strip():
                raise ValueError(f"{field} cannot be empty")
        return self

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class SpecError(ValueError):
    def __init__(self, code: str, path: str, message: str):
        self.code, self.path = code, path
        super().__init__(f"{code} at {path}: {message}")


def indexed(items, path):
    result = {}
    for item in items:
        if item.id in result:
            raise SpecError("DUPLICATE_ID", path, item.id)
        result[item.id] = item
    return result


def require(condition, code, path, message):
    if not condition:
        raise SpecError(code, path, message)
