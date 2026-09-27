from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class LightweightCDSError(Exception):
    """Base class for expected, user-actionable failures."""


class ArtifactExistsError(LightweightCDSError):
    pass


class SourceCountMismatchError(LightweightCDSError):
    pass


class SourceUnavailableError(LightweightCDSError):
    pass


class UnsupportedDocumentError(LightweightCDSError):
    pass


class InvalidCitationError(LightweightCDSError):
    pass


class IncompatibleIndexError(LightweightCDSError):
    pass


class EnvironmentCheckError(LightweightCDSError):
    pass


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return value


def redact_secrets(value: Any) -> Any:
    """Remove credential-looking values before serialization."""

    if isinstance(value, Mapping):
        secret_keys = {"hf_token", "access_token", "api_key", "password", "secret"}
        return {
            str(key): "[REDACTED]"
            if str(key).casefold() in secret_keys
            else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    token = os.environ.get("HF_TOKEN")
    if token and isinstance(value, str):
        return value.replace(token, "[REDACTED]")
    return value


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = redact_secrets(jsonable(value))
    path.write_text(
        json.dumps(safe, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, values: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            safe = redact_secrets(jsonable(value))
            handle.write(json.dumps(safe, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def require_new_artifact(target: Path) -> None:
    if target.resolve().exists():
        raise ArtifactExistsError(f"immutable artifact path already exists: {target.resolve()}")


@contextmanager
def atomic_directory(target: Path) -> Iterator[Path]:
    target = target.resolve()
    require_new_artifact(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        yield staging
        os.replace(staging, target)
    except BaseException:
        if staging.exists() and staging.parent == target.parent:
            shutil.rmtree(staging)
        raise
