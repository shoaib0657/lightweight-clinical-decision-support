from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import tempfile
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from lightweight_cds._utils import (
    SourceCountMismatchError,
    SourceUnavailableError,
    atomic_directory,
    require_new_artifact,
    sha256_file,
    write_json,
    write_jsonl,
)
from lightweight_cds.config import (
    AppConfig,
    DatasetRecord,
    MedQuADSourceConfig,
    WikiDocSourceConfig,
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def normalize_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(html.unescape(value))
    text = unicodedata.normalize("NFKC", " ".join(parser.parts))
    return re.sub(r"\s+", " ", text).strip()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_dataset_record(
    *,
    source: str,
    instruction: str,
    question: str,
    answer: str,
    source_locator: str,
    license_name: str,
    metadata: dict[str, Any] | None = None,
) -> DatasetRecord:
    instruction = normalize_text(instruction)
    question = normalize_text(question)
    answer = normalize_text(answer)
    source_locator = normalize_text(source_locator)
    content_hash = _sha256_text(f"{question}\n{answer}")
    record_id = _sha256_text(
        f"{source}\n{source_locator}\n{question}\n{answer}"
    )[:24]
    return DatasetRecord(
        id=record_id,
        source=source,
        instruction=instruction,
        question=question,
        answer=answer,
        source_locator=source_locator,
        license=license_name,
        content_sha256=content_hash,
        metadata=metadata or {},
    )


def parse_wikidoc_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_locator_prefix: str = "wikidoc",
    license_name: str = "CC-BY-SA-3.0",
) -> Iterator[DatasetRecord]:
    for index, row in enumerate(rows):
        instruction = normalize_text(str(row.get("instruction", "")))
        question = normalize_text(str(row.get("input", "")))
        answer = normalize_text(str(row.get("output", "")))
        if not instruction or not question or not answer:
            continue
        yield make_dataset_record(
            source="medical_meadow_wikidoc",
            instruction=instruction,
            question=question,
            answer=answer,
            source_locator=f"{source_locator_prefix}#{index}",
            license_name=license_name,
            metadata={"row_index": index},
        )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_text(element: ET.Element, name: str) -> str:
    for child in element.iter():
        if _local_name(child.tag) == name:
            return normalize_text("".join(child.itertext()))
    return ""


def parse_medquad_directory(
    root_dir: Path,
    *,
    license_name: str = "CC-BY-4.0",
) -> Iterator[DatasetRecord]:
    root_dir = root_dir.resolve()
    for xml_path in sorted(root_dir.rglob("*.xml")):
        document = ET.parse(xml_path).getroot()
        focus = _first_text(document, "Focus")
        relative = xml_path.relative_to(root_dir).as_posix()
        collection = relative.split("/", 1)[0] if "/" in relative else xml_path.parent.name
        pairs = (item for item in document.iter() if _local_name(item.tag) == "QAPair")
        for pair_index, pair in enumerate(pairs):
            question_element = next(
                (item for item in pair.iter() if _local_name(item.tag) == "Question"), None
            )
            answer_element = next(
                (item for item in pair.iter() if _local_name(item.tag) == "Answer"), None
            )
            if question_element is None or answer_element is None:
                continue
            question = normalize_text("".join(question_element.itertext()))
            answer = normalize_text("".join(answer_element.itertext()))
            if not question or not answer:
                continue
            question_id = question_element.attrib.get("qid", str(pair_index))
            yield make_dataset_record(
                source="medquad",
                instruction="Answer this question truthfully",
                question=question,
                answer=answer,
                source_locator=f"{relative}#{question_id}",
                license_name=license_name,
                metadata={
                    "collection": collection,
                    "pair_id": pair.attrib.get("pid"),
                    "question_id": question_id,
                    "question_type": question_element.attrib.get("qtype"),
                    "focus": focus or None,
                },
            )


def _load_local_wikidoc(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"WikiDoc source must be a JSON array of objects: {path}")
    return rows


def _acquire_wikidoc(
    config: WikiDocSourceConfig, *, cache_dir: Path, download_if_missing: bool
) -> Iterable[Mapping[str, Any]]:
    if config.local_path is not None:
        return _load_local_wikidoc(config.local_path)
    if not download_if_missing:
        raise SourceUnavailableError("WikiDoc local_path is absent and downloads are disabled")
    from datasets import load_dataset

    return load_dataset(
        config.repo_id,
        revision=config.revision,
        split=config.split,
        cache_dir=str(cache_dir / "huggingface"),
    )


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise SourceUnavailableError(f"unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def _resolve_medquad_root(source_dir: Path) -> Path:
    roots = [path for path in source_dir.iterdir() if path.is_dir()]
    root = roots[0] if len(roots) == 1 else source_dir
    if not any(root.rglob("*.xml")):
        raise SourceUnavailableError(f"MedQuAD cache contains no XML files: {source_dir}")
    return root


def _acquire_medquad(
    config: MedQuADSourceConfig, *, cache_dir: Path, download_if_missing: bool
) -> Path:
    if config.local_path is not None:
        return config.local_path
    if not download_if_missing:
        raise SourceUnavailableError("MedQuAD local_path is absent and downloads are disabled")
    source_dir = cache_dir / f"medquad-{config.revision[:12]}"
    if source_dir.exists():
        return _resolve_medquad_root(source_dir)
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{source_dir.name}-", dir=source_dir.parent))
    try:
        archive = staging / "source.zip"
        url = f"{config.repository_url}/archive/{config.revision}.zip"
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        _safe_extract(archive, staging)
        archive.unlink()
        _resolve_medquad_root(staging)
        staging.replace(source_dir)
    except BaseException:
        if staging.exists() and staging.parent == source_dir.parent:
            shutil.rmtree(staging)
        raise
    return _resolve_medquad_root(source_dir)


def count_duplicates(records: list[DatasetRecord]) -> dict[str, Any]:
    counts = Counter(record.content_sha256 for record in records)
    duplicate_hashes = {key: count for key, count in counts.items() if count > 1}
    return {
        "total_records": len(records),
        "unique_records": len(counts),
        "duplicate_records": sum(count - 1 for count in duplicate_hashes.values()),
        "duplicate_hashes": duplicate_hashes,
    }


def validate_source_counts(
    *, observed: dict[str, int], expected: dict[str, int], expected_total: int
) -> None:
    mismatches = [
        f"{source}: expected {expected[source]}, observed {observed.get(source, 0)}"
        for source in expected
        if observed.get(source) != expected[source]
    ]
    if sum(observed.values()) != expected_total:
        mismatches.append(f"total: expected {expected_total}, observed {sum(observed.values())}")
    if mismatches:
        raise SourceCountMismatchError("source count drift detected; " + "; ".join(mismatches))


def prepare_dataset(config: AppConfig) -> dict[str, Any]:
    """Create the normalized training corpus and its provenance manifest."""

    data = config.data
    require_new_artifact(data.output_dir)
    wikidoc = list(
        parse_wikidoc_rows(
            _acquire_wikidoc(
                data.wikidoc,
                cache_dir=data.cache_dir,
                download_if_missing=data.download_if_missing,
            ),
            source_locator_prefix=(
                f"hf://{data.wikidoc.repo_id}@{data.wikidoc.revision}/{data.wikidoc.split}"
            ),
            license_name=data.wikidoc.license,
        )
    )
    medquad = list(
        parse_medquad_directory(
            _acquire_medquad(
                data.medquad,
                cache_dir=data.cache_dir,
                download_if_missing=data.download_if_missing,
            ),
            license_name=data.medquad.license,
        )
    )
    observed = {"wikidoc": len(wikidoc), "medquad": len(medquad)}
    expected = {
        "wikidoc": data.wikidoc.expected_count,
        "medquad": data.medquad.expected_count,
    }
    validate_source_counts(observed=observed, expected=expected, expected_total=data.expected_total)
    source_records = [*wikidoc, *medquad]
    duplicate_report = count_duplicates(source_records)
    records = source_records
    if not data.retain_exact_duplicates:
        records = list({record.content_sha256: record for record in source_records}.values())

    with atomic_directory(data.output_dir) as staging:
        dataset_path = staging / "training.jsonl"
        write_jsonl(dataset_path, records)
        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_revisions": {
                data.wikidoc.repo_id: data.wikidoc.revision,
                data.medquad.repository_url: data.medquad.revision,
            },
            "source_licenses": {
                data.wikidoc.repo_id: data.wikidoc.license,
                data.medquad.repository_url: data.medquad.license,
            },
            "expected_counts": expected,
            "observed_counts": observed,
            "total_records": len(records),
            "dataset_sha256": sha256_file(dataset_path),
            "duplicate_report": duplicate_report,
            "retain_exact_duplicates": data.retain_exact_duplicates,
            "assumptions": [
                "The configured count combines 10,000 WikiDoc and 16,412 MedQuAD rows.",
                "The default profile reports and retains exact duplicate QA pairs.",
            ],
        }
        write_json(staging / "manifest.json", manifest)
    return manifest
