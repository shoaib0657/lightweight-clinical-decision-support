from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from lightweight_cds._utils import (
    EnvironmentCheckError,
    UnsupportedDocumentError,
    atomic_directory,
    require_new_artifact,
    stable_json_hash,
    write_json,
    write_jsonl,
)
from lightweight_cds.config import AppConfig, ChunkRecord, DocumentRecord
from lightweight_cds.data import normalize_text


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown", ".jsonl"}
MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
Token = TypeVar("Token")


def _document_id(source_uri: str, checksum: str) -> str:
    return hashlib.sha256(f"{source_uri}:{checksum}".encode()).hexdigest()[:20]


def _optional_date(value: Any, source_uri: str) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise UnsupportedDocumentError(
            f"invalid ISO date {value!r} in {source_uri}"
        ) from error


def _build_document(
    *, source_uri: str, title: str, text: str, metadata: dict[str, Any] | None = None
) -> DocumentRecord:
    metadata = dict(metadata or {})
    normalized = normalize_text(text)
    if not normalized:
        raise UnsupportedDocumentError(f"no extractable text in {source_uri}; OCR is out of scope")
    keywords = metadata.get("keywords", [])
    if not isinstance(keywords, list):
        raise UnsupportedDocumentError(f"keywords must be a list in {source_uri}")
    checksum = hashlib.sha256(normalized.encode()).hexdigest()
    reserved = {"document_type", "department", "authored_at", "updated_at", "keywords"}
    return DocumentRecord(
        id=_document_id(source_uri, checksum),
        title=title,
        source_uri=source_uri,
        text=normalized,
        checksum_sha256=checksum,
        document_type=metadata.get("document_type"),
        department=metadata.get("department"),
        authored_at=_optional_date(metadata.get("authored_at"), source_uri),
        updated_at=_optional_date(metadata.get("updated_at"), source_uri),
        keywords=[str(item) for item in keywords],
        metadata={key: value for key, value in metadata.items() if key not in reserved},
    )


def _markdown_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    lines: list[str] = []
    for line in text.splitlines():
        match = MARKDOWN_HEADING.match(line)
        if match:
            if any(item.strip() for item in lines):
                sections.append((heading, "\n".join(lines)))
            heading = match.group(1).strip()
            lines = [heading]
        else:
            lines.append(line)
    if any(item.strip() for item in lines):
        sections.append((heading, "\n".join(lines)))
    return sections


def parse_document(path: Path) -> list[DocumentRecord]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocumentError(f"unsupported document type: {path}")
    source_uri = path.resolve().as_posix()
    if suffix == ".txt":
        return [
            _build_document(
                source_uri=source_uri, title=path.stem, text=path.read_text(encoding="utf-8")
            )
        ]
    if suffix in {".md", ".markdown"}:
        sections = _markdown_sections(path.read_text(encoding="utf-8"))
        if not sections:
            raise UnsupportedDocumentError(f"no extractable text in {source_uri}")
        return [
            _build_document(
                source_uri=f"{source_uri}#section={index}",
                title=heading or path.stem,
                text=text,
                metadata={"section": heading, "section_index": index},
            )
            for index, (heading, text) in enumerate(sections, start=1)
        ]
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(path)
        documents = [
            _build_document(
                source_uri=f"{source_uri}#page={index}",
                title=f"{path.stem} — page {index}",
                text=text,
                metadata={"page": index, "page_count": len(reader.pages)},
            )
            for index, page in enumerate(reader.pages, start=1)
            if (text := page.extract_text() or "").strip()
        ]
        if not documents:
            raise UnsupportedDocumentError(
                f"no extractable text in {source_uri}; scanned PDFs require OCR"
            )
        return documents

    documents: list[DocumentRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise UnsupportedDocumentError(
                f"{path}:{line_number} contains malformed JSON: {error.msg}"
            ) from error
        if not isinstance(row, dict) or "text" not in row:
            raise UnsupportedDocumentError(f"{path}:{line_number} requires a text field")
        raw_metadata = row.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise UnsupportedDocumentError(f"{path}:{line_number} metadata must be an object")
        metadata = dict(raw_metadata)
        for key in ("document_type", "department", "authored_at", "updated_at", "keywords"):
            if key in row:
                metadata[key] = row[key]
        documents.append(
            _build_document(
                source_uri=f"{source_uri}#{line_number}",
                title=str(row.get("title", f"{path.stem} row {line_number}")),
                text=str(row["text"]),
                metadata=metadata,
            )
        )
    if not documents:
        raise UnsupportedDocumentError(f"no extractable records in {source_uri}")
    return documents


def ingest_paths(paths: list[Path]) -> list[DocumentRecord]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                item
                for item in sorted(path.rglob("*"))
                if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    documents = [document for path in files for document in parse_document(path)]
    if not documents:
        raise UnsupportedDocumentError("no supported documents were found")
    return documents


class Tokenizer(Protocol[Token]):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[Token]: ...

    def decode(self, token_ids: list[Token], skip_special_tokens: bool = True) -> str: ...


class TokenChunker(Generic[Token]):
    def __init__(self, tokenizer: Tokenizer[Token], *, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk size must be positive and overlap smaller than it")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: DocumentRecord) -> list[ChunkRecord]:
        tokens = self.tokenizer.encode(document.text, add_special_tokens=False)
        chunks: list[ChunkRecord] = []
        step = self.chunk_size - self.overlap
        for start in range(0, len(tokens), step):
            end = min(start + self.chunk_size, len(tokens))
            text = self.tokenizer.decode(tokens[start:end], skip_special_tokens=True).strip()
            if text:
                checksum = hashlib.sha256(text.encode()).hexdigest()
                chunk_id = hashlib.sha256(
                    f"{document.id}:{start}:{end}:{checksum}".encode()
                ).hexdigest()[:24]
                chunks.append(
                    ChunkRecord(
                        id=chunk_id,
                        document_id=document.id,
                        citation_key=f"{document.id}#{chunk_id}",
                        text=text,
                        token_start=start,
                        token_end=end,
                        checksum_sha256=checksum,
                        metadata={
                            "source_uri": document.source_uri,
                            "title": document.title,
                            "document_type": document.document_type,
                            "department": document.department,
                            "authored_at": (
                                document.authored_at.isoformat() if document.authored_at else None
                            ),
                            "updated_at": (
                                document.updated_at.isoformat() if document.updated_at else None
                            ),
                            "keywords": document.keywords,
                            "source_metadata": document.metadata,
                        },
                    )
                )
            if end == len(tokens):
                break
        return chunks


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


@dataclass(frozen=True)
class BM25Index:
    document_tokens: dict[str, list[str]]
    document_frequency: dict[str, int]
    average_document_length: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(
        cls, documents: dict[str, str], *, k1: float = 1.5, b: float = 0.75
    ) -> "BM25Index":
        tokenized = {document_id: tokenize(text) for document_id, text in documents.items()}
        frequencies: Counter[str] = Counter()
        for tokens in tokenized.values():
            frequencies.update(set(tokens))
        average = (
            sum(len(tokens) for tokens in tokenized.values()) / len(tokenized)
            if tokenized
            else 0.0
        )
        return cls(tokenized, dict(frequencies), average, k1, b)

    def search(self, query: str, *, limit: int) -> list[tuple[str, float]]:
        query_tokens = tokenize(query)
        document_count = len(self.document_tokens)
        scored: list[tuple[str, float]] = []
        for document_id, tokens in self.document_tokens.items():
            terms = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = terms[token]
                if not frequency:
                    continue
                document_frequency = self.document_frequency.get(token, 0)
                inverse_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                length_norm = 1 - self.b
                if self.average_document_length:
                    length_norm += self.b * len(tokens) / self.average_document_length
                score += inverse_frequency * (
                    frequency * (self.k1 + 1) / (frequency + self.k1 * length_norm)
                )
            if score > 0:
                scored.append((document_id, score))
        return sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "document_tokens": self.document_tokens,
            "document_frequency": self.document_frequency,
            "average_document_length": self.average_document_length,
            "k1": self.k1,
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BM25Index":
        if value.get("schema_version") != "1.0":
            raise ValueError("unsupported BM25 index schema")
        return cls(
            document_tokens={
                str(key): [str(token) for token in tokens]
                for key, tokens in value["document_tokens"].items()
            },
            document_frequency={
                str(key): int(count) for key, count in value["document_frequency"].items()
            },
            average_document_length=float(value["average_document_length"]),
            k1=float(value["k1"]),
            b=float(value["b"]),
        )


def _normalize_rows(matrix: Any) -> Any:
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def build_index(
    config: AppConfig, documents: list[DocumentRecord] | None = None
) -> dict[str, Any]:
    """Parse documents and build the local hierarchical hybrid index."""

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    rag = config.rag
    require_new_artifact(rag.index_dir)
    documents = ingest_paths(rag.input_paths) if documents is None else documents
    if not documents:
        raise UnsupportedDocumentError("no documents were supplied")
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        rag.embedding_model, revision=rag.embedding_revision, token=token
    )
    chunker = TokenChunker(tokenizer, chunk_size=rag.chunk_size, overlap=rag.chunk_overlap)
    chunks = [chunk for document in documents for chunk in chunker.chunk(document)]
    if not chunks:
        raise UnsupportedDocumentError("documents produced no token chunks")

    encoder = SentenceTransformer(
        rag.embedding_model, revision=rag.embedding_revision, token=token
    )
    chunk_embeddings = np.asarray(
        encoder.encode(
            [f"passage: {chunk.text}" for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype="float32",
    )
    if chunk_embeddings.ndim != 2:
        raise EnvironmentCheckError("embedding model did not return a two-dimensional matrix")

    chunk_positions: dict[str, list[int]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        chunk_positions[chunk.document_id].append(index)
    document_embeddings = _normalize_rows(
        np.asarray(
            [
                chunk_embeddings[chunk_positions[document.id]].mean(axis=0)
                for document in documents
            ],
            dtype="float32",
        )
    )
    document_dense = faiss.IndexFlatIP(document_embeddings.shape[1])
    document_dense.add(document_embeddings)

    chunks_by_document: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_document[chunk.document_id].append(chunk.text)
    document_bm25 = BM25Index.build(
        {document.id: " ".join(chunks_by_document[document.id]) for document in documents}
    )
    chunk_bm25 = BM25Index.build({chunk.id: chunk.text for chunk in chunks})

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": rag.embedding_model,
        "embedding_revision": rag.embedding_revision,
        "embedding_dimension": int(chunk_embeddings.shape[1]),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunk_size": rag.chunk_size,
        "chunk_overlap": rag.chunk_overlap,
        "documents_sha256": stable_json_hash(documents),
        "chunks_sha256": stable_json_hash(chunks),
        "source_paths": [str(path) for path in rag.input_paths],
    }
    with atomic_directory(rag.index_dir) as staging:
        write_jsonl(staging / "documents.jsonl", documents)
        write_jsonl(staging / "chunks.jsonl", chunks)
        write_json(staging / "document_ids.json", [document.id for document in documents])
        write_json(staging / "chunk_ids.json", [chunk.id for chunk in chunks])
        write_json(staging / "document_bm25.json", document_bm25.to_dict())
        write_json(staging / "chunk_bm25.json", chunk_bm25.to_dict())
        np.save(staging / "chunk_embeddings.npy", chunk_embeddings)
        faiss.write_index(document_dense, str(staging / "document_dense.faiss"))
        write_json(staging / "index_manifest.json", manifest)
    return manifest
