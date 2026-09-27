from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date
from typing import Any

from lightweight_cds._utils import IncompatibleIndexError, read_json, read_jsonl, stable_json_hash
from lightweight_cds.config import (
    AppConfig,
    ChunkRecord,
    DocumentRecord,
    RagAnswer,
    RetrievedChunk,
)
from lightweight_cds.documents import BM25Index


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], *, rank_constant: int
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked_items in rankings.values():
        for rank, item_id in enumerate(ranked_items, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (rank_constant + rank)
    return scores


def apply_metadata_boosts(
    *,
    base_score: float,
    query: str,
    keywords: list[str],
    updated_at: date | None,
    reference_date: date,
    keyword_weight: float,
    recency_weight: float,
    recency_half_life_days: int,
) -> tuple[float, dict[str, object]]:
    lowered_query = query.casefold()
    keyword_matches = [keyword for keyword in keywords if keyword.casefold() in lowered_query]
    keyword_boost = keyword_weight * len(keyword_matches)
    recency_boost = 0.0
    if updated_at is not None:
        age_days = max(0, (reference_date - updated_at).days)
        recency_boost = recency_weight * math.exp(
            -math.log(2) * age_days / recency_half_life_days
        )
    return (
        base_score + keyword_boost + recency_boost,
        {
            "keyword_matches": keyword_matches,
            "keyword_boost": keyword_boost,
            "recency_boost": recency_boost,
        },
    )


@dataclass(frozen=True)
class RetrievalDecision:
    chunks: list[RetrievedChunk]
    answer: RagAnswer | None


def decide_retrieval(
    chunks: list[RetrievedChunk], *, minimum_score: float, source_scope: str
) -> RetrievalDecision:
    accepted = [
        chunk
        for chunk in chunks
        if float(chunk.diagnostics.get("retrieval_score", chunk.final_score)) >= minimum_score
    ]
    if accepted:
        return RetrievalDecision(chunks=accepted, answer=None)
    return RetrievalDecision(
        chunks=[],
        answer=RagAnswer.insufficient_context(
            source_scope=source_scope,
            reason=f"No retrieved chunk met the minimum score of {minimum_score}.",
        ),
    )


def validate_index_records(
    *,
    manifest: dict[str, Any],
    documents: list[DocumentRecord],
    chunks: list[ChunkRecord],
    document_ids: list[str],
    chunk_ids: list[str],
) -> None:
    checks = (
        (manifest.get("document_count") == len(documents), "document count"),
        (manifest.get("chunk_count") == len(chunks), "chunk count"),
        (manifest.get("documents_sha256") == stable_json_hash(documents), "document records"),
        (manifest.get("chunks_sha256") == stable_json_hash(chunks), "chunk records"),
        (document_ids == [document.id for document in documents], "document ID order"),
        (chunk_ids == [chunk.id for chunk in chunks], "chunk ID order"),
    )
    for valid, label in checks:
        if not valid:
            raise IncompatibleIndexError(f"{label} differs from the index manifest")


class Retriever:
    """Load an index once and expose hierarchical dense/BM25 retrieval."""

    def __init__(self, config: AppConfig) -> None:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.config = config
        rag = config.rag
        index_dir = rag.index_dir
        manifest = read_json(index_dir / "index_manifest.json")
        if manifest.get("schema_version") != "1.0":
            raise IncompatibleIndexError("unsupported RAG index schema")
        if (
            manifest.get("embedding_model") != rag.embedding_model
            or manifest.get("embedding_revision") != rag.embedding_revision
        ):
            raise IncompatibleIndexError("configured embedding model differs from the index")
        if (
            manifest.get("chunk_size") != rag.chunk_size
            or manifest.get("chunk_overlap") != rag.chunk_overlap
        ):
            raise IncompatibleIndexError("configured chunking differs from the index")

        self.documents = [
            DocumentRecord.model_validate(row)
            for row in read_jsonl(index_dir / "documents.jsonl")
        ]
        self.chunks = [
            ChunkRecord.model_validate(row) for row in read_jsonl(index_dir / "chunks.jsonl")
        ]
        self.document_ids = [str(item) for item in read_json(index_dir / "document_ids.json")]
        self.chunk_ids = [str(item) for item in read_json(index_dir / "chunk_ids.json")]
        validate_index_records(
            manifest=manifest,
            documents=self.documents,
            chunks=self.chunks,
            document_ids=self.document_ids,
            chunk_ids=self.chunk_ids,
        )
        self.documents_by_id = {document.id: document for document in self.documents}
        self.chunks_by_id = {chunk.id: chunk for chunk in self.chunks}
        self.document_dense = faiss.read_index(str(index_dir / "document_dense.faiss"))
        self.chunk_embeddings = np.load(index_dir / "chunk_embeddings.npy")
        dimension = int(manifest.get("embedding_dimension", 0))
        if self.document_dense.ntotal != len(self.document_ids):
            raise IncompatibleIndexError("document FAISS index size differs from its ID map")
        if self.document_dense.d != dimension:
            raise IncompatibleIndexError("document FAISS dimension differs from the manifest")
        if self.chunk_embeddings.shape != (len(self.chunk_ids), dimension):
            raise IncompatibleIndexError("chunk embeddings differ from the index manifest")
        self.document_bm25 = BM25Index.from_dict(read_json(index_dir / "document_bm25.json"))
        self.chunk_bm25 = BM25Index.from_dict(read_json(index_dir / "chunk_bm25.json"))
        self.encoder = SentenceTransformer(
            rag.embedding_model,
            revision=rag.embedding_revision,
            token=os.environ.get("HF_TOKEN"),
        )

    def _query_vector(self, query: str) -> Any:
        import numpy as np

        vector = self.encoder.encode(
            [f"query: {query}"], normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vector, dtype="float32")

    @staticmethod
    def _metadata_matches(document: DocumentRecord, filters: dict[str, str]) -> bool:
        values: dict[str, object] = {
            "document_type": document.document_type,
            "department": document.department,
            **document.metadata,
        }
        return all(values.get(key) == value for key, value in filters.items())

    def retrieve(
        self,
        query: str,
        *,
        filters: dict[str, str] | None = None,
        reference_date: date | None = None,
    ) -> list[RetrievedChunk]:
        import numpy as np

        rag = self.config.rag
        filters = filters or {}
        reference_date = reference_date or rag.reference_date
        eligible = {
            document.id
            for document in self.documents
            if self._metadata_matches(document, filters)
        }
        if not eligible:
            return []

        query_vector = self._query_vector(query)
        _, dense_indices = self.document_dense.search(query_vector, len(self.document_ids))
        document_dense = [
            self.document_ids[index]
            for index in dense_indices[0]
            if index >= 0 and self.document_ids[index] in eligible
        ]
        document_lexical = [
            document_id
            for document_id, _ in self.document_bm25.search(query, limit=len(self.documents))
            if document_id in eligible
        ]
        document_scores = reciprocal_rank_fusion(
            {"dense": document_dense, "bm25": document_lexical},
            rank_constant=rag.rank_constant,
        )
        selected_documents = [
            document_id
            for document_id, _ in sorted(
                document_scores.items(), key=lambda item: (-item[1], item[0])
            )[: rag.document_candidates]
        ]
        selected = set(selected_documents)
        candidate_indices = [
            index for index, chunk in enumerate(self.chunks) if chunk.document_id in selected
        ]
        if not candidate_indices:
            return []

        dense_scores = self.chunk_embeddings[candidate_indices] @ query_vector[0]
        dense_order = np.argsort(-dense_scores)
        chunk_dense = [self.chunks[candidate_indices[index]].id for index in dense_order]
        dense_similarity = {
            self.chunks[candidate_index].id: float(dense_scores[position])
            for position, candidate_index in enumerate(candidate_indices)
        }
        chunk_lexical = [
            chunk_id
            for chunk_id, _ in self.chunk_bm25.search(query, limit=len(self.chunks))
            if self.chunks_by_id[chunk_id].document_id in selected
        ]
        chunk_scores = reciprocal_rank_fusion(
            {"dense": chunk_dense, "bm25": chunk_lexical}, rank_constant=rag.rank_constant
        )
        dense_rank = {chunk_id: rank for rank, chunk_id in enumerate(chunk_dense, start=1)}
        lexical_rank = {chunk_id: rank for rank, chunk_id in enumerate(chunk_lexical, start=1)}

        scored: list[tuple[ChunkRecord, float, dict[str, object]]] = []
        for chunk_id, base_score in chunk_scores.items():
            chunk = self.chunks_by_id[chunk_id]
            document = self.documents_by_id[chunk.document_id]
            final_score, boosts = apply_metadata_boosts(
                base_score=base_score,
                query=query,
                keywords=document.keywords,
                updated_at=document.updated_at,
                reference_date=reference_date,
                keyword_weight=rag.keyword_weight,
                recency_weight=rag.recency_weight,
                recency_half_life_days=rag.recency_half_life_days,
            )
            diagnostics: dict[str, object] = {
                "document_rrf_score": document_scores.get(document.id, 0.0),
                "chunk_rrf_score": base_score,
                "retrieval_score": base_score,
                "dense_similarity": dense_similarity.get(chunk_id),
                "dense_rank": dense_rank.get(chunk_id),
                "bm25_rank": lexical_rank.get(chunk_id),
                "source_uri": document.source_uri,
                "title": document.title,
                **boosts,
            }
            scored.append((chunk, final_score, diagnostics))

        ranked = sorted(scored, key=lambda item: (-item[1], item[0].id))[: rag.final_chunks]
        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                citation_key=chunk.citation_key,
                text=chunk.text,
                final_score=score,
                rank=rank,
                diagnostics=diagnostics,
            )
            for rank, (chunk, score, diagnostics) in enumerate(ranked, start=1)
        ]
