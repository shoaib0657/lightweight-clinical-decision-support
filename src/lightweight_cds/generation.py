from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from lightweight_cds._utils import (
    EnvironmentCheckError,
    InvalidCitationError,
    atomic_directory,
    read_jsonl,
    require_new_artifact,
    stable_json_hash,
    write_json,
)
from lightweight_cds.config import (
    RESEARCH_NOTICE,
    AppConfig,
    Citation,
    RagAnswer,
    RetrievedChunk,
)
from lightweight_cds.retrieval import Retriever, decide_retrieval


PromptMode = Literal[
    "protocol_qa",
    "differential_support",
    "physician_summary",
    "patient_summary",
]

MODE_INSTRUCTIONS: dict[PromptMode, str] = {
    "protocol_qa": "Answer the institutional protocol question using only retrieved evidence.",
    "differential_support": (
        "Provide qualitative differential support and information to verify. "
        "Never assign probabilities or percentages."
    ),
    "physician_summary": "Summarize key findings and follow-up items for a clinician.",
    "patient_summary": (
        "Explain the evidence in plain language without adding diagnoses or treatment advice."
    ),
}

DIAGNOSTIC_PERCENTAGE = re.compile(
    r"(?:probability|likelihood|chance|diagnostic risk).{0,40}\b\d+(?:\.\d+)?\s*%"
    r"|\b\d+(?:\.\d+)?\s*%.{0,40}(?:probability|likelihood|chance|diagnostic risk)",
    flags=re.IGNORECASE,
)


def build_messages(
    mode: PromptMode, query: str, chunks: list[RetrievedChunk]
) -> list[dict[str, str]]:
    evidence = "\n\n".join(f"[{chunk.citation_key}]\n{chunk.text}" for chunk in chunks)
    system = (
        "You are a research-only medical information assistant. "
        f"{MODE_INSTRUCTIONS[mode]} Cite every material claim using only supplied citation "
        "keys. Distinguish evidence from uncertainty, identify evidence gaps, never invent a "
        "citation, and do not output diagnostic probability percentages. Return JSON with "
        "keys answer, citations, and evidence_gaps."
    )
    user = f"QUERY\n{query}\n\nRETRIEVED EVIDENCE\n{evidence}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_citation_keys(keys: list[str], chunks: list[RetrievedChunk]) -> None:
    allowed = {chunk.citation_key for chunk in chunks}
    invalid = sorted(set(keys) - allowed)
    if invalid:
        raise InvalidCitationError(f"generated unknown citations: {', '.join(invalid)}")


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        return stripped[7:-3].strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return stripped[3:-3].strip()
    return stripped


def parse_generated_answer(
    text: str, *, chunks: list[RetrievedChunk], source_scope: str
) -> RagAnswer:
    payload = json.loads(_strip_json_fence(text))
    if not isinstance(payload, dict):
        raise ValueError("model output must be a JSON object")
    raw_citations = payload.get("citations", [])
    if not isinstance(raw_citations, list):
        raise ValueError("citations must be a list")
    citations: list[Citation] = []
    for item in raw_citations:
        if isinstance(item, dict):
            if "citation_key" not in item:
                raise ValueError("citation objects require citation_key")
            citations.append(
                Citation(
                    citation_key=str(item["citation_key"]),
                    quote=str(item["quote"]) if item.get("quote") is not None else None,
                )
            )
        else:
            citations.append(Citation(citation_key=str(item)))
    validate_citation_keys([citation.citation_key for citation in citations], chunks)
    answer = str(payload.get("answer", "")).strip()
    if DIAGNOSTIC_PERCENTAGE.search(answer):
        raise ValueError("answer contains an uncalibrated diagnostic probability percentage")
    evidence_gaps = payload.get("evidence_gaps", [])
    if not isinstance(evidence_gaps, list):
        raise ValueError("evidence_gaps must be a list")
    return RagAnswer(
        status="ok",
        answer=answer,
        citations=citations,
        evidence_gaps=[str(item) for item in evidence_gaps],
        source_scope=source_scope,
        safety_notice=RESEARCH_NOTICE,
    )


def limit_context(
    chunks: list[RetrievedChunk],
    *,
    encode: Callable[[str], list[Any]],
    token_budget: int,
) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        cost = len(encode(chunk.text))
        if selected and used + cost > token_budget:
            break
        if cost > token_budget:
            continue
        selected.append(chunk)
        used += cost
    return selected


@dataclass
class TextGenerator:
    tokenizer: Any
    model: Any
    max_new_tokens: int
    do_sample: bool
    temperature: float

    def generate(self, messages: list[dict[str, str]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        options: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            options["temperature"] = self.temperature
        outputs = self.model.generate(**inputs, **options)
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def load_text_generator(config: AppConfig, *, use_adapter: bool = True) -> TextGenerator:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentCheckError("HF_TOKEN is required for gated Llama model access")
    generation = config.generation
    if use_adapter and not generation.adapter_path.is_dir():
        raise FileNotFoundError(generation.adapter_path)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        generation.base_model, revision=generation.model_revision, token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        generation.base_model,
        revision=generation.model_revision,
        token=token,
        quantization_config=quantization,
        device_map="auto",
    )
    if use_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, generation.adapter_path)
    return TextGenerator(
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=generation.max_new_tokens,
        do_sample=generation.do_sample,
        temperature=generation.temperature,
    )


def answer_query(
    config: AppConfig,
    *,
    query: str,
    mode: PromptMode,
    filters: dict[str, str] | None = None,
) -> RagAnswer:
    """Retrieve evidence, abstain when weak, otherwise generate a cited answer."""

    from transformers import AutoTokenizer

    chunks = Retriever(config).retrieve(query, filters=filters)
    decision = decide_retrieval(
        chunks,
        minimum_score=config.rag.minimum_score,
        source_scope=config.rag.source_scope,
    )
    if decision.answer is not None:
        return decision.answer
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentCheckError("HF_TOKEN is required for gated Llama model access")
    tokenizer = AutoTokenizer.from_pretrained(
        config.generation.base_model,
        revision=config.generation.model_revision,
        token=token,
    )
    limited = limit_context(
        decision.chunks,
        encode=lambda value: tokenizer.encode(value, add_special_tokens=False),
        token_budget=config.rag.context_token_budget,
    )
    if not limited:
        return RagAnswer.insufficient_context(
            source_scope=config.rag.source_scope,
            reason="Retrieved passages could not fit inside the context budget.",
        )
    raw = load_text_generator(config).generate(build_messages(mode, query, limited))
    return parse_generated_answer(raw, chunks=limited, source_scope=config.rag.source_scope)


def _load_gold_queries(path: Path) -> list[dict[str, Any]]:
    queries = read_jsonl(path)
    for row in queries:
        suffixes = row.get("expected_source_suffixes")
        if not row.get("id") or not row.get("query"):
            raise ValueError("each RAG evaluation query requires id and query")
        if not isinstance(suffixes, list) or not suffixes:
            raise ValueError("expected_source_suffixes must be a non-empty list")
    if not queries:
        raise ValueError("at least one RAG evaluation query is required")
    return queries


def _matching_rank(gold: dict[str, Any], chunks: list[RetrievedChunk]) -> int | None:
    suffixes = [str(item) for item in gold["expected_source_suffixes"]]
    for chunk in chunks:
        source_uri = str(chunk.diagnostics.get("source_uri", ""))
        source_without_fragment = source_uri.split("#", maxsplit=1)[0]
        if any(
            source_uri.endswith(suffix) or source_without_fragment.endswith(suffix)
            for suffix in suffixes
        ):
            return chunk.rank
    return None


def _retrieval_metrics(
    cases: list[tuple[dict[str, Any], list[RetrievedChunk]]], *, cutoff: int
) -> dict[str, float]:
    recalls: list[float] = []
    reciprocals: list[float] = []
    ndcgs: list[float] = []
    valid_citations = 0
    citation_count = 0
    for gold, chunks in cases:
        rank = _matching_rank(gold, chunks[:cutoff])
        recalls.append(1.0 if rank is not None else 0.0)
        reciprocals.append(0.0 if rank is None else 1 / rank)
        ndcgs.append(0.0 if rank is None else 1 / math.log2(rank + 1))
        for chunk in chunks[:cutoff]:
            citation_count += 1
            valid_citations += int(
                chunk.citation_key == f"{chunk.document_id}#{chunk.chunk_id}"
            )
    return {
        f"recall_at_{cutoff}": sum(recalls) / len(recalls),
        "mrr": sum(reciprocals) / len(reciprocals),
        "ndcg": sum(ndcgs) / len(ndcgs),
        "citation_validity": valid_citations / citation_count if citation_count else 1.0,
    }


def evaluate_rag(config: AppConfig) -> dict[str, Any]:
    """Run the small synthetic retrieval check; no generation model is loaded."""

    require_new_artifact(config.rag.evaluation_output_dir)
    queries = _load_gold_queries(config.rag.evaluation_queries)
    retriever = Retriever(config)
    cases = [(gold, retriever.retrieve(str(gold["query"]))) for gold in queries]
    metrics = _retrieval_metrics(cases, cutoff=config.rag.final_chunks)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "rag_evaluate",
        "config_sha256": stable_json_hash(config.rag),
        "index": str(config.rag.index_dir),
        "queries": str(config.rag.evaluation_queries),
        "metrics": metrics,
        "assumption": "Synthetic retrieval scores are engineering checks, not clinical evidence.",
    }
    with atomic_directory(config.rag.evaluation_output_dir) as staging:
        write_json(staging / "metrics.json", metrics)
        write_json(
            staging / "retrieval_samples.json",
            [
                {
                    "gold": gold,
                    "retrieved": [chunk.model_dump(mode="json") for chunk in chunks],
                }
                for gold, chunks in cases
            ],
        )
        write_json(staging / "manifest.json", manifest)
    return manifest
