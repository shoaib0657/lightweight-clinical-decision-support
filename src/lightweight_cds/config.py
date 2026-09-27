from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class StrictModel(BaseModel):
    """Base for immutable configuration and public records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    source_locator: str = Field(min_length=1)
    license: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    text: str = Field(min_length=1)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_type: str | None = None
    department: str | None = None
    authored_at: date | None = None
    updated_at: date | None = None
    keywords: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    citation_key: str = Field(min_length=1)
    text: str = Field(min_length=1)
    token_start: int = Field(ge=0)
    token_end: int = Field(gt=0)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    citation_key: str = Field(min_length=1)
    text: str = Field(min_length=1)
    final_score: float
    rank: int = Field(ge=1)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class Citation(StrictModel):
    citation_key: str = Field(min_length=1)
    quote: str | None = None


RESEARCH_NOTICE = (
    "Research and education only. This output is not medical advice and has not been "
    "clinically validated."
)


class RagAnswer(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    status: Literal["ok", "insufficient_context"]
    answer: str = Field(min_length=1)
    citations: list[Citation]
    evidence_gaps: list[str]
    source_scope: str = Field(min_length=1)
    safety_notice: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_citations(self) -> "RagAnswer":
        if self.status == "ok" and not self.citations:
            raise ValueError("successful answers require at least one citation")
        if self.status == "insufficient_context" and self.citations:
            raise ValueError("insufficient-context answers cannot claim citations")
        return self

    @classmethod
    def insufficient_context(cls, *, source_scope: str, reason: str) -> "RagAnswer":
        return cls(
            status="insufficient_context",
            answer="The indexed sources do not provide enough evidence to answer this query.",
            citations=[],
            evidence_gaps=[reason],
            source_scope=source_scope,
            safety_notice=RESEARCH_NOTICE,
        )


class WikiDocSourceConfig(StrictModel):
    repo_id: str
    revision: str
    split: str = "train"
    expected_count: int = Field(gt=0)
    local_path: Path | None = None
    license: str


class MedQuADSourceConfig(StrictModel):
    repository_url: str
    revision: str
    expected_count: int = Field(gt=0)
    local_path: Path | None = None
    license: str


class DataConfig(StrictModel):
    output_dir: Path
    cache_dir: Path
    download_if_missing: bool
    retain_exact_duplicates: bool
    expected_total: int = Field(gt=0)
    wikidoc: WikiDocSourceConfig
    medquad: MedQuADSourceConfig


class QuantizationConfig(StrictModel):
    load_in_4bit: bool
    quant_type: Literal["nf4"]
    double_quantization: bool
    compute_dtype: Literal["float16"]


class LoRAConfig(StrictModel):
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    bias: Literal["none"]
    target_modules: list[Literal["q_proj", "v_proj"]]


class TrainingConfig(StrictModel):
    dataset_path: Path
    output_dir: Path
    base_model: str
    model_revision: str
    minimum_gpu_memory_gib: float = Field(gt=0)
    epochs: int = Field(gt=0)
    max_sequence_length: int = Field(gt=0)
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    drop_incomplete_accumulation: bool
    learning_rate: float = Field(gt=0)
    optimizer: Literal["adamw_8bit"]
    scheduler: Literal["linear"]
    warmup_steps: int = Field(ge=0)
    seed: int
    fp16: bool
    bf16: bool
    packing: bool
    completion_only_loss: bool
    quantization: QuantizationConfig
    lora: LoRAConfig

    @computed_field
    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


class EvaluationConfig(StrictModel):
    base_model: str
    model_revision: str
    adapter_path: Path
    output_dir: Path
    num_fewshot: int = Field(ge=0)
    batch_size: int = Field(gt=0)
    device: str
    seed: int
    log_samples: bool
    tasks: list[str] = Field(min_length=1)


class RagConfig(StrictModel):
    input_paths: list[Path] = Field(min_length=1)
    index_dir: Path
    evaluation_queries: Path
    evaluation_output_dir: Path
    embedding_model: str
    embedding_revision: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    document_candidates: int = Field(gt=0)
    final_chunks: int = Field(gt=0)
    rank_constant: int = Field(gt=0)
    minimum_score: float = Field(ge=0)
    keyword_weight: float = Field(ge=0)
    recency_weight: float = Field(ge=0)
    recency_half_life_days: int = Field(gt=0)
    reference_date: date
    context_token_budget: int = Field(gt=0)
    source_scope: str

    @model_validator(mode="after")
    def validate_overlap(self) -> "RagConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class GenerationConfig(StrictModel):
    base_model: str
    model_revision: str
    adapter_path: Path
    max_new_tokens: int = Field(gt=0)
    do_sample: bool
    temperature: float = Field(ge=0)


class AppConfig(StrictModel):
    schema_version: Literal["1.0"]
    project_name: str
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    rag: RagConfig
    generation: GenerationConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    return AppConfig.model_validate(payload)
