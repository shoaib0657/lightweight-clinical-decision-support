from __future__ import annotations

import importlib.metadata
import math
import os
import platform
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from lightweight_cds._utils import (
    EnvironmentCheckError,
    atomic_directory,
    read_jsonl,
    require_new_artifact,
    sha256_file,
    stable_json_hash,
    write_json,
)
from lightweight_cds.config import AppConfig, DatasetRecord


def to_prompt_completion(record: DatasetRecord) -> dict[str, list[dict[str, str]]]:
    return {
        "prompt": [
            {"role": "user", "content": f"{record.instruction}\n\n{record.question}"}
        ],
        "completion": [{"role": "assistant", "content": record.answer}],
    }


def infer_optimizer_steps(
    *,
    sample_count: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    drop_incomplete_accumulation: bool,
) -> int:
    effective_batch = micro_batch_size * gradient_accumulation_steps
    operation = math.floor if drop_incomplete_accumulation else math.ceil
    return operation(sample_count / effective_batch)


def expected_lora_parameter_count(
    *,
    layer_count: int,
    hidden_size: int,
    key_value_size: int,
    rank: int,
    targets: Sequence[str],
) -> int:
    dimensions = {
        "q_proj": (hidden_size, hidden_size),
        "v_proj": (hidden_size, key_value_size),
    }
    unknown = set(targets) - dimensions.keys()
    if unknown:
        raise ValueError(f"unsupported configured targets: {sorted(unknown)}")
    per_layer = sum(
        rank * (input_size + output_size)
        for input_size, output_size in (dimensions[target] for target in targets)
    )
    return layer_count * per_layer


def _package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _format_batch(example: dict[str, Any]) -> dict[str, Any]:
    return to_prompt_completion(DatasetRecord.model_validate(example))


def train_adapter(config: AppConfig) -> dict[str, Any]:
    """Train and save a PEFT adapter; base-model weights are never merged or saved."""

    training = config.training
    require_new_artifact(training.output_dir)
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentCheckError(
            "HF_TOKEN is required after accepting the gated Meta Llama 3.2 license"
        )
    if not training.dataset_path.is_file():
        raise FileNotFoundError(training.dataset_path)
    rows = read_jsonl(training.dataset_path)
    if len(rows) != config.data.expected_total:
        raise EnvironmentCheckError(
            f"training corpus must contain {config.data.expected_total} rows, observed {len(rows)}"
        )

    import torch
    from datasets import load_dataset
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise EnvironmentCheckError("the training profile requires an NVIDIA CUDA GPU")
    memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if memory_gib < training.minimum_gpu_memory_gib:
        raise EnvironmentCheckError(
            f"GPU has {memory_gib:.1f} GiB; at least "
            f"{training.minimum_gpu_memory_gib:g} GiB is configured"
        )

    quantization = BitsAndBytesConfig(
        load_in_4bit=training.quantization.load_in_4bit,
        bnb_4bit_quant_type=training.quantization.quant_type,
        bnb_4bit_use_double_quant=training.quantization.double_quantization,
        bnb_4bit_compute_dtype=torch.float16,
    )
    with atomic_directory(training.output_dir) as staging:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=training.base_model,
            revision=training.model_revision,
            max_seq_length=training.max_sequence_length,
            dtype=torch.float16,
            load_in_4bit=training.quantization.load_in_4bit,
            quantization_config=quantization,
            token=token,
            use_exact_model_name=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=training.lora.rank,
            target_modules=list(training.lora.target_modules),
            lora_alpha=training.lora.alpha,
            lora_dropout=training.lora.dropout,
            bias=training.lora.bias,
            use_gradient_checkpointing="unsloth",
            random_state=training.seed,
            max_seq_length=training.max_sequence_length,
        )
        trainable, _ = model.get_nb_trainable_parameters()
        expected_trainable = expected_lora_parameter_count(
            layer_count=28,
            hidden_size=3_072,
            key_value_size=1_024,
            rank=training.lora.rank,
            targets=training.lora.target_modules,
        )
        if trainable != expected_trainable:
            raise EnvironmentCheckError(
                f"LoRA parameter contract changed: expected {expected_trainable}, "
                f"observed {trainable}"
            )

        dataset = load_dataset("json", data_files=str(training.dataset_path), split="train")
        dataset = dataset.map(_format_batch, remove_columns=list(dataset.column_names))
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=str(staging / "checkpoints"),
                num_train_epochs=training.epochs,
                max_length=training.max_sequence_length,
                per_device_train_batch_size=training.micro_batch_size,
                gradient_accumulation_steps=training.gradient_accumulation_steps,
                dataloader_drop_last=training.drop_incomplete_accumulation,
                learning_rate=training.learning_rate,
                optim=training.optimizer,
                lr_scheduler_type=training.scheduler,
                warmup_steps=training.warmup_steps,
                seed=training.seed,
                fp16=training.fp16,
                bf16=training.bf16,
                packing=training.packing,
                completion_only_loss=training.completion_only_loss,
                save_strategy="no",
                report_to="none",
                logging_steps=10,
            ),
        )
        result = trainer.train()
        adapter_dir = staging / "adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        metrics = {
            key: value
            for key, value in result.metrics.items()
            if isinstance(value, (str, int, float))
        }
        write_json(staging / "metrics.json", metrics)
        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": "training",
            "config_sha256": stable_json_hash(training),
            "inputs": {
                "dataset": str(training.dataset_path),
                "dataset_sha256": sha256_file(training.dataset_path),
                "base_model": training.base_model,
                "model_revision": training.model_revision,
            },
            "outputs": {"adapter": str(training.output_dir / "adapter")},
            "metrics": {**metrics, "trainable_parameters": trainable},
            "settings": training.model_dump(mode="json"),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                **_package_versions(["unsloth", "transformers", "trl", "torch"]),
            },
            "assumptions": [
                "q_proj and v_proj match the configured approximate 2.4M parameter target.",
                "Micro-batch 4 and accumulation 4 produce 1,650 full optimizer steps.",
            ],
        }
        write_json(staging / "manifest.json", manifest)
    return manifest
