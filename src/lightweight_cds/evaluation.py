from __future__ import annotations

import importlib.metadata
import json
import os
from datetime import datetime, timezone
from typing import Any, Literal

from lightweight_cds._utils import (
    EnvironmentCheckError,
    atomic_directory,
    require_new_artifact,
    stable_json_hash,
    write_json,
)
from lightweight_cds.config import AppConfig


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        return str(value)
    return value


def _model_args(config: AppConfig, variant: Literal["base", "adapter"]) -> str:
    evaluation = config.evaluation
    arguments = [
        f"pretrained={evaluation.base_model}",
        f"revision={evaluation.model_revision}",
        "dtype=float16",
        "load_in_4bit=True",
        "trust_remote_code=False",
    ]
    if variant == "adapter":
        arguments.append(f"peft={evaluation.adapter_path}")
    return ",".join(arguments)


def _extract_metrics(raw: dict[str, Any], tasks: list[str]) -> dict[str, dict[str, float]]:
    raw_results = raw.get("results", {})
    results = raw_results if isinstance(raw_results, dict) else {}
    extracted: dict[str, dict[str, float]] = {}
    for task in tasks:
        task_result = results.get(task, {})
        if not isinstance(task_result, dict):
            task_result = {}
        extracted[task] = {
            str(key): float(value)
            for key, value in task_result.items()
            if isinstance(value, (int, float)) and not str(key).endswith("_stderr,none")
        }
    return extracted


def _result_payload(
    config: AppConfig,
    variant: Literal["base", "adapter"],
    raw: dict[str, Any],
    *,
    evaluator_version: str,
) -> dict[str, Any]:
    evaluation = config.evaluation
    versions = raw.get("versions", {})
    return {
        "schema_version": "1.0",
        "variant": variant,
        "model": evaluation.base_model,
        "model_revision": evaluation.model_revision,
        "adapter_path": str(evaluation.adapter_path) if variant == "adapter" else None,
        "evaluator_version": evaluator_version,
        "task_versions": versions if isinstance(versions, dict) else {},
        "seed": evaluation.seed,
        "num_fewshot": evaluation.num_fewshot,
        "config_sha256": stable_json_hash(evaluation),
        "metrics": _extract_metrics(raw, evaluation.tasks),
        "raw": raw,
    }


def _evaluate_variant(
    config: AppConfig,
    variant: Literal["base", "adapter"],
) -> dict[str, Any]:
    from lm_eval import simple_evaluate

    evaluation = config.evaluation
    raw = simple_evaluate(
        model="hf",
        model_args=_model_args(config, variant),
        tasks=evaluation.tasks,
        num_fewshot=evaluation.num_fewshot,
        batch_size=evaluation.batch_size,
        device=evaluation.device,
        random_seed=evaluation.seed,
        numpy_random_seed=evaluation.seed,
        torch_random_seed=evaluation.seed,
        fewshot_random_seed=evaluation.seed,
        log_samples=evaluation.log_samples,
    )
    safe_raw = _json_safe(raw)
    if not isinstance(safe_raw, dict):
        raise TypeError("lm-eval must return a dictionary")
    return _result_payload(
        config,
        variant,
        safe_raw,
        evaluator_version=importlib.metadata.version("lm-eval"),
    )


def evaluate_models(config: AppConfig) -> dict[str, Any]:
    """Evaluate the base and adapter models into one inspectable artifact directory."""

    evaluation = config.evaluation
    require_new_artifact(evaluation.output_dir)
    if not os.environ.get("HF_TOKEN"):
        raise EnvironmentCheckError(
            "HF_TOKEN is required after accepting the gated Meta Llama 3.2 license"
        )
    if not evaluation.adapter_path.is_dir():
        raise FileNotFoundError(evaluation.adapter_path)
    with atomic_directory(evaluation.output_dir) as staging:
        base = _evaluate_variant(config, "base")
        adapter = _evaluate_variant(config, "adapter")
        write_json(staging / "base.json", base)
        write_json(staging / "adapter.json", adapter)
        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stage": "evaluation",
            "config_sha256": stable_json_hash(evaluation),
            "model": evaluation.base_model,
            "model_revision": evaluation.model_revision,
            "adapter_path": str(evaluation.adapter_path),
            "tasks": evaluation.tasks,
            "seed": evaluation.seed,
            "num_fewshot": evaluation.num_fewshot,
            "evaluator_version": base["evaluator_version"],
            "outputs": ["base.json", "adapter.json"],
            "assumption": "Zero-shot choice log likelihood is the configured protocol.",
        }
        write_json(staging / "manifest.json", manifest)
    return manifest
