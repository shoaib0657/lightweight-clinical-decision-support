"""Minimal, research-only clinical language model and retrieval pipeline."""

from lightweight_cds.config import load_config
from lightweight_cds.data import prepare_dataset
from lightweight_cds.documents import build_index
from lightweight_cds.evaluation import evaluate_models
from lightweight_cds.generation import answer_query, evaluate_rag
from lightweight_cds.training import train_adapter

__all__ = [
    "load_config",
    "prepare_dataset",
    "train_adapter",
    "evaluate_models",
    "build_index",
    "answer_query",
    "evaluate_rag",
]

__version__ = "0.2.0"
