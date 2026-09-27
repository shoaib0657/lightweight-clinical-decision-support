# Lightweight Clinical Decision Support System

Research implementation accompanying **“Lightweight Clinical Decision Support System using QLoRA-Fine-Tuned LLMs and Retrieval-Augmented Generation.”** The project combines QLoRA fine-tuning of Llama 3.2 3B Instruct with hybrid BM25 and dense retrieval, citation-aware generation, and benchmark evaluation.

## Components

- Dataset preparation and validation for WikiDoc and MedQuAD
- Four-bit QLoRA adapter training
- Base-model and adapter evaluation with `lm-eval`
- Hybrid BM25/E5 retrieval and retrieval-augmented generation
- Synthetic clinical documents and queries for local retrieval checks

## Installation

Python 3.11 is required. Training and model evaluation require a compatible CUDA GPU.

```bash
python -m venv .venv
python -m pip install --upgrade pip
pip install -e ".[train,eval,rag]"
```

The default experiment settings are recorded in [`configs/default.yaml`](configs/default.yaml), including pinned dataset and model revisions.

## Data sources

- [WikiDoc / Medical Meadow WikiDoc](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc) — 10,000 question-answer records
- [MedQuAD](https://github.com/abachaa/MedQuAD) — source repository; this project is configured for a 16,412-record subset

The source datasets are downloaded from their original repositories and are not redistributed here. Users must comply with their respective licenses and terms.

## Reproducibility note

This repository provides the implementation and configuration associated with the methodology described in the manuscript. The original experiments were conducted on an institutional GPU environment that is no longer available; therefore, runtime and numerical results may vary with hardware and software configuration.

## Disclaimer

This software is provided for research and educational purposes only. It is not a medical device and must not be used for diagnosis, treatment, or other clinical decisions without review by qualified healthcare professionals.
