# ArmBench-TextEmbed

A comprehensive evaluation benchmark for text embedding models on Armenian language tasks. ArmBench-TextEmbed provides standardized evaluation across semantic textual similarity, passage retrieval, and reranking benchmarks.

## Overview

ArmBench-TextEmbed evaluates embedding models on three core tasks, plus integration with the official MTEB benchmark:

| Task | Dataset | Metrics |
|------|---------|---------|
| **STS** | [Metric-AI/mteb_sts_hye](https://huggingface.co/datasets/Metric-AI/mteb_sts_hye) | Pearson & Spearman correlation |
| **Retrieval** | [Metric-AI/retrieval_dataset_hye](https://huggingface.co/datasets/Metric-AI/retrieval_dataset_hye) | Top-k accuracy (k=1,3,5,20) |
| **MS MARCO** | [Metric-AI/ms_marco_hye](https://huggingface.co/datasets/Metric-AI/ms_marco_hye) | MRR, Top-5/10 accuracy |
| **MTEB** | [Armenian subset of MTEB](https://github.com/embeddings-benchmark/mteb) | Bitext mining, classification, clustering, paraphrase, retrieval |

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

Tested on Python 3.12 and PyTorch 2.10+. When installing PyTorch, make sure to account for your CUDA version.

## Quick Start

1. **Configure your models** in `configs/config.yml`:

```yaml
models:
  - name: ate-2-base
    model_name: "Metric-AI/armenian-text-embeddings-2-base"
    max_length: 512

general:
  batch_size: 64
  output_dir: "evaluation_results"
```

2. **Run evaluation**:

```bash
./evaluate.sh --config configs/config.yml
```

Results are saved to `evaluation_results/` as CSV files.

## Configuration

### Structure

Configuration is split into two files for clarity:

- **`configs/config.yml`** - User-configurable parameters (models, tasks, runtime settings)
- **`configs/defaults.yml`** - Internal algorithm constants and dataset identifiers

### Full Configuration Example

```yaml
evaluation:
  instructions: "configs/task_instructions.yml"
  tasks:
    - name: retrieval
      prompts:
        query_prefix: "query: "
        passage_prefix: "passage: "
    - name: sts
    - name: ms_marco
      prompts:
        query_prefix: "query: "
        passage_prefix: "passage: "
    - name: mteb

models:
  - name: ate-2-base
    model_name: "Metric-AI/armenian-text-embeddings-2-base"
    max_length: 512

general:
  batch_size: 32
  output_dir: "evaluation_results"
  translit: false
  save_embeddings: false
```

### Model Configuration

ArmBench-TextEmbed supports any HuggingFace model, including SentenceTransformers and standard transformers:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `name` | Yes | Display name for results |
| `model_name` | Yes | HuggingFace model ID or local path |
| `max_length` | Yes | Maximum sequence length |
| `pooling` | No* | Pooling strategy: `mean`, `cls_token`, or `last_token` |

*Required for standard transformer models (not SentenceTransformers)

### Task Configuration

Each task supports optional prompt prefixes for instruction-tuned models:

```yaml
- name: retrieval
  prompts:
    query_prefix: "query: "      # Prepended to queries
    passage_prefix: "passage: "  # Prepended to passages
    instruct_prefix: "instruction: "  # For instruction-tuned models
```

## Evaluation Tasks

### Semantic Textual Similarity (STS)

Evaluates the model's ability to capture semantic similarity between Armenian sentence pairs. Reports Pearson and Spearman correlations between predicted cosine similarities and human-annotated scores.

### Retrieval

Evaluates passage retrieval from chunked documents. Metrics include:
- **Top-k within document**: Accuracy of retrieving relevant passages within the same document
- **Top-20 group mean macro**: Average accuracy across document type groups
- **Top-20 all**: Global retrieval accuracy across all passages

### MS MARCO (Armenian)

Armenian-translated MS MARCO benchmark measuring:
- **Reranking MRR**: Mean Reciprocal Rank for reranking passages per query
- **Retrieval MRR**: MRR across the full corpus
- **Top-5/10 Accuracy**: Hit rate in top-k retrieved passages

### MTEB (Armenian Subset)

Integrates with the official [MTEB benchmark](https://github.com/embeddings-benchmark/mteb) to run all available Armenian (`hye`) language tasks. Results are saved in two formats:
- **MTEB-compatible format**: Full results under `evaluation_results/<model_name>/mteb/` for compatibility with MTEB tooling
- **CSV summary**: Main scores saved to the standard CSV output alongside other task results

## Output

Results are saved as CSV files in the output directory:

```
evaluation_results/
├── sts.csv
├── retrieval.csv
├── ms_marco.csv
├── retrieval_translit.csv      # If translit: true
└── <model_name>/
    ├── mteb/                   # MTEB results in official format (if mteb task enabled)
    └── embeddings/             # Saved embeddings (if save_embeddings: true)
        ├── sts_embeddings.npz
        ├── retrieval_embeddings.npz
        ├── ms_marco_embeddings.npz
        └── *_translit_embeddings.npz  # If translit: true
```

Each CSV contains model names and corresponding metrics for easy comparison. MTEB results are saved in both the official MTEB format and as CSV summaries.

## Publishing Results to HuggingFace

Generate a `results.json` file from your evaluation results using the export script:

```bash
python scripts/prepare_results.py --model "Metric-AI/armenian-text-embeddings-2-base"
```

Upload the generated `results.json` to your model repository with the `ArmBench-TextEmbed` tag to appear on the leaderboard.

## Embedding Storage

Enable embedding saving with `save_embeddings: true` in your config. Embeddings are saved in `.npz` format per task, with `load_embeddings` functions available in each task module for easy loading.

## Transliteration Support

For models that may perform differently on transliterated Armenian text, enable transliteration evaluation:

```yaml
general:
  translit: true
```

This runs additional evaluations on retrieval and MS MARCO tasks using transliterated queries.

## Documentation

For detailed information about the repository structure and configuration, see:

- **[STRUCTURE.md](STRUCTURE.md)** - Repository organization and how everything works

## Citation

If you use ArmBench-TextEmbed in your research, please cite:

```bibtex
@misc{armbench-textembed,
  title={ArmBench-TextEmbed: A Benchmark for Armenian Text Embedding Models},
  year={2026},
  url={https://github.com/Metric-AI-Lab/ArmBench-TextEmbed}
}
@inproceedings{navasardyan2026lessismore,
  title = {Less is More: Adapting Text Embeddings for Low-Resource Languages with Small Scale Noisy Synthetic Data},
  author = {Navasardyan, Zaruhi and Bughdaryan, Spartak and Minasyan, Bagratuni and Davtyan, Hrant},
  booktitle = {Proceedings of the Workshop on Language Models for Low-Resource Languages (LoResLM) at EACL 2026},
  year = {2026}
}
```
