# ArmBench-TextEmbed Repository Structure

## Overview

The repository is organized following best practices for evaluation benchmarks with clean separation of concerns and zero configuration duplication.

```
ArmBench-TextEmbed/
├── evaluate.sh                 # ← Shell script entry point (executable)
├── README.md                   # Quick start and usage guide
├── requirements.txt
├── STRUCTURE.md                # This file: repository organization
│
├── armbench/                   # Main Python package
│   ├── evaluate.py             # Orchestration script (loads config, dispatches tasks)
│   ├── models.py               # Embedding model wrapper
│   ├── utils.py                # Utilities (load_config, load_defaults, save_csv, etc.)
│   ├── sts_eval.py             # Semantic Textual Similarity evaluation
│   ├── retrieval_evaluation.py # Passage retrieval evaluation
│   ├── ms_marco_eval.py        # MS MARCO reranking evaluation
│   └── mteb_eval.py            # MTEB integration
│
├── configs/                    # Configuration (split for clarity)
│   ├── config.yml              # User-configurable parameters
│   ├── defaults.yml            # Internal constants (dataset IDs, algorithm params)
│   └── task_instructions.yml   # Task instruction strings for models
│
├── scripts/                    # Standalone utility scripts
│   └── prepare_ռեսւլտս.py       # Export results.json for HuggingFace upload
│
├── evaluation_results/         # Runtime output (gitignored)
│   ├── sts.csv
│   ├── retrieval.csv
│   ├── ms_marco.csv
│   ├── mteb.csv
│   └── <model_name>/
│       ├── mteb/               # MTEB official format
│       └── embeddings/         # Optional saved embeddings
│
└── [documentation files]
```

## Running the Benchmark

### Quick Start

```bash
# Direct shell script execution (recommended)
./evaluate.sh --config configs/config.yml

# Show help
./evaluate.sh --help

# Alternative: Python invocation
python -m armbench.evaluate --config configs/config.yml
```

### How It Works

1. **Shell Script Wrapper** (`./evaluate.sh`):
   - Sets up Python path to include `armbench/` package
   - Calls `armbench/evaluate.py:main()` with arguments

2. **Main Orchestration** (`armbench/evaluate.py`):
   - Parses command-line arguments
   - Loads and validates configuration
   - Instantiates embedding models
   - Dispatches to task-specific evaluation modules
   - Aggregates and saves results to CSV

3. **Task Modules**:
   - Each module loads its constants from `configs/defaults.yml`
   - Each module implements a single evaluation task
   - Results returned to orchestration for CSV export

## Output

Results are saved to `general.output_dir` as CSV files:

```
evaluation_results/
├── sts.csv                      # STS Pearson/Spearman scores
├── retrieval.csv                # Top-k accuracy metrics
├── retrieval_translit.csv       # Transliterated query results
├── ms_marco.csv                 # MRR and Top-k accuracy
├── mteb.csv                     # Per-task MTEB scores
└── <model_name>/
    ├── mteb/                    # Official MTEB format (for tooling)
    └── embeddings/              # Optional .npz files (if save_embeddings: true)
```

## Exporting for HuggingFace

Create a `results.json` for model card submission:

```bash
python scripts/prepare_ռեսւլտս.py --model "Metric-AI/armenian-text-embeddings-2-base"
```

## Documentation

- **[README.md](README.md)** - User guide and quick start
- **[STRUCTURE.md](STRUCTURE.md)** - This file: repository organization

