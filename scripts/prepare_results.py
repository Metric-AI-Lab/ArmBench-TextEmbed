"""
Script to prepare evaluation data from CSV files and export as JSON for HuggingFace space.
Run this script locally before pushing to HuggingFace.

Exports all detailed metrics so no CSVs need to be pushed.

Output format:
[
    {
        "model_name": "...",
        "results": {
            "mteb_avg": ...,
            "mteb_detailed": {"flores": ..., "ntrex": ..., ...},
            "msmarco_detailed": {"reranking_mrr": ..., ...},
            ...
        }
    },
    ...
]

Usage:
    # Export model to results.json
    python prepare_data.py --model "org/model-name"

    # Export model to custom path
    python prepare_data.py --model "org/model-name" --output path/to/results.json
"""
import json
import pandas as pd
import argparse
from pathlib import Path


MTEB_TEST_COLS = [
    "FloresBitextMining_devtest",
    "NTREXBitextMining_test",
    "Tatoeba_test",
    "MassiveIntentClassification_test",
    "MassiveScenarioClassification_test",
    "SIB200Classification_test",
    "SIB200ClusteringS2S_test",
    "ArmenianParaphrasePC_test",
    "BelebeleRetrieval_test",
]

# MS MARCO columns
MSMARCO_COLS = [
    "reranking_mrr",
    "retrieval_mrr",
    "retrieval_top5_accuracy",
    "retrieval_top10_accuracy",
]

# Retrieval translit columns
RETRIEVAL_TRANSLIT_COLS = [
    "top1 within document",
    "top3 within document",
    "top5 within document",
    "top20 group mean macro",
    "top20 all",
]

# STS columns
STS_COLS = [
    "Pearson_correlation",
    "Spearman_correlation",
]

# Retrieval columns (native script)
RETRIEVAL_COLS = [
    "top1 within document",
    "top3 within document",
    "top5 within document",
    "top20 group mean macro",
    "top20 all",
]


def load_and_clean_csv(filepath):
    """Load CSV and drop duplicate rows."""
    try:
        df = pd.read_csv(filepath)
        df = df.drop_duplicates(subset=['model_name'], keep='last')
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()


def extract_single_model_metrics(base_path, model_name):
    """Extract metrics for a single model from CSV files."""
    # Load all CSVs
    mteb_df = load_and_clean_csv(base_path / "mteb.csv")
    sts_df = load_and_clean_csv(base_path / "sts.csv")
    retrieval_df = load_and_clean_csv(base_path / "retrieval.csv")
    retrieval_translit_df = load_and_clean_csv(base_path / "retrieval_translit.csv")
    ms_marco_df = load_and_clean_csv(base_path / "ms_marco.csv")
    ms_marco_translit_df = load_and_clean_csv(base_path / "ms_marco_translit.csv")

    results = {}

    # MTEB - average and detailed scores
    if not mteb_df.empty and model_name in mteb_df['model_name'].values:
        model_row = mteb_df[mteb_df['model_name'] == model_name].iloc[0]
        available_cols = [c for c in MTEB_TEST_COLS if c in model_row.index]
        if available_cols:
            results["mteb_avg"] = round(model_row[available_cols].mean(), 4)
            # Add detailed MTEB scores
            mteb_detailed = {}
            for col in available_cols:
                mteb_detailed[col] = round(float(model_row[col]), 4)
            results["mteb_detailed"] = mteb_detailed

    # STS - Spearman correlation and detailed
    if not sts_df.empty and model_name in sts_df['model_name'].values:
        model_row = sts_df[sts_df['model_name'] == model_name].iloc[0]
        if 'Spearman_correlation' in model_row.index:
            results["sts_spearman"] = round(model_row['Spearman_correlation'], 4)
        # Add detailed STS scores
        sts_detailed = {}
        for col in STS_COLS:
            if col in model_row.index:
                sts_detailed[col] = round(float(model_row[col]), 4)
        if sts_detailed:
            results["sts_detailed"] = sts_detailed

    # Retrieval - top20 all and detailed
    if not retrieval_df.empty and model_name in retrieval_df['model_name'].values:
        model_row = retrieval_df[retrieval_df['model_name'] == model_name].iloc[0]
        if 'top20 all' in model_row.index:
            results["retrieval_top20"] = round(model_row['top20 all'], 4)
        # Add detailed retrieval scores
        retrieval_detailed = {}
        for col in RETRIEVAL_COLS:
            if col in model_row.index:
                retrieval_detailed[col] = round(float(model_row[col]), 4)
        if retrieval_detailed:
            results["retrieval_detailed"] = retrieval_detailed

    # MS MARCO - all detailed scores
    if not ms_marco_df.empty and model_name in ms_marco_df['model_name'].values:
        model_row = ms_marco_df[ms_marco_df['model_name'] == model_name].iloc[0]
        if 'retrieval_top10_accuracy' in model_row.index:
            results["msmarco_top10"] = round(model_row['retrieval_top10_accuracy'], 4)
        # Add detailed MS MARCO scores
        msmarco_detailed = {}
        for col in MSMARCO_COLS:
            if col in model_row.index:
                msmarco_detailed[col] = round(float(model_row[col]), 4)
        if msmarco_detailed:
            results["msmarco_detailed"] = msmarco_detailed

    # Retrieval Transliterated - all detailed scores
    if not retrieval_translit_df.empty and model_name in retrieval_translit_df['model_name'].values:
        model_row = retrieval_translit_df[retrieval_translit_df['model_name'] == model_name].iloc[0]
        if 'top20 all' in model_row.index:
            results["retrieval_translit_top20"] = round(model_row['top20 all'], 4)
        # Add detailed retrieval translit scores
        retrieval_translit_detailed = {}
        for col in RETRIEVAL_TRANSLIT_COLS:
            if col in model_row.index:
                retrieval_translit_detailed[col] = round(float(model_row[col]), 4)
        if retrieval_translit_detailed:
            results["retrieval_translit_detailed"] = retrieval_translit_detailed

    # MS MARCO Transliterated - all detailed scores
    if not ms_marco_translit_df.empty and model_name in ms_marco_translit_df['model_name'].values:
        model_row = ms_marco_translit_df[ms_marco_translit_df['model_name'] == model_name].iloc[0]
        if 'retrieval_top10_accuracy' in model_row.index:
            results["msmarco_translit_top10"] = round(model_row['retrieval_top10_accuracy'], 4)
        # Add detailed MS MARCO translit scores
        msmarco_translit_detailed = {}
        for col in MSMARCO_COLS:
            if col in model_row.index:
                msmarco_translit_detailed[col] = round(float(model_row[col]), 4)
        if msmarco_translit_detailed:
            results["msmarco_translit_detailed"] = msmarco_translit_detailed

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract evaluation data for a single model to JSON")
    parser.add_argument("--model", type=str, required=True, help="Model name exactly as in CSV (e.g., 'org/model-name')")
    parser.add_argument("--output", type=str, default=None, help="Output path (defaults to results.json in current directory)")
    args = parser.parse_args()

    base_path = Path(__file__).parent / "evaluation_results"
    results = extract_single_model_metrics(base_path, args.model)

    if not results:
        print(f"Error: No results found for model '{args.model}'")
        return

    output_path = Path(args.output) if args.output else Path.cwd() / "results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results exported to {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
