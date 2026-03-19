"""Retrieval evaluation module."""

from __future__ import annotations

import os
import torch

if not torch.cuda.is_available():
    os.environ["XFORMERS_DISABLED"] = "1"

import logging

import numpy as np
import polars as pl
from torch.amp import autocast

from models import Embedding_Model
import utils
from huggingface_hub import hf_hub_download, list_repo_files

# Load retrieval-specific constants from defaults.yml
_defaults = utils.load_defaults()
_retrieval_defaults = _defaults.get("tasks", {}).get("retrieval", {})
DEFAULT_DATASET_NAME = _retrieval_defaults.get("dataset", "Metric-AI/retrieval_dataset_hye")
MIN_PASSAGES_THRESHOLD = _retrieval_defaults.get("min_passages_threshold", 8)
DEFAULT_MAX_LENGTH = 512  # Fallback only; actual value comes from model config

logger = logging.getLogger(__name__)


def save_retrieval_embeddings(embeddings: list[dict], path: str) -> None:
    """Save retrieval embeddings to a .npz file preserving structure."""
    np.savez(
        path,
        query_embeddings=np.array([e["query_embedding"] for e in embeddings]),
        passage_embeddings=np.array([e["passage_embeddings"] for e in embeddings], dtype=object),
        labels=np.array([e["labels"] for e in embeddings], dtype=object),
        query_indices=np.array([e["query_idx"] for e in embeddings]),
    )
    logger.info(f"Saved retrieval embeddings to {path}")


def load_retrieval_embeddings(path: str) -> list[dict]:
    """Load retrieval embeddings from a .npz file."""
    data = np.load(path, allow_pickle=True)
    embeddings = []
    for i in range(len(data["query_indices"])):
        embeddings.append({
            "query_embedding": data["query_embeddings"][i].tolist(),
            "passage_embeddings": data["passage_embeddings"][i].tolist(),
            "labels": data["labels"][i].tolist(),
            "query_idx": data["query_indices"][i],
        })
    logger.info(f"Loaded retrieval embeddings from {path}")
    return embeddings


def evaluate_retrieval(
    model: Embedding_Model,
    dataset_name: str = DEFAULT_DATASET_NAME,
    eval_config: dict | None = None,
    save_embeddings_path: str | None = None,
    **kwargs,
) -> dict[str, float]:
    """Evaluate embedding model on retrieval benchmark."""
    translit = kwargs.get("translit", False)

    logger.info("Loading dataset: %s", dataset_name)
    all_files = list_repo_files(repo_id=dataset_name, repo_type="dataset")
    target_files = [f for f in all_files if f.startswith("data/train-") and f.endswith(".parquet")]

    local_paths = [
        hf_hub_download(repo_id=dataset_name, filename=f, repo_type="dataset")
        for f in target_files
    ]
    df = pl.read_parquet(local_paths)

    logger.info("Processing %d queries", len(df))

    embeddings = _compute_embeddings_batched(df, model, eval_config, translit)

    if save_embeddings_path is not None:
        save_retrieval_embeddings(embeddings, save_embeddings_path)

    results = _compute_metrics_from_embeddings(embeddings, df)

    return results

def _compute_embeddings_batched(
    df: pl.DataFrame,
    model: Embedding_Model,
    eval_config: str | None = None,
    translit: bool = False,
) -> list[dict]:
    """Compute embeddings."""
    query_prefix, passage_prefix = utils._build_query_prefix(eval_config)

    all_queries = []
    all_passages = []
    passage_offsets = [0]
    rows_data = []

    for row in df.iter_rows(named=True):
        query_text = row["query"] if not translit else row["translit_query"]
        passage_texts = row["passages"]

        all_queries.append(f"{query_prefix}{query_text}")
        all_passages.extend([f"{passage_prefix}{p}" for p in passage_texts])
        passage_offsets.append(len(all_passages))
        rows_data.append({"labels": row["labels"]})

    logger.info("Encoding %d queries and %d passages", len(all_queries), len(all_passages))
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast(device_type="cuda", dtype=dtype):
            query_embeddings = model.encode(all_queries)
            passage_embeddings = model.encode(all_passages)
    else:
        query_embeddings = model.encode(all_queries)
        passage_embeddings = model.encode(all_passages)

    embeddings_data = []
    for i, row_data in enumerate(rows_data):
        start_idx = passage_offsets[i]
        end_idx = passage_offsets[i + 1]

        embeddings_data.append(
            {
                "query_embedding": query_embeddings[i].tolist(),
                "passage_embeddings": passage_embeddings[start_idx:end_idx].tolist(),
                "labels": row_data["labels"],
                "query_idx": i,
            }
        )

    return embeddings_data


def _compute_metrics_from_embeddings(
    embeddings: list[dict],
    df: pl.DataFrame,
) -> dict[str, float]:
    """Compute retrieval metrics from embeddings."""
    all_queries = [np.array(item["query_embedding"]) for item in embeddings]
    all_passages = [np.array(item["passage_embeddings"]) for item in embeddings]
    all_labels = [item["labels"] for item in embeddings]

    results = {}

    for k in [5, 3, 1]:
        score = _compute_top_k_accuracy(all_queries, all_passages, all_labels, k)
        results[f"top{k} within document"] = score

    grouped_scores = []
    for group_type in df["type"].unique().to_list():
        group_indices = (
            df.with_row_index().filter(pl.col("type") == group_type)["index"].to_list()
        )
        score = _compute_group_accuracy(
            all_queries, all_passages, all_labels, group_indices, k=20
        )
        grouped_scores.append(score)

    results["top20 group mean macro"] = np.mean(grouped_scores)
    results["top20 all"] = _compute_group_accuracy(
        all_queries, all_passages, all_labels, list(range(len(all_queries))), k=20
    )
    return results


def _compute_top_k_accuracy(
    all_queries: list[np.ndarray],
    all_passages: list[np.ndarray],
    all_labels: list[list[int]],
    k: int,
) -> float:
    """Compute top-k accuracy within each document's passages."""
    true = 0
    all_count = 0

    for i in range(len(all_queries)):
        if len(all_passages[i]) >= MIN_PASSAGES_THRESHOLD:
            query_emb = torch.tensor(all_queries[i])
            passage_embs = torch.tensor(all_passages[i])
            labels = all_labels[i]

            query_emb = query_emb.unsqueeze(0)
            scores = query_emb @ passage_embs.T
            _, topk_indices = torch.topk(scores, k, dim=1)

            for j in topk_indices[0]:
                if labels[j] != 0:
                    true += 1
                    break
            all_count += 1

    return true / all_count if all_count > 0 else 0.0


def _compute_group_accuracy(
    all_queries: list[np.ndarray],
    all_passages: list[np.ndarray],
    all_labels: list[list[int]],
    group_indices: list[int],
    k: int = 20,
) -> float:
    """Compute top-k accuracy across a group of documents."""
    true = 0
    all_count = 0

    group_passages = []
    for idx in group_indices:
        if idx < len(all_queries):
            group_passages.extend(all_passages[idx])

    group_passages_array = np.array(group_passages)
    group_passages_tensor = torch.tensor(group_passages_array)

    for idx in group_indices:
        if idx < len(all_queries):
            query_emb = torch.tensor(all_queries[idx])

            labels_cat = []
            for group_idx in group_indices:
                if group_idx < len(all_queries):
                    group_query_passage_count = len(all_passages[group_idx])
                    if group_idx == idx:
                        labels_cat.extend(all_labels[group_idx])
                    else:
                        labels_cat.extend([0] * group_query_passage_count)
            labels_cat = torch.tensor(labels_cat)

            query_emb = query_emb.unsqueeze(0)
            scores = query_emb @ group_passages_tensor.T
            actual_k = min(k, len(group_passages))
            _, topk_indices = torch.topk(scores, actual_k, dim=1)

            for j in topk_indices[0]:
                if labels_cat[j] != 0:
                    true += 1
                    break
            all_count += 1

    return true / all_count if all_count > 0 else 0.0
