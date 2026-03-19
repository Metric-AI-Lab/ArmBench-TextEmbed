"""MS MARCO evaluation module."""

from __future__ import annotations

import os

import torch

if not torch.cuda.is_available():
    os.environ["XFORMERS_DISABLED"] = "1"

import logging

import numpy as np
from datasets import load_dataset
from torch.amp import autocast

from models import Embedding_Model
import utils

# Load MS MARCO dataset from defaults.yml
_defaults = utils.load_defaults()
_msmarco_defaults = _defaults.get("tasks", {}).get("ms_marco", {})
DEFAULT_DATASET_NAME = _msmarco_defaults.get("dataset", "Metric-AI/ms_marco_hye")
# Fallback values (actual values come from config.yml)
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32

logging.basicConfig(level=logging.INFO)


def save_ms_marco_embeddings(embeddings: list[dict], path: str) -> None:
    """Save MS MARCO embeddings to a .npz file preserving structure."""
    np.savez(
        path,
        query_ids=[e["query_id"] for e in embeddings],
        query_embeddings=np.array([e["query_embedding"] for e in embeddings]),
        passage_embeddings=np.array([e["passage_embeddings"] for e in embeddings], dtype=object),
        labels=np.array([e["labels"] for e in embeddings], dtype=object),
        query_indices=[e["query_idx"] for e in embeddings],
    )
    logging.info(f"Saved MS MARCO embeddings to {path}")


def load_ms_marco_embeddings(path: str) -> list[dict]:
    """Load MS MARCO embeddings from a .npz file."""
    data = np.load(path, allow_pickle=True)
    embeddings = []
    for i in range(len(data["query_ids"])):
        embeddings.append({
            "query_id": data["query_ids"][i],
            "query_embedding": data["query_embeddings"][i].tolist(),
            "passage_embeddings": data["passage_embeddings"][i].tolist(),
            "labels": data["labels"][i].tolist(),
            "query_idx": data["query_indices"][i],
        })
    logging.info(f"Loaded MS MARCO embeddings from {path}")
    return embeddings


def evaluate_ms_marco(
    model: Embedding_Model,
    dataset_name: str = DEFAULT_DATASET_NAME,
    eval_config: dict | None = None,
    save_embeddings_path: str | None = None,
    **kwargs,
) -> dict[str, float]:
    """Evaluate embedding model on MS MARCO benchmark."""
    translit = kwargs.get("translit", False)

    logging.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="validation")

    logging.info(f"Processing {len(dataset)} queries")

    embeddings = _compute_embeddings(dataset, model, eval_config, translit)

    if save_embeddings_path is not None:
        save_ms_marco_embeddings(embeddings, save_embeddings_path)

    results = _compute_metrics_from_embeddings(embeddings)

    return results


def _compute_embeddings(
    dataset,
    model: Embedding_Model,
    eval_config: dict | None = None,
    translit: bool = False,
) -> list[dict]:
    """Compute embeddings for queries and passages."""
    query_prefix, passage_prefix = utils._build_query_prefix(eval_config)

    all_queries = [
        f"{query_prefix}{example['query']}"
        if not translit
        else f"{query_prefix}{example['translit_query']}"
        for example in dataset
    ]

    passage_lists = [
        [f"{passage_prefix}{p}" for p in example["passages"]]
        for example in dataset
    ]
    all_passages = [p for plist in passage_lists for p in plist]

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast(device_type="cuda", dtype=dtype):
            query_embeddings = model.encode(all_queries)
            passage_embeddings = model.encode(all_passages)
    else:
        query_embeddings = model.encode(all_queries)
        passage_embeddings = model.encode(all_passages)

    embeddings_data = []
    passage_idx = 0
    for i, example in enumerate(dataset):
        num_passages = len(passage_lists[i])
        embeddings_data.append(
            {
                "query_id": example["query_id"],
                "query_embedding": query_embeddings[i].tolist(),
                "passage_embeddings": passage_embeddings[
                    passage_idx : passage_idx + num_passages
                ].tolist(),
                "labels": example["is_selected"],
                "query_idx": i,
            }
        )
        passage_idx += num_passages

    return embeddings_data

def _compute_metrics_from_embeddings(
    embeddings: list[dict],
) -> dict[str, float]:
    """Compute MS MARCO metrics from embeddings."""
    all_queries = [np.array(item["query_embedding"]) for item in embeddings]
    all_passages = [np.array(item["passage_embeddings"]) for item in embeddings]
    all_labels = [item["labels"] for item in embeddings]

    logging.info(f"Computing metrics for {len(all_queries)} queries...")

    reranking_mrr = _compute_reranking_mrr(all_queries, all_passages, all_labels)
    retrieval_mrr, retrieval_top5_acc, retrieval_top10_acc = _compute_retrieval_metrics(
        all_queries, all_passages, all_labels
    )

    results = {
        "reranking_mrr": reranking_mrr,
        "retrieval_mrr": retrieval_mrr,
        "retrieval_top5_accuracy": retrieval_top5_acc,
        "retrieval_top10_accuracy": retrieval_top10_acc,
    }

    return results

def _compute_reranking_mrr(all_queries, all_passages, all_labels):
    mrr_scores = []

    for i in range(len(all_queries)):
        query_emb = torch.tensor(all_queries[i]).unsqueeze(0)
        passage_embs = torch.tensor(all_passages[i])
        labels = all_labels[i]

        scores = query_emb @ passage_embs.T
        ranked_indices = torch.argsort(scores, dim=1, descending=True)[0].cpu().numpy()

        mrr = 0.0
        for rank, passage_idx in enumerate(ranked_indices, start=1):
            passage_idx = int(passage_idx)
            if passage_idx < len(labels) and labels[passage_idx] == 1:
                mrr = 1.0 / rank
                break

        mrr_scores.append(mrr)

    return np.mean(mrr_scores)

def _compute_retrieval_metrics(all_queries, all_passages, all_labels):
    corpus_embeddings = np.vstack(all_passages)
    passage_to_query = np.repeat(
        np.arange(len(all_passages)), [len(p) for p in all_passages]
    )
    passage_relevance = np.concatenate(all_labels)

    logging.info(
        f"Retrieval: {len(all_queries)} queries × {len(corpus_embeddings)} passages corpus"
    )

    query_relevant_passages = {}
    for passage_idx, (query_idx, relevance) in enumerate(
        zip(passage_to_query, passage_relevance)
    ):
        if relevance == 1:
            if query_idx not in query_relevant_passages:
                query_relevant_passages[query_idx] = set()
            query_relevant_passages[query_idx].add(passage_idx)

    all_query_embs = torch.tensor(np.stack(all_queries))
    corpus_embs_tensor = torch.tensor(corpus_embeddings)

    all_similarities = all_query_embs @ corpus_embs_tensor.T

    mrr_scores = []
    top5_hits = []
    top10_hits = []

    for query_idx in range(len(all_queries)):
        ranked_passage_indices = (
            torch.argsort(all_similarities[query_idx], descending=True).cpu().numpy()
        )

        relevant_passages = query_relevant_passages.get(query_idx, set())

        mrr = 0.0
        for rank, passage_idx in enumerate(ranked_passage_indices, start=1):
            passage_idx = int(passage_idx)
            if passage_idx in relevant_passages:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)

        top5_indices = ranked_passage_indices[:5]
        hit_top5 = any(int(idx) in relevant_passages for idx in top5_indices)
        top5_hits.append(1.0 if hit_top5 else 0.0)

        top10_indices = ranked_passage_indices[:10]
        hit_top10 = any(int(idx) in relevant_passages for idx in top10_indices)
        top10_hits.append(1.0 if hit_top10 else 0.0)

    return np.mean(mrr_scores), np.mean(top5_hits), np.mean(top10_hits)
