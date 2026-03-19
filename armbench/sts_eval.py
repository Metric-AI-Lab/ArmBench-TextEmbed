"""Semantic Textual Similarity (STS) evaluation module."""

from __future__ import annotations

import os
import torch
import logging

import numpy as np
from datasets import Dataset, load_dataset
from scipy.stats import pearsonr, spearmanr

from models import Embedding_Model
import utils
from torch.amp import autocast
if not torch.cuda.is_available():
    os.environ["XFORMERS_DISABLED"] = "1"

# Load STS-specific algorithm constants from defaults.yml
_defaults = utils.load_defaults()
_sts_defaults = _defaults.get("tasks", {}).get("sts", {})
STS_SCALE_MAX = _sts_defaults.get("scale_max", 5.0)
EPSILON = _sts_defaults.get("epsilon", 1e-8)
DEFAULT_MAX_LENGTH = 512  # Fallback only; actual value comes from model config

logging.basicConfig(level=logging.INFO)


def save_sts_embeddings(dataset: Dataset, path: str) -> None:
    """Save STS embeddings to a .npz file preserving structure."""
    embeddings1 = np.array([ex["embedding1"] for ex in dataset])
    embeddings2 = np.array([ex["embedding2"] for ex in dataset])
    scores = np.array([ex["score"] for ex in dataset])

    np.savez(
        path,
        embeddings1=embeddings1,
        embeddings2=embeddings2,
        scores=scores,
    )
    logging.info(f"Saved STS embeddings to {path}")


def load_sts_embeddings(path: str) -> Dataset:
    """Load STS embeddings from a .npz file."""
    data = np.load(path, allow_pickle=True)
    data_list = []
    for i in range(len(data["scores"])):
        data_list.append({
            "embedding1": data["embeddings1"][i].tolist(),
            "embedding2": data["embeddings2"][i].tolist(),
            "score": float(data["scores"][i]),
        })
    logging.info(f"Loaded STS embeddings from {path}")
    return Dataset.from_list(data_list)


def evaluate_sts(
    model: Embedding_Model,
    dataset_name: str = "Metric-AI/mteb_sts_hye",
    eval_config: dict | None = None,
    save_embeddings_path: str | None = None,
    **kwargs
) -> dict[str, float]:
    """Evaluate embedding model on STS benchmark."""
    logging.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    ds = dataset["train"]

    logging.info(f"Processing {len(ds)} samples")

    dataset_with_embeddings = _add_embeddings_to_dataset(ds, model, eval_config)

    if save_embeddings_path is not None:
        save_sts_embeddings(dataset_with_embeddings, save_embeddings_path)

    results = _compute_metrics(dataset_with_embeddings)

    return results


def _add_embeddings_to_dataset(
    dataset: Dataset,
    model: Embedding_Model,
    eval_config: dict | None = None,
) -> Dataset:
    """Encode sentence pairs and add embeddings to dataset."""
    data_list = list(dataset)
    query_prefix, passage_prefix = utils._build_query_prefix(eval_config)

    sentences1 = [f"{query_prefix}{ex['sentence1']}" for ex in data_list]
    sentences2 = [f"{passage_prefix}{ex['sentence2']}" for ex in data_list]
    
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with autocast(device_type="cuda", dtype=dtype):
            embeddings1 = model.encode(sentences1, convert_to_numpy=True)
            embeddings2 = model.encode(sentences2, convert_to_numpy=True)
    else:
        embeddings1 = model.encode(sentences1, convert_to_numpy=True)
        embeddings2 = model.encode(sentences2, convert_to_numpy=True)

    for i, ex in enumerate(data_list):
        ex["embedding1"] = embeddings1[i].tolist()
        ex["embedding2"] = embeddings2[i].tolist()

    return Dataset.from_list(data_list)


def _compute_metrics(dataset: Dataset) -> dict[str, float]:
    """Compute Pearson and Spearman correlations from embedded dataset."""
    embeddings1 = np.array([ex["embedding1"] for ex in dataset])
    embeddings2 = np.array([ex["embedding2"] for ex in dataset])
    scores = [ex["score"] for ex in dataset]

    cos_similarities = _cosine_similarity_paired(embeddings1, embeddings2)
    scaled_similarities = STS_SCALE_MAX * (cos_similarities + 1) / 2

    pearson_corr, _ = pearsonr(scaled_similarities, scores)
    spearman_corr, _ = spearmanr(scaled_similarities, scores)

    return {
        "Pearson_correlation": pearson_corr,
        "Spearman_correlation": spearman_corr,
    }


def _cosine_similarity_paired(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute element-wise cosine similarity between paired vectors.
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + EPSILON)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + EPSILON)
    return np.sum(a_norm * b_norm, axis=1)
