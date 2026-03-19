"""Utility functions for evaluation."""

import os
import csv
import logging
import warnings
from typing import Any, Dict, List
import yaml


def _build_query_prefix(eval_config: dict | None) -> str:
    """Build instruction prefix for query encoding."""
    if eval_config is None:
        return "", ""
    if eval_config.get("prompts"):
        instruct_prefix = eval_config["prompts"].get("instruct_prefix", "")
        query_prefix = eval_config["prompts"].get("query_prefix", "")
        instruction = eval_config["prompts"].get("instruction", "")
        passage_prefix = eval_config["prompts"].get("passage_prefix", "")
        if instruct_prefix:
            return f"{instruct_prefix}{instruction}\n{query_prefix} ", passage_prefix
        return query_prefix, passage_prefix
    return "", ""


def ensure_output_dir(path: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_evaluation_to_csv(
    eval_name: str,
    model_name: str,
    metrics: Dict[str, Any],
    output_dir: str,
    model_path: str = None
) -> None:
    """Save evaluation metrics to a CSV file."""
    ensure_output_dir(output_dir)
    csv_path = os.path.join(output_dir, f"{eval_name}.csv")

    row = {
        "model_name": model_path if model_path else model_name,
        "name": model_name,
        **metrics,
    }

    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0

    if file_exists:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
        fieldnames = existing_fields + [k for k in row if k not in existing_fields]
    else:
        fieldnames = list(row.keys())

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logging.info("Saved %s results to %s", eval_name, csv_path)


def load_defaults(defaults_path: str = "configs/defaults.yml") -> Dict[str, Any]:
    """Load default parameters from YAML file."""
    if not os.path.exists(defaults_path):
        raise FileNotFoundError(f"Defaults file {defaults_path} not found.")

    with open(defaults_path, 'r') as f:
        data = yaml.safe_load(f)

    return data.get("defaults", {}) if data else {}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_model_config(models_config: List[Dict[str, Any]]) -> None:
    """Validate that all required keys are present in model configurations."""
    if not models_config:
        raise ValueError("No models defined in config file")

    required_keys = ["name", "model_name", "max_length"]
    for idx, model_cfg in enumerate(models_config):
        for key in required_keys:
            if key not in model_cfg:
                raise ValueError(f"Model config at index {idx} is missing required key: '{key}'")


def validate_evaluation_config(eval_config: List[Dict[str, Any]]) -> None:
    """Validate that evaluation tasks are among allowed types."""
    allowed_tasks = {"sts", "ms_marco", "retrieval", "mteb"}

    if not eval_config:
        raise ValueError("No evaluations defined in config file")

    for idx, eval_cfg in enumerate(eval_config):
        if "name" not in eval_cfg:
            raise ValueError(f"Evaluation config at index {idx} is missing required key: 'name'")

        task_name = eval_cfg["name"]
        if task_name not in allowed_tasks:
            warnings.warn(
                f"Evaluation task '{task_name}' at index {idx} is not supported. "
                f"Skipping. Supported tasks: {', '.join(sorted(allowed_tasks))}"
            )
