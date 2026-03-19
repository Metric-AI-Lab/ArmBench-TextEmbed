"""MTEB evaluation module for Armenian language tasks."""

import logging
from typing import Any, Dict
import mteb
from models import Embedding_Model
import utils

# Load defaults from YAML
_defaults = utils.load_defaults()
_mteb_defaults = _defaults.get("tasks", {}).get("mteb", {})
DEFAULT_BATCH_SIZE = _mteb_defaults.get("batch_size", 256)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_mteb(model: Embedding_Model, output_dir: str) -> Dict[str, Any]:
    """Run MTEB evaluation on Armenian tasks."""
    tasks = mteb.get_tasks(languages=["hye"])
    if not tasks:
        logger.warning("No Armenian tasks found in MTEB.")
        return {}

    logger.info(f"Found {len(tasks)} Armenian MTEB tasks")

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        model.model,
        output_folder=output_dir,
        verbosity=2,
        encode_kwargs={
            "batch_size": DEFAULT_BATCH_SIZE,
            "show_progress_bar": True
        }
    )

    metrics = {}
    for result in results:
        task_name = result.task_name
        for split, split_results in result.scores.items():
            if split_results:
                main_score = split_results[0].get("main_score", 0)
                metrics[f"{task_name}_{split}"] = main_score

    return metrics
