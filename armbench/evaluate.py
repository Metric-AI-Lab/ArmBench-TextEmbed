"""Main evaluation script for embedding models."""

import argparse
import logging
import os
import time
from typing import Any, Dict, List
import yaml
from utils import (
    ensure_output_dir,
    save_evaluation_to_csv,
    load_config,
    validate_model_config,
    validate_evaluation_config,
)

from models import Embedding_Model
from sts_eval import evaluate_sts
from retrieval_evaluation import evaluate_retrieval
from ms_marco_eval import evaluate_ms_marco
from mteb_eval import evaluate_mteb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = "configs/config.yml"

EVAL_SUITE = {
    "sts": {"fn": evaluate_sts},
    "ms_marco": {"fn": evaluate_ms_marco},
    "retrieval": {"fn": evaluate_retrieval},
    "mteb": {"fn": evaluate_mteb},
}

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate embedding models")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML file",
    )
    return parser.parse_args()


def run_all_metrics(
    model: Embedding_Model,
    eval_configs: List[Dict[str, Any]] = None,
    context: Dict[str, Any] = None,
    eval_suite: Dict[str, Any] = None,
    translit: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run all configured evaluation metrics on the model."""
    eval_configs_by_name = {}
    if eval_configs:
        for cfg in eval_configs:
            name = cfg.get("name")
            if name:
                eval_configs_by_name[name] = cfg

    save_embeddings = context.get("save_embeddings", False) if context else False
    output_dir = context.get("output_dir", OUTPUT_DIR) if context else OUTPUT_DIR
    model_name = context.get("model_name", "unknown") if context else "unknown"

    all_results = {}

    for eval_name, eval_info in eval_suite.items():
        if not eval_configs_by_name or eval_name not in eval_configs_by_name:
            continue

        eval_config = eval_configs_by_name[eval_name]
        prompts = eval_config.get("prompts", {})
        instruct_prefix = prompts.get("instruct_prefix", "")
        if instruct_prefix:
            instruction = None
            instructions_path = context.get("instructions_path", "") if context else ""
            if instructions_path and os.path.exists(instructions_path):
                with open(instructions_path, "r") as f:
                    task_instructions = yaml.safe_load(f)
                instruction = task_instructions.get(eval_name, {}).get(
                    "instruction", ""
                )
            if not instruction:
                raise ValueError(
                    f"Instruction prefix provided for '{eval_name}', but no instruction found in '{instructions_path}'."
                )

        start = time.time()
        if eval_name == "mteb":
            mteb_output_dir = os.path.join(OUTPUT_DIR, context["model_name"], "mteb")
            metrics = eval_info["fn"](model, mteb_output_dir)
        else:
            emb_path = None
            if save_embeddings:
                emb_dir = os.path.join(output_dir, model_name, "embeddings")
                os.makedirs(emb_dir, exist_ok=True)
                emb_path = os.path.join(emb_dir, f"{eval_name}_embeddings")

            metrics = eval_info["fn"](
                model, eval_config=eval_config,
                save_embeddings_path=emb_path, translit=False
            )
        seconds = round(time.time() - start, 2)
        logging.info("eval=%s seconds=%.2f metrics=%s", eval_name, seconds, metrics)
        save_evaluation_to_csv(eval_name, context["model_name"], metrics, OUTPUT_DIR, context.get("model_path"))
        all_results[eval_name] = metrics

        if translit and eval_name in {"retrieval", "ms_marco"}:
            start = time.time()
            emb_path_translit = None
            if save_embeddings:
                emb_dir = os.path.join(output_dir, model_name, "embeddings")
                os.makedirs(emb_dir, exist_ok=True)
                emb_path_translit = os.path.join(emb_dir, f"{eval_name}_translit_embeddings")

            metrics = eval_info["fn"](
                model, eval_config=eval_config,
                save_embeddings_path=emb_path_translit, translit=translit
            )
            seconds = round(time.time() - start, 2)
            logging.info("eval=%s seconds=%.2f metrics=%s", eval_name, seconds, metrics)
            save_evaluation_to_csv(
                f"{eval_name}_translit", context["model_name"], metrics, OUTPUT_DIR, context.get("model_path")
            )
            all_results[f"{eval_name}_translit"] = metrics

    return all_results


def main() -> Dict[str, Dict[str, Any]]:
    """Main entry point for evaluation."""
    args = parse_args()
    config = load_config(args.config)

    general = config.get("general", {})
    if "batch_size" not in general:
        raise ValueError("'general.batch_size' is required in config file")
    if "output_dir" not in general:
        raise ValueError("'general.output_dir' is required in config file")

    batch_size = general["batch_size"]
    output_dir = general["output_dir"]
    translit = general.get("translit", False)
    save_emb = general.get("save_embeddings", False)

    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    ensure_output_dir(output_dir)

    models_config = config.get("models", [])
    validate_model_config(models_config)

    eval_section = config.get("evaluation", {})
    eval_configs = (
        eval_section.get("tasks", [])
        if isinstance(eval_section, dict)
        else eval_section
    )
    instructions_path = (
        eval_section.get("instructions", "") if isinstance(eval_section, dict) else ""
    )
    validate_evaluation_config(eval_configs)

    all_results = {}
    for model_cfg in models_config:
        model_name = model_cfg["model_name"]
        pooling = model_cfg.get("pooling")
        max_length = model_cfg["max_length"]

        logging.info("Loading model: %s (%s)", model_cfg["name"], model_name)

        model = Embedding_Model(
            model_name,
            batch_size=batch_size,
            max_length=max_length,
            pooling=pooling,
        )

        context = {
            "model_name": model_cfg["name"],
            "model_path": model_name,
            "batch_size": batch_size,
            "max_length": max_length,
            "instructions_path": instructions_path,
            "save_embeddings": save_emb,
            "output_dir": output_dir,
        }
        res = run_all_metrics(model, eval_configs, context, EVAL_SUITE, translit)
        all_results[model_cfg["name"]] = res

    return all_results


if __name__ == "__main__":
    main()
