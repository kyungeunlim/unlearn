#!/usr/bin/env python3
"""Evaluate a model on WMDP bio robust benchmark using lm_eval."""

import argparse
import os

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn.utils.utils import assert_type

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    include_path = os.path.join(REPO_ROOT, "unlearn", "lm_eval_tasks")
    tm = TaskManager(verbosity="INFO", include_path=include_path)

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    print("Running evaluation on wmdp_bio_robust...")
    # The type hints for this package don't work for some reason.
    results = simple_evaluate( # type: ignore
        model=lm, # type: ignore
        tasks=["wmdp_bio_robust"], # type: ignore
        task_manager=tm, # type: ignore
    )
    results = assert_type(dict, results)

    print("\n" + "=" * 60)
    print("WMDP Bio Robust Results:")
    print("=" * 60)

    if "results" in results and "wmdp_bio_robust" in results["results"]:
        task_results = results["results"]["wmdp_bio_robust"]
        for metric, value in task_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    return results


if __name__ == "__main__":
    main()
