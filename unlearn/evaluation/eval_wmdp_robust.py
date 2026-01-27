#!/usr/bin/env python3
"""Evaluate a model on WMDP bio robust benchmark using lm_eval."""

import argparse

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn.utils.utils import assert_type


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
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Path to lm_eval_tasks",
    )
    args = parser.parse_args()

    tm = TaskManager(verbosity="INFO", include_path=args.include_path)

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

    # The type hints for this package don't work for some reason.
    results = simple_evaluate(  # type: ignore
        model=lm,  # type: ignore
        tasks=["wmdp_bio_robust"],  # type: ignore
        task_manager=tm,  # type: ignore
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
