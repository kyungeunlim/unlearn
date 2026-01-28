#!/usr/bin/env python3
"""
Run finetune (tamper) attack on a lens-unlearned model with frequent evaluation,
save results to disk, and plot WMDP accuracy over training steps.
"""

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from datasets import Dataset as hf_dataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import bitsandbytes as bnb

sys.path.append("./lm-evaluation-harness")

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MAX_LENGTH = 2048


@dataclass
class TamperAttackConfig:
    model_name: str = "models/EleutherAI/deep-ignorance-unfiltered_lens_ex8192_rm5.0_ret5.0"
    output_dir: str = "runs/tamper_attack"
    num_train_examples: int = 512
    epochs: int = 1
    eval_every: int = 10
    max_chunks: int = 5
    lr: float = 2e-5
    batch_size: int = 1
    grad_accumulation: int = 16


class WMDPEvalCallback(TrainerCallback):
    """Callback to evaluate WMDP accuracy during training and save results."""

    def __init__(self, model, eval_every: int, output_path: Path):
        self.model = model
        self.eval_every = eval_every
        self.output_path = output_path
        self.eval_results = []

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every == 0:
            wmdp_acc = self._evaluate_wmdp()
            result = {
                "step": state.global_step,
                "wmdp_bio_acc": wmdp_acc,
                "timestamp": datetime.now().isoformat(),
            }
            self.eval_results.append(result)
            print(f"Step {state.global_step}: WMDP Bio Acc = {wmdp_acc:.4f}")
            self._save_results()

    def _evaluate_wmdp(self) -> float:
        self.model.eval()
        with torch.no_grad():
            hflm_model = HFLM(self.model)
            eval_results = evaluator.simple_evaluate(
                model=hflm_model,
                tasks=["wmdp_bio_robust"],
                device=self.model.device,
                verbosity="ERROR",
                num_fewshot=0,
            )
            acc = eval_results["results"]["wmdp_bio_robust"]["acc,none"]
            del hflm_model
            del eval_results
            gc.collect()
        self.model.train()
        return acc

    def _save_results(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.eval_results, f, indent=2)
        print(f"Results saved to {self.output_path}")


def get_model_and_tokenizer(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({
        "pad_token": "<|padding|>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|startoftext|>",
    })
    tokenizer.padding_side = "left"
    return model, tokenizer


def tokenize_examples_fn(examples, tokenizer):
    cb_examples = [
        examples["title"][i] + "\n\n" + examples["abstract"][i] + "\n\n" + examples["text"][i]
        for i in range(len(examples["title"]))
    ]
    tokenized_output = tokenizer(cb_examples, padding="max_length", truncation=False)
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output


def chunk_example(example, tokenizer, chunk_size=MAX_LENGTH, max_chunks=5):
    pad_token_id = tokenizer.pad_token_id
    input_ids = example["input_ids"]
    attention_mask = example.get("attention_mask", [1] * len(input_ids))
    labels = example.get("labels", input_ids.copy())
    token_type_ids = example.get("token_type_ids", [0] * len(input_ids))
    chunks = []
    for i in range(0, len(input_ids), chunk_size):
        chunk_input_ids = input_ids[i : i + chunk_size]
        chunk_attention_mask = attention_mask[i : i + chunk_size]
        chunk_labels = labels[i : i + chunk_size]
        chunk_token_type_ids = token_type_ids[i : i + chunk_size]
        pad_len = chunk_size - len(chunk_input_ids)
        if pad_len > 0:
            chunk_input_ids += [pad_token_id] * pad_len
            chunk_attention_mask += [0] * pad_len
            chunk_labels += [-100] * pad_len
            chunk_token_type_ids += [0] * pad_len
        chunks.append({
            "input_ids": chunk_input_ids,
            "attention_mask": chunk_attention_mask,
            "labels": chunk_labels,
            "token_type_ids": chunk_token_type_ids,
        })
        if i >= (max_chunks - 1) * chunk_size:
            break
    return chunks


def prepare_dataset(config: TamperAttackConfig, tokenizer):
    wmdp_bio_forget = load_dataset("Unlearning/WMDP-Bio-Remove-Dataset")
    training_data = wmdp_bio_forget["train"].shuffle(seed=42).select(range(config.num_train_examples))
    tokenized_training_data = training_data.map(
        lambda x: tokenize_examples_fn(x, tokenizer), batched=True
    )
    chunked_examples = []
    for i, example in enumerate(tokenized_training_data):
        chunked_examples.extend(
            chunk_example(example, tokenizer, chunk_size=MAX_LENGTH, max_chunks=config.max_chunks)
        )
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1} examples, total chunks: {len(chunked_examples)}")
    return hf_dataset.from_list(chunked_examples).shuffle(seed=42)


def plot_results(results_path: Path, output_path: Optional[Path] = None):
    with open(results_path) as f:
        results = json.load(f)

    steps = [r["step"] for r in results]
    accs = [r["wmdp_bio_acc"] * 100 for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, accs, "b-o", linewidth=2, markersize=6)
    plt.axhline(y=25, color="r", linestyle="--", label="Random chance (25%)")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("WMDP Bio Accuracy (%)", fontsize=12)
    plt.title("Tamper Attack: WMDP Bio Recovery Over Training", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

    if output_path is None:
        output_path = results_path.with_suffix(".png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")
    return output_path


def run_tamper_attack(config: TamperAttackConfig):
    print(f"Running tamper attack on: {config.model_name}")
    print(f"Config: {config}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"tamper_results_{timestamp}.json"

    model, tokenizer = get_model_and_tokenizer(config.model_name)
    dataset = prepare_dataset(config, tokenizer)
    print(f"Training dataset size: {len(dataset)}")

    callback = WMDPEvalCallback(model, config.eval_every, results_path)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=config.lr,
        gradient_accumulation_steps=config.grad_accumulation,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=0.01,
        gradient_checkpointing=True,
        fp16=True,
        save_strategy="no",
        warmup_steps=0,
        logging_strategy="steps",
        logging_steps=10,
        report_to=[],
    )

    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[callback],
        optimizers=(optimizer, None),
    )

    for param in model.parameters():
        param.requires_grad = True

    model.train()
    trainer.train()

    final_acc = callback._evaluate_wmdp()
    callback.eval_results.append({
        "step": trainer.state.global_step,
        "wmdp_bio_acc": final_acc,
        "timestamp": datetime.now().isoformat(),
        "final": True,
    })
    callback._save_results()

    print("\n" + "=" * 50)
    print("WMDP Bio Accuracy Over Training:")
    print("=" * 50)
    for r in callback.eval_results:
        print(f"Step {r['step']:4d}: {r['wmdp_bio_acc']*100:.2f}%")
    print("=" * 50)

    plot_path = plot_results(results_path)

    return results_path, plot_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run tamper attack with evaluation and plotting")
    parser.add_argument(
        "--model_name",
        type=str,
        default="models/EleutherAI/deep-ignorance-unfiltered_lens_ex8192_rm5.0_ret5.0",
        help="Path to the lens-unlearned model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/tamper_attack",
        help="Directory to save results and plots",
    )
    parser.add_argument(
        "--num_train_examples",
        type=int,
        default=512,
        help="Number of training examples",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=10,
        help="Evaluate WMDP every N steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--plot_only",
        type=str,
        default=None,
        help="Path to existing results JSON to plot (skip training)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.plot_only:
        plot_results(Path(args.plot_only))
    else:
        assert torch.cuda.is_available(), "CUDA is not available"

        config = TamperAttackConfig(
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_train_examples=args.num_train_examples,
            epochs=args.epochs,
            eval_every=args.eval_every,
            lr=args.lr,
        )

        results_path, plot_path = run_tamper_attack(config)
        print(f"\nResults saved to: {results_path}")
        print(f"Plot saved to: {plot_path}")
