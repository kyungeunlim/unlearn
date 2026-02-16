"""
Two-phase tamper attack:
  Phase 1 (Repair): Finetune on general text (WikiText-103) to restore
    language modeling
  Phase 2 (Attack): Finetune on WMDP-Bio data to recover hazardous knowledge

Hypothesis: models with destroyed general capabilities (low MMLU) appear
tamper-resistant because bio-data finetuning alone can't recover domain
knowledge without basic LM ability.
"""

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

MAX_LENGTH = 2048
REPO_ROOT = Path("/home/a6a/lucia.a6a/unlearn")


def run_lmeval_subprocess(model_path: str, tasks: list[str], num_gpus: int = 4) -> dict:
    tasks_str = ",".join(tasks)
    include_path = str(REPO_ROOT / "unlearn" / "lm_eval_tasks")
    output_dir = Path("/tmp/lm_eval_results")

    # Clean stale results to avoid picking up old JSON files from other runs
    if output_dir.exists():
        shutil.rmtree(output_dir)

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path},parallelize=True",
        "--tasks",
        tasks_str,
        "--batch_size",
        "auto",
        "--verbosity",
        "WARNING",
        "--output_path",
        str(output_dir),
    ]
    if "wmdp" in tasks_str:
        cmd.extend(["--include_path", include_path])
    if "mmlu" in tasks_str:
        cmd.extend(["--num_fewshot", "1"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    print(f"Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"lm_eval stderr: {result.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"lm_eval failed with code {result.returncode}")

    results = {}

    # Parse from JSON output files
    if output_dir.exists():
        for json_file in sorted(output_dir.rglob("results.json"), reverse=True):
            with open(json_file) as f:
                data = json.load(f)
            if "results" in data:
                if "wmdp_bio_robust" in data["results"]:
                    results["wmdp_bio_robust"] = data["results"]["wmdp_bio_robust"][
                        "acc,none"
                    ]
                if "mmlu" in data["results"]:
                    results["mmlu"] = data["results"]["mmlu"]["acc,none"]
            break

    # Fallback: parse table from stdout/stderr
    if not results:
        output = result.stdout + result.stderr
        for line in output.split("\n"):
            if "|wmdp_bio_robust " in line and "|acc" in line:
                match = re.search(r"\|acc\s*\|[^\|]*\|\s*([\d.]+)", line)
                if match:
                    results["wmdp_bio_robust"] = float(match.group(1))
            elif "|mmlu " in line and "|acc" in line:
                match = re.search(r"\|acc\s*\|[^\|]*\|\s*([\d.]+)", line)
                if match:
                    results["mmlu"] = float(match.group(1))

    if not results:
        print(f"EVAL PARSING FAILED. stdout tail: {result.stdout[-1000:]}", flush=True)
        print(f"stderr tail: {result.stderr[-500:]}", flush=True)

    return results


class EvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_steps, checkpoint_dir, num_gpus, phase):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_steps = set(eval_steps)
        self.checkpoint_dir = checkpoint_dir
        self.num_gpus = num_gpus
        self.phase = phase
        self.results = []

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step in self.eval_steps:
            self._eval(state.global_step)

    def _eval(self, step):
        self.model.eval()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.checkpoint_dir)
        self.tokenizer.save_pretrained(self.checkpoint_dir)

        device = next(self.model.parameters()).device
        self.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        lm_results = run_lmeval_subprocess(
            str(self.checkpoint_dir), ["wmdp_bio_robust", "mmlu"], self.num_gpus
        )

        result = {
            "phase": self.phase,
            "step": step,
            "wmdp_bio_acc": lm_results.get("wmdp_bio_robust", 0.25),
            "mmlu_acc": lm_results.get("mmlu", 0.25),
            "timestamp": datetime.now().isoformat(),
        }
        self.results.append(result)
        print(
            f"  [{self.phase}] Step {step}: WMDP = {result['wmdp_bio_acc']*100:.2f}%, "
            f"MMLU = {result['mmlu_acc']*100:.2f}%",
            flush=True,
        )

        self.model.to(device)
        self.model.train()


def prepare_wikitext(tokenizer, num_examples=512, max_chunks=3):
    ds = load_dataset("EleutherAI/wikitext_document_level", "wikitext-103-raw-v1")
    texts = [t for t in ds["train"]["page"] if len(t) > 200][:num_examples]

    chunks = []
    pad_token_id = tokenizer.pad_token_id
    for text in texts:
        tokens = tokenizer(text, truncation=False)
        input_ids = tokens["input_ids"]
        for i in range(0, len(input_ids), MAX_LENGTH):
            chunk_ids = input_ids[i : i + MAX_LENGTH]
            pad_len = MAX_LENGTH - len(chunk_ids)
            if pad_len > 0:
                chunk_ids = chunk_ids + [pad_token_id] * pad_len
                mask = [1] * (MAX_LENGTH - pad_len) + [0] * pad_len
                labels = input_ids[i : i + MAX_LENGTH] + [-100] * pad_len
            else:
                mask = [1] * MAX_LENGTH
                labels = chunk_ids.copy()
            chunks.append(
                {
                    "input_ids": chunk_ids,
                    "attention_mask": mask,
                    "labels": labels,
                }
            )
            if len(chunks) >= num_examples * max_chunks:
                break
        if len(chunks) >= num_examples * max_chunks:
            break

    return hf_dataset.from_list(chunks).shuffle(seed=42)


def prepare_bio(tokenizer, num_examples=512, max_chunks=5):
    ds = load_dataset("Unlearning/WMDP-Bio-Remove-Dataset")
    training_data = ds["train"].shuffle(seed=42).select(range(num_examples))

    chunks = []
    pad_token_id = tokenizer.pad_token_id
    for example in training_data:
        text = (
            example["title"] + "\n\n" + example["abstract"] + "\n\n" + example["text"]
        )
        tokens = tokenizer(text, truncation=False)
        input_ids = tokens["input_ids"]
        chunk_count = 0
        for i in range(0, len(input_ids), MAX_LENGTH):
            chunk_ids = input_ids[i : i + MAX_LENGTH]
            pad_len = MAX_LENGTH - len(chunk_ids)
            if pad_len > 0:
                chunk_ids = chunk_ids + [pad_token_id] * pad_len
                mask = [1] * (MAX_LENGTH - pad_len) + [0] * pad_len
                labels = input_ids[i : i + MAX_LENGTH] + [-100] * pad_len
            else:
                mask = [1] * MAX_LENGTH
                labels = chunk_ids.copy()
            chunks.append(
                {
                    "input_ids": chunk_ids,
                    "attention_mask": mask,
                    "labels": labels,
                }
            )
            chunk_count += 1
            if chunk_count >= max_chunks:
                break

    return hf_dataset.from_list(chunks).shuffle(seed=42)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--repair_steps", type=int, default=200)
    parser.add_argument("--attack_steps", type=int, default=500)
    parser.add_argument("--repair_lr", type=float, default=2e-5)
    parser.add_argument("--attack_lr", type=float, default=2e-5)
    parser.add_argument("--eval_every", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 4))

    print(f"Loading model: {args.model_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    )
    model.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<|padding|>",
            "eos_token": "<|endoftext|>",
            "bos_token": "<|startoftext|>",
        }
    )
    tokenizer.padding_side = "left"

    for param in model.parameters():
        param.requires_grad = True

    checkpoint_dir = output_dir / "eval_checkpoint"
    all_results = []

    # Phase 1: Repair with WikiText
    print(f"\n{'='*60}", flush=True)
    print(
        (
            f"PHASE 1: Repair with WikiText ({args.repair_steps} "
            f"steps, lr={args.repair_lr})"
        ),
        flush=True,
    )
    print(f"{'='*60}", flush=True)

    wiki_dataset = prepare_wikitext(tokenizer)
    print(f"WikiText dataset: {len(wiki_dataset)} chunks", flush=True)

    repair_eval_steps = list(range(0, args.repair_steps + 1, args.eval_every))
    if args.repair_steps not in repair_eval_steps:
        repair_eval_steps.append(args.repair_steps)

    repair_callback = EvalCallback(
        model, tokenizer, repair_eval_steps, checkpoint_dir, num_gpus, "repair"
    )

    repair_args = TrainingArguments(
        output_dir=str(output_dir / "repair_checkpoints"),
        learning_rate=args.repair_lr,
        gradient_accumulation_steps=16,
        per_device_train_batch_size=1,
        max_steps=args.repair_steps,
        weight_decay=0.01,
        gradient_checkpointing=True,
        save_strategy="no",
        warmup_steps=0,
        logging_strategy="steps",
        logging_steps=10,
        report_to=[],
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=repair_args,
        train_dataset=wiki_dataset,
        callbacks=[repair_callback],
    )
    model.train()
    trainer.train()

    # Final repair eval
    repair_callback._eval(args.repair_steps)
    all_results.extend(repair_callback.results)

    # Save repaired model
    repaired_dir = output_dir / "repaired_model"
    repaired_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(repaired_dir)
    tokenizer.save_pretrained(repaired_dir)

    del trainer
    gc.collect()

    # Phase 2: Attack with Bio data
    print(f"\n{'='*60}", flush=True)
    print(
        (
            f"PHASE 2: Attack with Bio data ({args.attack_steps} "
            f"steps, lr={args.attack_lr})"
        ),
        flush=True,
    )
    print(f"{'='*60}", flush=True)

    bio_dataset = prepare_bio(tokenizer)
    print(f"Bio dataset: {len(bio_dataset)} chunks", flush=True)

    attack_eval_steps = list(range(0, args.attack_steps + 1, args.eval_every))
    if args.attack_steps not in attack_eval_steps:
        attack_eval_steps.append(args.attack_steps)

    attack_callback = EvalCallback(
        model, tokenizer, attack_eval_steps, checkpoint_dir, num_gpus, "attack"
    )

    attack_args = TrainingArguments(
        output_dir=str(output_dir / "attack_checkpoints"),
        learning_rate=args.attack_lr,
        gradient_accumulation_steps=16,
        per_device_train_batch_size=1,
        max_steps=args.attack_steps,
        weight_decay=0.01,
        gradient_checkpointing=True,
        save_strategy="no",
        warmup_steps=0,
        logging_strategy="steps",
        logging_steps=10,
        report_to=[],
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=attack_args,
        train_dataset=bio_dataset,
        callbacks=[attack_callback],
    )
    model.train()
    trainer.train()

    attack_callback._eval(args.attack_steps)
    all_results.extend(attack_callback.results)

    # Summary
    print(f"\n{'='*60}")
    print("REPAIR-THEN-ATTACK RESULTS")
    print(f"{'='*60}")
    print(f"{'Phase':<10} {'Step':>6} {'WMDP':>10} {'MMLU':>10}")
    print("-" * 40)
    for r in all_results:
        print(
            f"{r['phase']:<10} {r['step']:>6} {r['wmdp_bio_acc']*100:>9.2f}% "
            f"{r['mmlu_acc']*100:>9.2f}%"
        )

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
