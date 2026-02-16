#!/usr/bin/env python
"""
Train transformer probes to map activations between model checkpoints.
"""

import argparse

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn.probe import (
    TransformerProbeConfig,
    TransformerProbeTrainer,
    save_transformer_probe,
)
from unlearn.reference.cas.utils import (
    RETAIN_TEXT_DS_NAME,
    wikitext_tokenize_function,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train transformer probes to map activations " "between model checkpoints"
        )
    )
    parser.add_argument(
        "--source_model",
        type=str,
        default="EleutherAI/deep-ignorance-pretraining-stage-unfiltered",
    )
    parser.add_argument("--source_revision", type=str, default="global_step38144")
    parser.add_argument(
        "--target_model",
        type=str,
        default="EleutherAI/deep-ignorance-unfiltered",
    )
    parser.add_argument("--target_revision", type=str, default="main")

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to train probes for (default: 1/4, 1/2, 3/4 of model depth)",
    )
    parser.add_argument("--probe_dim", type=int, default=64)
    parser.add_argument("--probe_layers", type=int, default=2)
    parser.add_argument("--probe_heads", type=int, default=4)
    parser.add_argument("--ffn_multiplier", type=int, default=4)

    parser.add_argument("--num_train_examples", type=int, default=10000)
    parser.add_argument("--num_eval_examples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--save_dir", type=str, default="./models/transformer_probes")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def collate_fn(batch, max_length):
    """Collate batch with padding."""
    input_ids = []
    attention_masks = []

    for item in batch:
        ids = item["input_ids"]
        if isinstance(ids, list):
            ids = torch.tensor(ids, dtype=torch.long)
        if ids.ndim > 1:
            ids = ids.view(-1)
        ids = ids[:max_length]

        mask = item["attention_mask"]
        if isinstance(mask, list):
            mask = torch.tensor(mask, dtype=torch.long)
        if mask.ndim > 1:
            mask = mask.view(-1)
        mask = mask[:max_length]

        input_ids.append(ids)
        attention_masks.append(mask)

    max_len = max(len(ids) for ids in input_ids)

    padded_ids = []
    padded_masks = []
    for ids, mask in zip(input_ids, attention_masks):
        pad_len = max_len - len(ids)
        padded_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)]))
        padded_masks.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(padded_masks),
    }


def main():
    args = parse_args()

    print("=" * 60)
    print("Transformer Probe Training")
    print("=" * 60)
    print(f"Source: {args.source_model} @ {args.source_revision}")
    print(f"Target: {args.target_model} @ {args.target_revision}")
    print(
        f"Probe: dim={args.probe_dim}, layers={args.probe_layers}"
        f", heads={args.probe_heads}"
    )
    print(f"Training: {args.num_train_examples} examples, {args.epochs} epochs")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    torch.manual_seed(args.seed)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading source model: {args.source_model}...")
    source_model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        revision=args.source_revision,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    source_model.eval()

    print(f"Loading target model: {args.target_model}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        revision=args.target_revision,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    target_model.eval()

    model_dim = target_model.config.hidden_size
    num_model_layers = target_model.config.num_hidden_layers

    layers = args.layers
    if layers is None:
        layers = [
            num_model_layers // 4,
            num_model_layers // 2,
            3 * num_model_layers // 4,
        ]
    print(f"Training probes for layers: {layers}")

    print("Loading dataset...")
    dataset = load_dataset(RETAIN_TEXT_DS_NAME, "bio-retain-corpus")["train"]
    dataset = dataset.shuffle(seed=args.seed)

    total_needed = args.num_train_examples + args.num_eval_examples
    dataset = dataset.select(range(min(total_needed, len(dataset))))

    print("Tokenizing...")
    dataset = dataset.map(
        lambda x: wikitext_tokenize_function(x, tokenizer, max_length=args.max_length),
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
    )

    train_dataset = dataset.select(range(args.num_train_examples))
    eval_dataset = dataset.select(
        range(args.num_train_examples, min(total_needed, len(dataset)))
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, args.max_length),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, args.max_length),
    )

    probe_config = TransformerProbeConfig(
        model_dim=model_dim,
        probe_dim=args.probe_dim,
        num_layers=args.probe_layers,
        num_heads=args.probe_heads,
        ffn_multiplier=args.ffn_multiplier,
    )

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Training probe for layer {layer_idx}")
        print("=" * 60)

        trainer = TransformerProbeTrainer(
            source_model=source_model,
            target_model=target_model,
            layer_idx=layer_idx,
            probe_config=probe_config,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )

        param_count = sum(p.numel() for p in trainer.probe.parameters())
        print(f"Probe parameters: {param_count:,}")

        best_eval_mse = float("inf")
        final_train_mse = None

        for epoch in range(args.epochs):
            train_mse = trainer.train_epoch(train_loader)
            eval_mse = trainer.evaluate(eval_loader)

            print(
                f"Epoch {epoch + 1}/{args.epochs}: "
                f"train_mse={train_mse:.6f}, eval_mse={eval_mse:.6f}"
            )

            final_train_mse = train_mse
            if eval_mse < best_eval_mse:
                best_eval_mse = eval_mse

        print(f"Saving probe for layer {layer_idx}...")
        save_transformer_probe(
            probe=trainer.probe,
            save_dir=args.save_dir,
            layer_idx=layer_idx,
            train_mse=final_train_mse,
            eval_mse=best_eval_mse,
        )

    print(f"\nAll probes saved to {args.save_dir}")
    print("Training complete")


if __name__ == "__main__":
    main()
