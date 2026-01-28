#!/usr/bin/env python
"""
Train depth-scaled transformer probes on forget data.
Probe depth at layer L = (total_layers - L) transformer layers.
"""

import argparse
import os
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn.probe import (
    TransformerProbe,
    TransformerProbeConfig,
    save_transformer_probe,
)
from unlearn.reference.cas.utils import (
    BIO_REMOVE_DS_NAME,
    cb_tokenize_function,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train depth-scaled transformer probes on forget data"
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
        help="Layers to train probes for. Default: every 4th layer from layer 8",
    )
    parser.add_argument("--probe_dim", type=int, default=64)
    parser.add_argument("--probe_heads", type=int, default=4)
    parser.add_argument("--ffn_multiplier", type=int, default=4)
    parser.add_argument("--min_probe_layers", type=int, default=1, help="Minimum probe depth")

    parser.add_argument("--num_train_examples", type=int, default=5000)
    parser.add_argument("--num_eval_examples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--save_dir", type=str, default="./models/depth_scaled_probes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="depth-scaled-probes")
    parser.add_argument("--wandb_entity", type=str, default=None)

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


def train_probe(
    probe,
    source_model,
    target_model,
    layer_idx,
    train_loader,
    eval_loader,
    epochs,
    lr,
    weight_decay,
    device,
):
    """Train a single probe and return metrics."""
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    source_model.eval()
    target_model.eval()

    total_train_steps = len(train_loader) * epochs
    step = 0

    for epoch in range(epochs):
        probe.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                source_out = source_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                target_out = target_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                source_acts = source_out.hidden_states[layer_idx].float()
                target_acts = target_out.hidden_states[layer_idx].float()

            pred_acts = probe(source_acts, attention_mask.bool())

            mask = attention_mask.bool().unsqueeze(-1)
            pred_masked = pred_acts * mask
            target_masked = target_acts * mask

            loss = loss_fn(pred_masked, target_masked)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            step += 1

            pbar.set_postfix({"loss": loss.item(), "step": f"{step}/{total_train_steps}"})

            wandb.log({
                f"layer_{layer_idx}/train_loss": loss.item(),
                f"layer_{layer_idx}/step": step,
            })

        avg_train_loss = epoch_loss / num_batches

        # Eval
        probe.eval()
        eval_mse = 0.0
        eval_tokens = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Eval", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                source_out = source_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                target_out = target_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                source_acts = source_out.hidden_states[layer_idx].float()
                target_acts = target_out.hidden_states[layer_idx].float()

                pred_acts = probe(source_acts, attention_mask.bool())

                mask = attention_mask.bool()
                pred_flat = pred_acts[mask]
                target_flat = target_acts[mask]

                mse = ((pred_flat - target_flat) ** 2).sum().item()
                eval_mse += mse
                eval_tokens += pred_flat.numel()

        avg_eval_mse = eval_mse / eval_tokens

        print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, eval_mse={avg_eval_mse:.6f}")

        wandb.log({
            f"layer_{layer_idx}/epoch": epoch + 1,
            f"layer_{layer_idx}/avg_train_loss": avg_train_loss,
            f"layer_{layer_idx}/eval_mse": avg_eval_mse,
        })

    return avg_train_loss, avg_eval_mse


def main():
    args = parse_args()

    # Load env file for wandb
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    print(f"Model: {num_model_layers} layers, hidden_dim={model_dim}")

    layers = args.layers
    if layers is None:
        # Default: every 4th layer starting from layer 8
        layers = list(range(8, num_model_layers - 1, 4))
    print(f"Training probes for layers: {layers}")

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "source_model": args.source_model,
            "source_revision": args.source_revision,
            "target_model": args.target_model,
            "target_revision": args.target_revision,
            "layers": layers,
            "probe_dim": args.probe_dim,
            "probe_heads": args.probe_heads,
            "ffn_multiplier": args.ffn_multiplier,
            "min_probe_layers": args.min_probe_layers,
            "num_train_examples": args.num_train_examples,
            "num_eval_examples": args.num_eval_examples,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "num_model_layers": num_model_layers,
            "model_dim": model_dim,
        },
    )

    print(f"WandB run: {wandb.run.get_url()}")

    print("Loading forget dataset (WMDP-Bio-Remove)...")
    dataset = load_dataset(BIO_REMOVE_DS_NAME)["train"]
    dataset = dataset.shuffle(seed=args.seed)

    total_needed = args.num_train_examples + args.num_eval_examples
    dataset = dataset.select(range(min(total_needed, len(dataset))))

    print("Tokenizing...")
    dataset = dataset.map(
        lambda x: cb_tokenize_function(x, tokenizer, max_length=args.max_length),
        batched=True,
        num_proc=4,
        remove_columns=dataset.column_names,
    )

    train_dataset = dataset.select(range(min(args.num_train_examples, len(dataset))))
    eval_start = min(args.num_train_examples, len(dataset))
    eval_dataset = dataset.select(range(eval_start, min(total_needed, len(dataset))))

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

    print(f"\nTrain batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in layers:
        # Compute probe depth: (total_layers - current_layer)
        probe_layers = max(num_model_layers - layer_idx, args.min_probe_layers)

        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}: probe_depth={probe_layers} transformer layers")
        print("=" * 60)

        probe_config = TransformerProbeConfig(
            model_dim=model_dim,
            probe_dim=args.probe_dim,
            num_layers=probe_layers,
            num_heads=args.probe_heads,
            ffn_multiplier=args.ffn_multiplier,
        )

        probe = TransformerProbe(probe_config).to(device)
        param_count = sum(p.numel() for p in probe.parameters())
        print(f"Probe parameters: {param_count:,}")

        wandb.log({
            f"layer_{layer_idx}/probe_depth": probe_layers,
            f"layer_{layer_idx}/probe_params": param_count,
        })

        train_mse, eval_mse = train_probe(
            probe=probe,
            source_model=source_model,
            target_model=target_model,
            layer_idx=layer_idx,
            train_loader=train_loader,
            eval_loader=eval_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
        )

        print(f"Saving probe for layer {layer_idx}...")
        save_transformer_probe(
            probe=probe,
            save_dir=save_dir,
            layer_idx=layer_idx,
            train_mse=train_mse,
            eval_mse=eval_mse,
        )

        wandb.log({
            f"layer_{layer_idx}/final_train_mse": train_mse,
            f"layer_{layer_idx}/final_eval_mse": eval_mse,
        })

    wandb.finish()
    print(f"\nAll probes saved to {save_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
