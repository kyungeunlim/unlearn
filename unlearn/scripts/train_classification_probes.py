#!/usr/bin/env python
"""
Train depth-scaled classification transformer probes to distinguish
forget vs. retain tokens using hidden state activations.

Uses RMU training data (bio-forget-corpus vs bio-retain-corpus) so both
classes are bio-domain text, making the classification non-trivial.

Probe at every `layer_stride` layers. Probe depth at layer L = (total_layers - L).
Uses ActivationCapture hooks rather than output_hidden_states.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from unlearn.probe import TransformerProbeConfig
from unlearn.utils.hook import ActivationCapture, resolve_layer_names


@dataclass
class ProbeTrainConfig:
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    revision: str = "main"

    probe_type: str = "transformer"  # "transformer" or "linear"
    probe_dim: int = 64
    probe_heads: int = 4
    ffn_multiplier: int = 4

    num_examples: int = 2048
    eval_fraction: float = 0.1
    batch_size: int = 4
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_length: int = 512
    layer_stride: int = 4

    save_dir: str = ""
    seed: int = 42
    wandb_project: str = "classification-probes"
    wandb_entity: str | None = None


class PairedForgetRetainDataset(Dataset):
    """Pairs tokenized forget and retain examples for classification probing."""

    def __init__(self, forget_dataset, retain_dataset):
        self.forget = forget_dataset
        self.retain = retain_dataset

    def __len__(self):
        return min(len(self.forget), len(self.retain))

    def __getitem__(self, idx):
        return {
            "forget_input_ids": self.forget[idx]["input_ids"],
            "forget_attention_mask": self.forget[idx]["attention_mask"],
            "retain_input_ids": self.retain[idx]["input_ids"],
            "retain_attention_mask": self.retain[idx]["attention_mask"],
        }


class ClassificationTransformerProbe(nn.Module):
    """Per-token binary classifier using a depth-scaled transformer."""

    def __init__(self, config: TransformerProbeConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.model_dim, config.probe_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.probe_dim,
            nhead=config.num_heads,
            dim_feedforward=config.probe_dim * config.ffn_multiplier,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        self.classifier = nn.Linear(config.probe_dim, 1)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: [batch, seq, model_dim]
            attention_mask: [batch, seq] boolean (True = valid)
        Returns:
            [batch, seq] logits
        """
        h = self.input_proj(x)

        if attention_mask is not None:
            padding_mask = ~attention_mask
        else:
            padding_mask = None

        h = self.transformer(h, src_key_padding_mask=padding_mask)
        return self.classifier(h).squeeze(-1)


class LinearClassificationProbe(nn.Module):
    """Per-token binary classifier using a single linear layer."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        self.linear = nn.Linear(model_dim, 1)

    def forward(self, x, attention_mask=None):
        return self.linear(x).squeeze(-1)


def collate_fn(batch, max_length):
    forget_ids, forget_masks = [], []
    retain_ids, retain_masks = [], []

    for item in batch:
        for src_key, dst_list, mask_key, dst_mask_list in [
            ("forget_input_ids", forget_ids, "forget_attention_mask", forget_masks),
            ("retain_input_ids", retain_ids, "retain_attention_mask", retain_masks),
        ]:
            ids = item[src_key]
            if isinstance(ids, list):
                ids = torch.tensor(ids, dtype=torch.long)
            if ids.ndim > 1:
                ids = ids.view(-1)
            ids = ids[:max_length]

            mask = item[mask_key]
            if isinstance(mask, list):
                mask = torch.tensor(mask, dtype=torch.long)
            if mask.ndim > 1:
                mask = mask.view(-1)
            mask = mask[:max_length]

            dst_list.append(ids)
            dst_mask_list.append(mask)

    all_seqs = forget_ids + retain_ids
    max_len = max(len(s) for s in all_seqs)

    def pad_and_stack(seqs):
        return torch.stack(
            [torch.cat([s, torch.zeros(max_len - len(s), dtype=s.dtype)]) for s in seqs]
        )

    return {
        "forget_input_ids": pad_and_stack(forget_ids),
        "forget_attention_mask": pad_and_stack(forget_masks),
        "retain_input_ids": pad_and_stack(retain_ids),
        "retain_attention_mask": pad_and_stack(retain_masks),
    }


def compute_metrics(all_preds, all_labels):
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run_forward_and_capture(model, capture, module_name, input_ids, attention_mask):
    """Run a forward pass and return captured activations as float32."""
    capture.clear()
    model(input_ids=input_ids, attention_mask=attention_mask)
    return capture.activations[module_name].float().clone()


def train_and_evaluate_probe(
    model,
    probe,
    layer_idx,
    train_loader,
    eval_loader,
    probe_train_cfg,
    device,
):
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=probe_train_cfg.lr,
        weight_decay=probe_train_cfg.weight_decay,
    )

    layer_names = resolve_layer_names(model, [layer_idx])
    module_name = layer_names[layer_idx]
    capture = ActivationCapture(model, [module_name])
    capture.register()

    model.eval()
    total_train_steps = len(train_loader) * probe_train_cfg.epochs
    step = 0
    avg_loss = 0.0

    for epoch in range(probe_train_cfg.epochs):
        probe.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Layer {layer_idx} epoch {epoch + 1}/{probe_train_cfg.epochs}",
            leave=False,
        )
        for batch in pbar:
            forget_ids = batch["forget_input_ids"].to(device)
            forget_mask = batch["forget_attention_mask"].to(device)
            retain_ids = batch["retain_input_ids"].to(device)
            retain_mask = batch["retain_attention_mask"].to(device)

            with torch.no_grad():
                forget_acts = run_forward_and_capture(
                    model, capture, module_name, forget_ids, forget_mask
                ).to(device)
                retain_acts = run_forward_and_capture(
                    model, capture, module_name, retain_ids, retain_mask
                ).to(device)

            all_acts = torch.cat([forget_acts, retain_acts], dim=0)
            all_masks = torch.cat([forget_mask, retain_mask], dim=0).bool()

            seq_len = all_acts.shape[1]
            labels = torch.cat(
                [
                    torch.ones(forget_acts.shape[0], seq_len, device=device),
                    torch.zeros(retain_acts.shape[0], seq_len, device=device),
                ]
            )

            logits = probe(all_acts, all_masks)
            loss = F.binary_cross_entropy_with_logits(
                logits[all_masks], labels[all_masks]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            step += 1

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "step": f"{step}/{total_train_steps}"}
            )
            wandb.log(
                {
                    f"layer_{layer_idx}/train_loss": loss.item(),
                    f"layer_{layer_idx}/step": step,
                }
            )

        avg_loss = epoch_loss / num_batches
        wandb.log(
            {
                f"layer_{layer_idx}/epoch": epoch + 1,
                f"layer_{layer_idx}/avg_train_loss": avg_loss,
            }
        )
        print(f"  Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

    # Evaluation
    probe.eval()
    all_preds = []
    all_labels = []
    eval_loss_sum = 0.0
    eval_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Layer {layer_idx} eval", leave=False):
            forget_ids = batch["forget_input_ids"].to(device)
            forget_mask = batch["forget_attention_mask"].to(device)
            retain_ids = batch["retain_input_ids"].to(device)
            retain_mask = batch["retain_attention_mask"].to(device)

            forget_acts = run_forward_and_capture(
                model, capture, module_name, forget_ids, forget_mask
            ).to(device)
            retain_acts = run_forward_and_capture(
                model, capture, module_name, retain_ids, retain_mask
            ).to(device)

            all_acts = torch.cat([forget_acts, retain_acts], dim=0)
            all_masks = torch.cat([forget_mask, retain_mask], dim=0).bool()

            seq_len = all_acts.shape[1]
            labels = torch.cat(
                [
                    torch.ones(forget_acts.shape[0], seq_len, device=device),
                    torch.zeros(retain_acts.shape[0], seq_len, device=device),
                ]
            )

            logits = probe(all_acts, all_masks)
            loss = F.binary_cross_entropy_with_logits(
                logits[all_masks], labels[all_masks]
            )
            eval_loss_sum += loss.item()
            eval_batches += 1

            preds = (logits[all_masks] > 0).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels[all_masks].long().cpu().tolist())

    capture.remove()

    avg_eval_loss = eval_loss_sum / eval_batches if eval_batches > 0 else 0.0
    metrics = compute_metrics(all_preds, all_labels)
    metrics["eval_loss"] = avg_eval_loss
    metrics["final_train_loss"] = avg_loss

    return metrics


def save_classification_probe(
    probe, save_dir, layer_idx, metrics, probe_type="transformer"
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tensors = {k: v.cpu() for k, v in probe.state_dict().items()}
    save_file(tensors, save_dir / f"layer_{layer_idx}.safetensors")

    metadata = {
        "layer_idx": layer_idx,
        "probe_type": probe_type,
        "task": "forget_vs_retain_classification",
        **metrics,
    }
    if probe_type == "transformer":
        metadata.update(
            {
                "model_dim": probe.config.model_dim,
                "probe_dim": probe.config.probe_dim,
                "num_layers": probe.config.num_layers,
                "num_heads": probe.config.num_heads,
                "ffn_multiplier": probe.config.ffn_multiplier,
                "dropout": probe.config.dropout,
            }
        )
    else:
        metadata["model_dim"] = probe.model_dim

    with open(save_dir / f"layer_{layer_idx}_config.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_jsonl_robust(path):
    """Load a JSONL file line-by-line, handling schema inconsistencies."""
    from datasets import Dataset as HFDataset

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, str):
                    obj = {"text": obj}
                records.append(obj)
            except json.JSONDecodeError:
                continue
    return HFDataset.from_list(records)


def load_rmu_paired_dataset(tokenizer, num_examples, max_length, seed):
    """Load bio-forget and bio-retain from RMU training data and pair them."""
    from huggingface_hub import hf_hub_download

    rmu_repo = "Unlearning/rmu-training-data"

    # Try HF hub download, fall back to scanning the cache
    for name in ["bio-forget-corpus.jsonl", "bio-retain-corpus.jsonl"]:
        try:
            hf_hub_download(repo_id=rmu_repo, filename=name, repo_type="dataset")
        except Exception:
            pass  # already cached from login node

    cache_dir = (
        Path.home() / ".cache/huggingface/hub/datasets--Unlearning--rmu-training-data"
    )
    snapshot_dirs = sorted((cache_dir / "snapshots").iterdir())
    snapshot = snapshot_dirs[-1]

    forget_path = snapshot / "bio-forget-corpus.jsonl"
    retain_path = snapshot / "bio-retain-corpus.jsonl"
    print(f"Loading forget from {forget_path}")
    print(f"Loading retain from {retain_path}")

    forget_raw = load_jsonl_robust(forget_path)
    retain_raw = load_jsonl_robust(retain_path)

    print(f"Raw sizes: forget={len(forget_raw)}, retain={len(retain_raw)}")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    print("Tokenizing forget corpus...")
    forget_tok = forget_raw.map(tokenize_fn, batched=True, num_proc=4)
    print("Tokenizing retain corpus...")
    retain_tok = retain_raw.map(tokenize_fn, batched=True, num_proc=4)

    # Shuffle and truncate to num_examples
    forget_tok = forget_tok.shuffle(seed=seed).select(
        range(min(num_examples, len(forget_tok)))
    )
    retain_tok = retain_tok.shuffle(seed=seed).select(
        range(min(num_examples, len(retain_tok)))
    )

    n_paired = min(len(forget_tok), len(retain_tok))
    print(f"Paired dataset size: {n_paired}")

    return PairedForgetRetainDataset(forget_tok, retain_tok)


def main():
    parser = ArgumentParser()
    parser.add_arguments(ProbeTrainConfig, dest="probe_train_cfg")
    probe_train_cfg = parser.parse_args().probe_train_cfg

    assert probe_train_cfg.save_dir, "--save_dir is required"

    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(probe_train_cfg.seed)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(probe_train_cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize dataset before loading the model to avoid CUDA fork issues
    dataset = load_rmu_paired_dataset(
        tokenizer,
        probe_train_cfg.num_examples,
        probe_train_cfg.max_length,
        probe_train_cfg.seed,
    )

    print(f"Loading model: {probe_train_cfg.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        probe_train_cfg.model_name,
        revision=probe_train_cfg.revision,
        device_map="auto",
        dtype=torch.float16,
    )
    model.eval()

    model_dim = model.config.hidden_size
    num_model_layers = model.config.num_hidden_layers
    print(f"Model: {num_model_layers} layers, hidden_dim={model_dim}")

    layers = list(range(0, num_model_layers, probe_train_cfg.layer_stride))
    print(f"Probe layers: {layers}")

    n = len(dataset)
    n_eval = max(1, int(n * probe_train_cfg.eval_fraction))
    n_train = n - n_eval

    indices = torch.randperm(
        n, generator=torch.Generator().manual_seed(probe_train_cfg.seed)
    ).tolist()
    train_dataset = Subset(dataset, indices[:n_train])
    eval_dataset = Subset(dataset, indices[n_train:])

    train_loader = DataLoader(
        train_dataset,
        batch_size=probe_train_cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, probe_train_cfg.max_length),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=probe_train_cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, probe_train_cfg.max_length),
    )

    print(f"Train: {n_train} examples ({len(train_loader)} batches)")
    print(f"Eval: {n_eval} examples ({len(eval_loader)} batches)")

    wandb.init(
        project=probe_train_cfg.wandb_project,
        entity=probe_train_cfg.wandb_entity,
        config={
            "model_name": probe_train_cfg.model_name,
            "revision": probe_train_cfg.revision,
            "probe_type": probe_train_cfg.probe_type,
            "dataset": "Unlearning/rmu-training-data (bio-forget vs bio-retain)",
            "layers": layers,
            "probe_dim": probe_train_cfg.probe_dim,
            "probe_heads": probe_train_cfg.probe_heads,
            "ffn_multiplier": probe_train_cfg.ffn_multiplier,
            "num_examples": probe_train_cfg.num_examples,
            "batch_size": probe_train_cfg.batch_size,
            "epochs": probe_train_cfg.epochs,
            "lr": probe_train_cfg.lr,
            "weight_decay": probe_train_cfg.weight_decay,
            "max_length": probe_train_cfg.max_length,
            "layer_stride": probe_train_cfg.layer_stride,
            "num_model_layers": num_model_layers,
            "model_dim": model_dim,
        },
    )
    print(f"WandB run: {wandb.run.get_url()}")

    save_dir = Path(probe_train_cfg.save_dir)

    is_linear = probe_train_cfg.probe_type == "linear"

    for layer_idx in layers:
        probe_layers = num_model_layers - layer_idx

        print(f"\n{'=' * 60}")
        if is_linear:
            print(f"Layer {layer_idx}: linear probe")
        else:
            print(f"Layer {layer_idx}: probe_depth={probe_layers}")
        print("=" * 60)

        if is_linear:
            probe = LinearClassificationProbe(model_dim).to(device)
        else:
            probe_config = TransformerProbeConfig(
                model_dim=model_dim,
                probe_dim=probe_train_cfg.probe_dim,
                num_layers=probe_layers,
                num_heads=probe_train_cfg.probe_heads,
                ffn_multiplier=probe_train_cfg.ffn_multiplier,
            )
            probe = ClassificationTransformerProbe(probe_config).to(device)

        param_count = sum(p.numel() for p in probe.parameters())
        print(f"Probe parameters: {param_count:,}")

        metrics = train_and_evaluate_probe(
            model=model,
            probe=probe,
            layer_idx=layer_idx,
            train_loader=train_loader,
            eval_loader=eval_loader,
            probe_train_cfg=probe_train_cfg,
            device=device,
        )

        print(
            f"Results: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} Acc={metrics['accuracy']:.4f}"
        )

        save_classification_probe(
            probe,
            save_dir,
            layer_idx,
            metrics,
            probe_type=probe_train_cfg.probe_type,
        )

        wandb.log(
            {
                f"layer_{layer_idx}/probe_params": param_count,
                f"layer_{layer_idx}/f1": metrics["f1"],
                f"layer_{layer_idx}/precision": metrics["precision"],
                f"layer_{layer_idx}/recall": metrics["recall"],
                f"layer_{layer_idx}/accuracy": metrics["accuracy"],
                f"layer_{layer_idx}/eval_loss": metrics["eval_loss"],
            }
        )
        if not is_linear:
            wandb.log({f"layer_{layer_idx}/probe_depth": probe_layers})

        del probe

    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    header = f"{'Layer':<8} {'F1':<8} {'Prec':<8} {'Recall':<8} {'Acc':<8}"
    if not is_linear:
        header = f"{'Layer':<8} {'Depth':<8}" + header[8:]
    print(header)
    for layer_idx in layers:
        config_path = save_dir / f"layer_{layer_idx}_config.json"
        with open(config_path) as f:
            m = json.load(f)
        row = (
            f"{layer_idx:<8} {m['f1']:<8.4f} "
            f"{m['precision']:<8.4f} {m['recall']:<8.4f} {m['accuracy']:<8.4f}"
        )
        if not is_linear:
            row = f"{layer_idx:<8} {m['num_layers']:<8}" + row[8:]
        print(row)

    wandb.finish()
    print(f"\nAll probes saved to {save_dir}")


if __name__ == "__main__":
    main()
