"""
Transformer-based probe for mapping activations between model checkpoints.
Uses self-attention to leverage context across sequence positions.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from tqdm import tqdm


@dataclass
class TransformerProbeConfig:
    model_dim: int
    probe_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ffn_multiplier: int = 4
    dropout: float = 0.0

    def __post_init__(self):
        if self.probe_dim % self.num_heads != 0:
            raise ValueError(
                f"probe_dim ({self.probe_dim}) must be divisible "
                f"by num_heads ({self.num_heads})"
            )


class TransformerProbe(nn.Module):
    """
    Narrow transformer that maps activations from source to target model.

    Architecture:
        [batch, seq, model_dim]
        -> project to [batch, seq, probe_dim]
        -> N transformer encoder layers
        -> project to [batch, seq, model_dim]
    """

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

        self.output_proj = nn.Linear(config.probe_dim, config.model_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """
        Args:
            x: [batch, seq, model_dim] source activations
            attention_mask: [batch, seq] boolean mask (True = valid, False = padding)

        Returns:
            [batch, seq, model_dim] transformed activations
        """
        h = self.input_proj(x)

        if attention_mask is not None:
            # TransformerEncoder expects src_key_padding_mask where True = ignore
            padding_mask = ~attention_mask
        else:
            padding_mask = None

        h = self.transformer(h, src_key_padding_mask=padding_mask)
        return self.output_proj(h)


class TransformerProbeTrainer:
    """Trains TransformerProbe to map source activations to target activations."""

    def __init__(
        self,
        source_model,
        target_model,
        layer_idx: int,
        probe_config: TransformerProbeConfig,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        device: str = "cuda",
    ):
        self.source_model = source_model
        self.target_model = target_model
        self.layer_idx = layer_idx
        self.device = device

        self.probe = TransformerProbe(probe_config).to(device)
        self.optimizer = torch.optim.AdamW(
            self.probe.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.loss_fn = nn.MSELoss()

    def train_epoch(self, dataloader) -> float:
        """Train for one epoch. Returns mean loss."""
        self.probe.train()
        self.source_model.eval()
        self.target_model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                source_out = self.source_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                target_out = self.target_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                source_acts = source_out.hidden_states[self.layer_idx].float()
                target_acts = target_out.hidden_states[self.layer_idx].float()

            pred_acts = self.probe(source_acts, attention_mask.bool())

            # Only compute loss on non-padding tokens
            mask = attention_mask.bool().unsqueeze(-1)
            pred_masked = pred_acts * mask
            target_masked = target_acts * mask

            loss = self.loss_fn(pred_masked, target_masked)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, dataloader) -> float:
        """Evaluate MSE on a dataloader. Returns mean MSE."""
        self.probe.eval()
        self.source_model.eval()
        self.target_model.eval()

        total_mse = 0.0
        total_tokens = 0

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            source_out = self.source_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            target_out = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            source_acts = source_out.hidden_states[self.layer_idx].float()
            target_acts = target_out.hidden_states[self.layer_idx].float()

            pred_acts = self.probe(source_acts, attention_mask.bool())

            # Compute MSE only on valid tokens
            mask = attention_mask.bool()
            pred_flat = pred_acts[mask]
            target_flat = target_acts[mask]

            mse = ((pred_flat - target_flat) ** 2).sum().item()
            total_mse += mse
            total_tokens += pred_flat.shape[0] * pred_flat.shape[1]

        return total_mse / total_tokens


def save_transformer_probe(
    probe: TransformerProbe,
    save_dir: str | Path,
    layer_idx: int,
    train_mse: float | None = None,
    eval_mse: float | None = None,
):
    """Save probe weights and config."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tensors = {k: v.cpu() for k, v in probe.state_dict().items()}
    save_file(tensors, save_dir / f"layer_{layer_idx}.safetensors")

    metadata = {
        "layer_idx": layer_idx,
        "model_dim": probe.config.model_dim,
        "probe_dim": probe.config.probe_dim,
        "num_layers": probe.config.num_layers,
        "num_heads": probe.config.num_heads,
        "ffn_multiplier": probe.config.ffn_multiplier,
        "dropout": probe.config.dropout,
    }
    if train_mse is not None:
        metadata["train_mse"] = train_mse
    if eval_mse is not None:
        metadata["eval_mse"] = eval_mse

    with open(save_dir / f"layer_{layer_idx}_config.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return save_dir


def load_transformer_probe(load_dir: str | Path, layer_idx: int, device: str = "cpu"):
    """Load a saved probe."""
    load_dir = Path(load_dir)

    with open(load_dir / f"layer_{layer_idx}_config.json") as f:
        metadata = json.load(f)

    config = TransformerProbeConfig(
        model_dim=metadata["model_dim"],
        probe_dim=metadata["probe_dim"],
        num_layers=metadata["num_layers"],
        num_heads=metadata["num_heads"],
        ffn_multiplier=metadata["ffn_multiplier"],
        dropout=metadata["dropout"],
    )

    probe = TransformerProbe(config)
    weights = load_file(load_dir / f"layer_{layer_idx}.safetensors")
    probe.load_state_dict(weights)

    return probe.to(device)
