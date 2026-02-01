import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import HfApi, create_repo
from safetensors.torch import load_file, save_file
from tqdm import tqdm


class OnlineAffineFitter:
    """
    Accumulates statistics to solve Y = XW + b using Ridge Regression
    without holding all activations in memory.
    """

    def __init__(self, hidden_dim, device="cpu", alpha=0.01):
        self.dim = hidden_dim
        self.device = device
        self.alpha = alpha

        # Accumulators (using float64 for precision during summation)
        self.n_samples = 0
        self.sum_x = torch.zeros(hidden_dim, dtype=torch.float64, device=device)
        self.sum_y = torch.zeros(hidden_dim, dtype=torch.float64, device=device)
        self.xtx = torch.zeros(
            (hidden_dim, hidden_dim), dtype=torch.float64, device=device
        )
        self.xty = torch.zeros(
            (hidden_dim, hidden_dim), dtype=torch.float64, device=device
        )

    def update(self, x, y):
        """
        x: [Batch * Seq, Dim] - Source Activations
        y: [Batch * Seq, Dim] - Target Activations
        """
        x = x.to(self.device).double()
        y = y.to(self.device).double()

        self.n_samples += x.shape[0]
        self.sum_x += x.sum(dim=0)
        self.sum_y += y.sum(dim=0)

        # X^T X
        self.xtx += x.T @ x
        # X^T Y
        self.xty += x.T @ y

    def solve(self):
        """
        Solves for W and b.
        Returns a torch.nn.Linear module initialized with the mapping.
        """
        # Compute Means
        mu_x = (self.sum_x / self.n_samples).float()
        mu_y = (self.sum_y / self.n_samples).float()

        # Compute Centered Covariances
        # Cov(X,X) = sum(x^2) - n * mu_x^2
        # We need to be careful with shapes here for outer product
        mu_x_64 = self.sum_x / self.n_samples
        mu_y_64 = self.sum_y / self.n_samples

        cov_xx = self.xtx - self.n_samples * torch.outer(mu_x_64, mu_x_64)
        cov_xy = self.xty - self.n_samples * torch.outer(mu_x_64, mu_y_64)

        # Ridge Regression: W = (Cov_XX + alpha*I)^-1 @ Cov_XY
        eye = torch.eye(self.dim, device=self.device, dtype=torch.float64)
        cov_xx_reg = cov_xx + (self.alpha * eye)

        # Solve
        W = torch.linalg.solve(cov_xx_reg, cov_xy).float()

        # Calculate Bias: b = mu_y - mu_x @ W
        b = mu_y - (mu_x @ W)

        # Create Linear Module
        module = nn.Linear(self.dim, self.dim, bias=True)
        with torch.no_grad():
            module.weight.copy_(W.T)  # nn.Linear stores as (Out, In)
            module.bias.copy_(b)

        return module.to(mu_x.device)


def train_affine_transform(
    source_model,
    target_model,
    dataset,
    target_layers,
    num_examples=100_000,
    batch_size=4,
    device="cuda",
    alpha=0.01,
):
    """
    Passes data through both models and trains affine transforms for specified layers.
    Returns: Dict[layer_idx, nn.Linear]
    """
    print(f"🔄 Training Affine Transform (Old -> New) on {num_examples} examples...")

    # 1. Setup Fitters for each layer
    hidden_dim = target_model.config.hidden_size
    fitters = {
        layer: OnlineAffineFitter(hidden_dim, device=device, alpha=alpha)
        for layer in target_layers
    }

    # 2. Prepare Data
    # Assuming dataset supports slicing or we just take the first N
    if hasattr(dataset, "select"):
        subset = dataset.select(range(min(num_examples, len(dataset))))
    else:
        subset = [dataset[i] for i in range(min(num_examples, len(dataset)))]

    def collate(batch_items, device):
        # Handle both dict-of-lists (HF dataset slice) and list-of-dicts formats
        if isinstance(batch_items, dict):
            # HF dataset slice format: {"input_ids": [[...], [...]], ...}
            input_ids_raw = batch_items["input_ids"]
            attention_mask_raw = batch_items["attention_mask"]
        else:
            # List of dicts format
            input_ids_raw = [x["input_ids"] for x in batch_items]
            attention_mask_raw = [x["attention_mask"] for x in batch_items]

        # Process Input IDs
        input_ids_list = []
        for ids in input_ids_raw:
            tensor = torch.tensor(ids, dtype=torch.long)
            if tensor.ndim > 1:
                tensor = tensor.view(-1)
            input_ids_list.append(tensor)

        input_ids = torch.stack(input_ids_list).to(device)

        # Process Attention Mask
        mask_list = []
        for mask in attention_mask_raw:
            tensor = torch.tensor(mask)
            if tensor.ndim > 1:
                tensor = tensor.view(-1)
            mask_list.append(tensor)

        attention_mask = torch.stack(mask_list).to(device=device, dtype=torch.bool)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # 3. Loop
    source_model.eval()
    target_model.eval()

    num_batches = (len(subset) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Collecting Activations"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(subset))
            batch = subset[start_idx:end_idx]

            inputs = collate(batch, device)

            mask = inputs["attention_mask"].bool().reshape(-1)  # [Batch*Seq]

            # Forward Source (Old)
            out_source = source_model(**inputs, output_hidden_states=True)

            # Forward Target (New)
            out_target = target_model(**inputs, output_hidden_states=True)

            # Update Fitters
            for layer_idx in target_layers:
                # [Batch, Seq, Dim]
                act_src = out_source.hidden_states[layer_idx]
                act_tgt = out_target.hidden_states[layer_idx]

                # Flatten: [Batch*Seq, Dim]
                flat_src = act_src.flatten(0, 1)
                flat_tgt = act_tgt.flatten(0, 1)

                # Filter Padding
                clean_src = flat_src[mask]
                clean_tgt = flat_tgt[mask]

                fitters[layer_idx].update(clean_src, clean_tgt)

    # 4. Solve
    print("📐 Solving Linear Systems...")
    affine_transforms = {}
    for layer_idx, fitter in fitters.items():
        affine_transforms[layer_idx] = fitter.solve()

    print("✅ Affine Transformation Trained.")
    return affine_transforms


def evaluate_affine_mse(
    affine_transforms,
    source_model,
    target_model,
    dataset,
    target_layers,
    num_examples=10000,
    batch_size=4,
    device="cuda",
):
    """
    Evaluate MSE between affine-transformed source activations and target activations.
    Returns: Dict with per-layer MSE and mean MSE.
    """
    print(f"Evaluating affine transform MSE on {num_examples} examples...")

    if hasattr(dataset, "select"):
        subset = dataset.select(range(min(num_examples, len(dataset))))
    else:
        subset = [dataset[i] for i in range(min(num_examples, len(dataset)))]

    def collate(batch_items, device):
        # Handle both dict-of-lists (HF dataset slice) and list-of-dicts formats
        if isinstance(batch_items, dict):
            input_ids_raw = batch_items["input_ids"]
            attention_mask_raw = batch_items["attention_mask"]
        else:
            input_ids_raw = [x["input_ids"] for x in batch_items]
            attention_mask_raw = [x["attention_mask"] for x in batch_items]

        input_ids_list = []
        for ids in input_ids_raw:
            tensor = torch.tensor(ids, dtype=torch.long)
            if tensor.ndim > 1:
                tensor = tensor.view(-1)
            input_ids_list.append(tensor)

        input_ids = torch.stack(input_ids_list).to(device)

        mask_list = []
        for mask in attention_mask_raw:
            tensor = torch.tensor(mask)
            if tensor.ndim > 1:
                tensor = tensor.view(-1)
            mask_list.append(tensor)

        attention_mask = torch.stack(mask_list).to(device=device, dtype=torch.bool)

        return dict(input_ids=input_ids, attention_mask=attention_mask)

    source_model.eval()
    target_model.eval()

    layer_mse_accum = {layer: 0.0 for layer in target_layers}
    layer_count = {layer: 0 for layer in target_layers}

    num_batches = (len(subset) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Evaluating MSE"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(subset))
            batch = subset[start_idx:end_idx]

            inputs = collate(batch, device)
            mask = inputs["attention_mask"].bool().reshape(-1)

            out_source = source_model(**inputs, output_hidden_states=True)
            out_target = target_model(**inputs, output_hidden_states=True)

            for layer_idx in target_layers:
                act_src = out_source.hidden_states[layer_idx]
                act_tgt = out_target.hidden_states[layer_idx]

                flat_src = act_src.flatten(0, 1)
                flat_tgt = act_tgt.flatten(0, 1)

                clean_src = flat_src[mask]
                clean_tgt = flat_tgt[mask]

                if layer_idx in affine_transforms:
                    transform = affine_transforms[layer_idx].to(device)
                    transformed = transform(clean_src.float())
                else:
                    transformed = clean_src.float()

                mse = torch.nn.functional.mse_loss(
                    transformed, clean_tgt.float()
                ).item()

                layer_mse_accum[layer_idx] += mse * clean_src.shape[0]
                layer_count[layer_idx] += clean_src.shape[0]

    metrics = {}
    for layer_idx in target_layers:
        if layer_count[layer_idx] > 0:
            metrics[f"layer_{layer_idx}"] = (
                layer_mse_accum[layer_idx] / layer_count[layer_idx]
            )
        else:
            metrics[f"layer_{layer_idx}"] = 0.0

    metrics["mean_mse"] = sum(metrics[f"layer_{l}"] for l in target_layers) / len(
        target_layers
    )

    print(f"Mean MSE: {metrics['mean_mse']:.6f}")
    return metrics


def save_affine_transforms(affine_transforms, save_dir, alpha=0.01):
    """
    Save affine transforms to a directory as safetensors + metadata JSON.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    layer_indices = []
    hidden_dim = None

    for layer_idx, linear in affine_transforms.items():
        layer_indices.append(layer_idx)
        tensors[f"layer_{layer_idx}.weight"] = linear.weight.data.cpu()
        tensors[f"layer_{layer_idx}.bias"] = linear.bias.data.cpu()
        if hidden_dim is None:
            hidden_dim = linear.in_features

    save_file(tensors, save_dir / "affine_transforms.safetensors")

    metadata = {
        "layer_indices": sorted(layer_indices),
        "hidden_dim": hidden_dim,
        "alpha": alpha,
    }

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved affine transforms to {save_dir}")
    return save_dir


def load_affine_transforms(load_dir, device="cpu"):
    """
    Load affine transforms from a directory.
    Returns: Dict[int, nn.Linear]
    """
    load_dir = Path(load_dir)

    with open(load_dir / "metadata.json") as f:
        metadata = json.load(f)

    weights = load_file(load_dir / "affine_transforms.safetensors")

    affine_transforms = {}
    hidden_dim = metadata["hidden_dim"]

    for layer_idx in metadata["layer_indices"]:
        linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        linear.weight.data = weights[f"layer_{layer_idx}.weight"]
        linear.bias.data = weights[f"layer_{layer_idx}.bias"]
        affine_transforms[layer_idx] = linear.to(device)

    return affine_transforms


def upload_affine_transforms_to_hub(
    affine_transforms,
    repo_id,
    mse_metrics,
    source_model_name,
    target_model_name,
    alpha=0.01,
    num_training_examples=None,
    private=False,
):
    """
    Upload affine transforms to HuggingFace Hub with a model card.
    """
    print(f"Uploading affine transforms to {repo_id}...")

    api = HfApi()

    try:
        create_repo(repo_id, private=private, exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        save_affine_transforms(affine_transforms, tmpdir, alpha=alpha)

        layer_indices = sorted(affine_transforms.keys())
        hidden_dim = next(iter(affine_transforms.values())).in_features

        mse_table = "\n".join(
            f"| {layer_idx} | {mse_metrics.get(f'layer_{layer_idx}', 'N/A'):.6f} |"
            for layer_idx in layer_indices
        )

        model_card = f"""---
tags:
  - affine-transform
  - activation-mapping
library_name: safetensors
---

# Affine Transform: {source_model_name} → {target_model_name}

Learned affine transformation mapping hidden state activations
from a source checkpoint to a target model.

## Usage

```python
from safetensors.torch import load_file
import torch.nn as nn
from huggingface_hub import hf_hub_download

# Download files
weights_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="affine_transforms.safetensors",
)
metadata_path = hf_hub_download(repo_id="{repo_id}", filename="metadata.json")

# Load
import json
with open(metadata_path) as f:
    metadata = json.load(f)

weights = load_file(weights_path)
affine_transforms = {{}}
for layer_idx in metadata["layer_indices"]:
    linear = nn.Linear(metadata["hidden_dim"], metadata["hidden_dim"], bias=True)
    linear.weight.data = weights[f"layer_{{layer_idx}}.weight"]
    linear.bias.data = weights[f"layer_{{layer_idx}}.bias"]
    affine_transforms[layer_idx] = linear
```

## MSE Metrics

| Layer | MSE |
|-------|-----|
{mse_table}

**Mean MSE: {mse_metrics.get('mean_mse', 'N/A'):.6f}**

## Training Details

- **Source Model:** {source_model_name}
- **Target Model:** {target_model_name}
- **Hidden Dimension:** {hidden_dim}
- **Ridge Alpha:** {alpha}
- **Layers:** {layer_indices}
{f'- **Training Examples:** {num_training_examples}' if num_training_examples else ''}
"""

        with open(tmpdir / "README.md", "w") as f:
            f.write(model_card)

        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_id,
            commit_message="Upload affine transforms with MSE metrics",
        )

    print(f"Uploaded to https://huggingface.co/{repo_id}")
    return repo_id
