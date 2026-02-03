import csv
import json
import os
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM


def _load_keyword_pattern(blocklist_path):
    with open(blocklist_path) as f:
        reader = csv.DictReader(f)
        keywords = [row["keyword"] for row in reader]
    keywords.sort(key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(kw) for kw in keywords), re.IGNORECASE)
    print(f"Loaded {len(keywords)} keywords from {blocklist_path}")
    return pattern


def add_keyword_masks(tokenized_dataset, tokenizer, blocklist_path, cache_dir):
    """Pre-compute per-token binary mask indicating blocklist keyword matches."""
    if os.path.exists(cache_dir):
        print(f"Loading cached keyword masks from {cache_dir}")
        return load_from_disk(cache_dir)

    pattern = _load_keyword_pattern(blocklist_path)

    seq_len = len(tokenized_dataset["input_ids"][0])

    def compute_mask(example):
        attn = example["attention_mask"]
        ids = example["input_ids"]
        num_real = sum(attn)
        real_ids = ids[:num_real]

        text = tokenizer.decode(real_ids, skip_special_tokens=False)
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=num_real,
        )
        offsets = encoding["offset_mapping"]

        mask = [0] * len(offsets)
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_end > start and tok_start < end:
                    mask[i] = 1

        # Pad to full seq_len
        mask = mask[:seq_len] + [0] * max(0, seq_len - len(mask))
        return {"keyword_mask": mask[:seq_len]}

    masked_dataset = tokenized_dataset.map(compute_mask, num_proc=1)

    total_keyword_tokens = 0
    total_real_tokens = 0
    for i in range(len(masked_dataset)):
        total_keyword_tokens += sum(masked_dataset["keyword_mask"][i])
        total_real_tokens += sum(masked_dataset["attention_mask"][i])
    frac = total_keyword_tokens / max(total_real_tokens, 1)
    print(
        f"Keyword mask stats: {total_keyword_tokens} keyword tokens / "
        f"{total_real_tokens} real tokens = {frac:.4f} fraction"
    )

    os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
    masked_dataset.save_to_disk(cache_dir)
    print(f"Saved keyword masks to {cache_dir}")
    return masked_dataset


def add_activation_masks(
    tokenized_dataset,
    tokenizer,
    blocklist_path,
    model_name,
    regex_cache_dir,
    cache_dir,
    threshold,
    layer_idx,
    n_keyword_samples=1000,
):
    """Pre-compute per-token mask via cosine similarity to keyword activation prototype.

    Only rank 0 computes; other ranks poll for the cache file.
    """
    done_marker = cache_dir + ".done"
    if os.path.exists(done_marker):
        print(f"Loading cached activation masks from {cache_dir}")
        return load_from_disk(cache_dir)

    global_rank = int(os.environ.get("RANK", 0))
    if global_rank != 0:
        print(f"Rank {global_rank} waiting for activation masks from rank 0...")
        while not os.path.exists(done_marker):
            time.sleep(10)
        time.sleep(5)
        print(f"Rank {global_rank} loading activation masks from {cache_dir}")
        return load_from_disk(cache_dir)

    # --- Rank 0 only below ---

    # Step 1: Get regex keyword positions (cached separately)
    regex_dataset = add_keyword_masks(
        tokenized_dataset, tokenizer, blocklist_path, regex_cache_dir
    )

    # Step 2: Sample keyword (example_idx, token_idx) pairs
    keyword_positions = []
    for i in range(len(regex_dataset)):
        mask = regex_dataset[i]["keyword_mask"]
        attn = regex_dataset[i]["attention_mask"]
        for j in range(len(mask)):
            if mask[j] == 1 and attn[j] == 1:
                keyword_positions.append((i, j))

    random.seed(42)
    random.shuffle(keyword_positions)
    keyword_positions = keyword_positions[:n_keyword_samples]
    print(f"Sampled {len(keyword_positions)} keyword positions for prototype")

    example_positions: dict[int, list[int]] = {}
    for ex_idx, tok_idx in keyword_positions:
        example_positions.setdefault(ex_idx, []).append(tok_idx)

    # Step 3: Load model for activation extraction
    print(
        f"Loading model {model_name} for activation extraction (layer {layer_idx})..."
    )
    act_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    act_model.eval()

    # Step 4: Collect keyword activations -> prototype
    keyword_activations = []
    needed_examples = sorted(example_positions.keys())
    batch_size = 4

    for batch_start in range(0, len(needed_examples), batch_size):
        batch_indices = needed_examples[batch_start : batch_start + batch_size]
        input_ids = torch.stack(
            [torch.tensor(tokenized_dataset[idx]["input_ids"]) for idx in batch_indices]
        ).cuda()

        with torch.no_grad():
            outputs = act_model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx + 1]

        for local_idx, global_idx in enumerate(batch_indices):
            for tok_idx in example_positions[global_idx]:
                keyword_activations.append(hidden[local_idx, tok_idx].float().cpu())

    prototype = torch.stack(keyword_activations).mean(dim=0)
    prototype = F.normalize(prototype, dim=0)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(prototype, os.path.join(cache_dir, "prototype.pt"))
    print(
        f"Computed keyword prototype from {len(keyword_activations)} activations "
        f"(dim={prototype.shape[0]})"
    )

    # Step 5: Compute cosine similarity for all tokens
    all_masks = []
    all_cos_sims = []
    all_attn_masks = []
    prototype_gpu = prototype.bfloat16().cuda()

    for batch_start in range(0, len(tokenized_dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(tokenized_dataset))
        input_ids = torch.stack(
            [
                torch.tensor(tokenized_dataset[i]["input_ids"])
                for i in range(batch_start, batch_end)
            ]
        ).cuda()
        attn = torch.stack(
            [
                torch.tensor(tokenized_dataset[i]["attention_mask"])
                for i in range(batch_start, batch_end)
            ]
        )

        with torch.no_grad():
            outputs = act_model(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx + 1].float()
        hidden_norm = F.normalize(hidden, dim=-1)
        cos_sim = (hidden_norm * prototype_gpu.float()).sum(dim=-1).cpu()

        batch_mask = (cos_sim >= threshold).int()
        for i in range(batch_mask.shape[0]):
            all_masks.append(batch_mask[i].tolist())
        all_cos_sims.append(cos_sim)
        all_attn_masks.append(attn)

        if batch_start % (batch_size * 32) == 0:
            print(f"  Activation mask progress: {batch_end}/{len(tokenized_dataset)}")

    del act_model, prototype_gpu
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Cosine similarity statistics
    all_cos_sims = torch.cat(all_cos_sims, dim=0)
    all_attn_masks = torch.cat(all_attn_masks, dim=0)
    real_cos_sims = all_cos_sims[all_attn_masks.bool()]
    percentiles = [10, 25, 50, 75, 80, 90, 95]
    pct_values = torch.quantile(
        real_cos_sims.float(), torch.tensor([p / 100 for p in percentiles])
    )
    pct_str = ", ".join(f"{p}th={v:.3f}" for p, v in zip(percentiles, pct_values))
    print(f"Cosine similarity percentiles: {pct_str}")

    # Mask statistics
    all_mask_tensor = torch.tensor(
        [m for masks in all_masks for m in masks], dtype=torch.int
    ).reshape(all_attn_masks.shape)
    total_masked = (all_mask_tensor.bool() & all_attn_masks.bool()).sum().item()
    total_real = all_attn_masks.sum().item()
    frac = total_masked / max(total_real, 1)
    print(
        f"Activation mask stats: {total_masked} masked tokens / "
        f"{total_real} real tokens = {frac:.4f} fraction (threshold={threshold})"
    )

    masked_dataset = tokenized_dataset.add_column("keyword_mask", all_masks)
    masked_dataset.save_to_disk(cache_dir)
    Path(done_marker).touch()
    print(f"Saved activation masks to {cache_dir}")
    return masked_dataset


def add_sae_masks(
    tokenized_dataset,
    model_name,
    latents_path,
    cache_dir,
    mask_frac,
):
    """Pre-compute per-token mask using SAE latent activation counts.

    For each token, count how many WMDP-adjacent SAE features fire (appear in
    the SAE's top-k indices). Mask the top mask_frac fraction of tokens by count.

    Only rank 0 computes; other ranks poll for the cache file.
    """
    from sparsify import SparseCoder

    done_marker = cache_dir + ".done"
    if os.path.exists(done_marker):
        print(f"Loading cached SAE masks from {cache_dir}")
        return load_from_disk(cache_dir)

    global_rank = int(os.environ.get("RANK", 0))
    if global_rank != 0:
        print(f"Rank {global_rank} waiting for SAE masks from rank 0...")
        while not os.path.exists(done_marker):
            time.sleep(10)
        time.sleep(5)
        print(f"Rank {global_rank} loading SAE masks from {cache_dir}")
        return load_from_disk(cache_dir)

    # --- Rank 0 only below ---
    with open(latents_path) as f:
        latents_config = json.load(f)

    latents_by_hookpoint: dict[str, list[int]] = {}
    for entry in latents_config["all_latents"]:
        hp = entry["hookpoint"]
        latents_by_hookpoint.setdefault(hp, []).append(entry["latent"])

    hookpoints = sorted(latents_by_hookpoint.keys())
    total_latents = sum(len(v) for v in latents_by_hookpoint.values())
    print(
        f"SAE mask: {total_latents} target latents across "
        f"{len(hookpoints)} hookpoints: {hookpoints}"
    )

    sae_name = latents_config["sae"]
    print(f"Loading SAEs from {sae_name}...")
    from huggingface_hub import snapshot_download

    sae_repo = Path(snapshot_download(sae_name))
    # SAE layers may be nested in a subdirectory
    if not any((sae_repo / hp).exists() for hp in hookpoints):
        for subdir in sae_repo.iterdir():
            if subdir.is_dir() and any((subdir / hp).exists() for hp in hookpoints):
                sae_repo = subdir
                break
    saes = {
        hp: SparseCoder.load_from_disk(sae_repo / hp, device="cuda", decoder=False)
        for hp in hookpoints
    }
    for hp, sae in saes.items():
        print(f"  {hp}: k={sae.cfg.k}, num_latents={sae.num_latents}, d_in={sae.d_in}")

    print(f"Loading model {model_name} for SAE feature extraction...")
    sae_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    sae_model.eval()

    layer_indices = {hp: int(hp.split(".")[-1]) for hp in hookpoints}

    batch_size = 2
    all_counts = []
    all_attn_masks = []

    for batch_start in range(0, len(tokenized_dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(tokenized_dataset))
        input_ids = torch.stack(
            [
                torch.tensor(tokenized_dataset[i]["input_ids"])
                for i in range(batch_start, batch_end)
            ]
        ).cuda()
        attn = torch.stack(
            [
                torch.tensor(tokenized_dataset[i]["attention_mask"])
                for i in range(batch_start, batch_end)
            ]
        )

        with torch.no_grad():
            outputs = sae_model(input_ids, output_hidden_states=True)

        bs, seq_len = input_ids.shape
        counts = torch.zeros(bs, seq_len, dtype=torch.int32)

        for hp, sae in saes.items():
            layer_idx = layer_indices[hp]
            hidden = outputs.hidden_states[layer_idx + 1]
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            enc = sae.encode(hidden_flat)
            top_indices = enc.top_indices.reshape(bs, seq_len, -1)

            for lat_idx in latents_by_hookpoint[hp]:
                matches = (top_indices == lat_idx).any(dim=-1)
                counts += matches.cpu().int()

        all_counts.append(counts)
        all_attn_masks.append(attn)

        if batch_start % (batch_size * 32) == 0:
            print(f"  SAE mask progress: {batch_end}/{len(tokenized_dataset)}")

    del sae_model, saes
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    all_counts = torch.cat(all_counts, dim=0)
    all_attn_masks = torch.cat(all_attn_masks, dim=0)

    real_counts = all_counts[all_attn_masks.bool()]
    max_count = int(real_counts.max().item())
    for c in range(max_count + 1):
        n = (real_counts == c).sum().item()
        frac = n / len(real_counts)
        print(f"  {c} latents: {n} tokens ({frac:.4f})")

    target_quantile = 1.0 - mask_frac
    threshold = torch.quantile(real_counts.float(), target_quantile).item()
    threshold = max(threshold, 1.0)
    print(f"  Count threshold: >= {threshold:.0f}")

    mask = (all_counts >= threshold).int()
    masked_real = (mask.bool() & all_attn_masks.bool()).sum().item()
    total_real = all_attn_masks.sum().item()
    actual_frac = masked_real / max(total_real, 1)
    print(
        f"SAE mask stats: {masked_real} masked / {total_real} real = "
        f"{actual_frac:.4f} (target: {mask_frac})"
    )

    all_masks = mask.tolist()
    masked_dataset = tokenized_dataset.add_column("keyword_mask", all_masks)
    os.makedirs(os.path.dirname(cache_dir) or ".", exist_ok=True)
    masked_dataset.save_to_disk(cache_dir)
    Path(done_marker).touch()
    print(f"Saved SAE masks to {cache_dir}")
    return masked_dataset
