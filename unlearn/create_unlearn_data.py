import fnmatch
import json
import os
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoTokenizer

from unlearn.utils.utils import assert_type

location = "google"

if location == "mnt":
    base_path = Path("/mnt/ssd-1/lucia")
elif location == "google":
    base_path = Path("~/lucia")
else:
    base_path = Path("/projects/a5k/public/lucia")

bio_retain_path = base_path / "bio_retain"
bio_forget_path = base_path / "bio-forget"
wmdp_rewritten_path = base_path / "wmdp-lie-o-rewritten"
output_dir = base_path / "bio_tmp"


def is_debug():
    return True


def find_and_load_rmu_file(dataset_name, target_pattern):
    """Find and load a specific file from the rmu-training-data dataset.

    Args:
        dataset_name: Name of the HuggingFace dataset
            (e.g., "Unlearning/rmu-training-data")
        target_pattern: Pattern to match the file name
            (e.g., "bio-forget-corpus")

    Returns:
        Dataset loaded from the matching file
    """
    # List all files in the Hugging Face repository
    all_files = list_repo_files(repo_id=dataset_name, repo_type="dataset")
    data_files_list = [
        f
        for f in all_files
        if f.endswith(".json") or f.endswith(".jsonl") or f.endswith(".parquet")
    ]

    if is_debug():
        print(f"Found data files in {dataset_name}:")
        for f in data_files_list:
            print(f"  - {f}")

    # Try to find exact match first
    target_file = None
    for f in data_files_list:
        if target_pattern in f:
            target_file = f
            break

    if target_file is None:
        # Fallback: try to find a file that matches the pattern
        matches = fnmatch.filter(data_files_list, f"*{target_pattern}*")
        if matches:
            target_file = matches[0]
        else:
            raise ValueError(
                f"Could not find a file matching '{target_pattern}' in {dataset_name}"
            )

    if is_debug():
        print(f"Loading specific data file: {target_file}")

    # Download the file and read line by line to handle schema inconsistencies
    # Download the file to a temporary location
    local_file = hf_hub_download(
        repo_id=dataset_name, filename=target_file, repo_type="dataset"
    )

    if is_debug():
        print(f"Downloaded file to: {local_file}")

    # Read JSONL file line by line to handle schema inconsistencies
    dataset_list = []
    with open(local_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():  # Skip empty lines
                try:
                    example = json.loads(line)

                    # FIX: Handle cases where the JSON object is a simple string
                    if isinstance(example, str):
                        example = {"text": example}

                    dataset_list.append(example)
                    if is_debug() and (i + 1) % 1000 == 0:
                        print(f"Loaded {i + 1} examples...", flush=True)
                except json.JSONDecodeError as e:
                    if is_debug():
                        print(f"Warning: Skipping invalid JSON on line {i + 1}: {e}")
                    continue

    if is_debug():
        print(f"Successfully loaded {len(dataset_list)} examples")

    # Create dataset from list
    dataset = Dataset.from_list(dataset_list)
    return dataset


def tokenize_ds(datasets, tokenizer, max_length):
    """Tokenize datasets if not already tokenized."""

    def is_tokenized(example):
        return "input_ids" in example

    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
        )

    for key in datasets:
        sample = datasets[key][0]
        if not is_tokenized(sample):
            datasets[key] = datasets[key].map(
                tokenize_function,
                batched=False,
                remove_columns=["text"],
            )

    return datasets


def load_datasets():
    """Load all required datasets."""
    # Try to resolve paths - check if relative or absolute
    project_root = Path(__file__).parent.parent.parent

    def resolve_path(path):
        if os.path.isabs(path):
            return path
        full_path = project_root / path
        if full_path.exists():
            return str(full_path)
        return path

    bio_forget_path = resolve_path(bio_forget_path)  # noqa: F823
    rewritten_path = resolve_path(wmdp_rewritten_path)
    retain_path = resolve_path(bio_retain_path)

    try:
        bio_forget = Dataset.load_from_disk(bio_forget_path)
    except Exception as e:
        print(
            f"Error loading bio-forget from {bio_forget_path}, "
            f"trying to load from HuggingFace Hub... {e}"
        )
        bio_forget = find_and_load_rmu_file(
            "Unlearning/rmu-training-data", "bio-forget-corpus"
        )

    try:
        rewritten = Dataset.load_from_disk(rewritten_path)
    except Exception as e:
        print(
            f"Error loading rewritten from {rewritten_path}, "
            f"trying to load from HuggingFace Hub... {e}"
        )
        rewritten = load_dataset("Unlearning/wmdp-lie-o-rewritten", split="train")

    try:
        retain = Dataset.load_from_disk(retain_path)
    except Exception as e:
        print(
            f"Error loading retain from {retain_path}, "
            f"trying to load from HuggingFace Hub... {e}"
        )
        retain = find_and_load_rmu_file(
            "Unlearning/rmu-training-data", "bio-retain-corpus"
        )

    # Bio and rewritten have the same length.
    assert len(bio_forget) == len(rewritten), (
        f"Bio-forget and rewritten datasets must have the same length for alignment."
        f"Got {len(bio_forget)} and {len(rewritten)}."
    )

    return {
        "bio_forget": assert_type(Dataset, bio_forget),
        "rewritten": assert_type(Dataset, rewritten),
        "retain": assert_type(Dataset, retain),
    }


def truncate_transfer_dataset(example, max_seq_len):
    """Truncate and add attention masks for source and target."""
    source_ids = example["source_input_ids"]
    target_ids = example["target_input_ids"]

    # Convert to list if needed
    if not isinstance(source_ids, list):
        source_ids = source_ids.tolist()
    if not isinstance(target_ids, list):
        target_ids = target_ids.tolist()

    # Truncate
    source_ids = source_ids[:max_seq_len]
    target_ids = target_ids[:max_seq_len]

    # Create attention masks
    source_attention_mask = [1] * len(source_ids)
    target_attention_mask = [1] * len(target_ids)

    return {
        "source_input_ids": source_ids,
        "source_attention_mask": source_attention_mask,
        "target_input_ids": target_ids,
        "target_attention_mask": target_attention_mask,
    }


def truncate_retain_dataset(example, max_seq_len):
    """Process retain dataset: truncate and add attention mask."""
    input_ids = example.get("input_ids", [])

    # Convert to list if needed
    if not isinstance(input_ids, list):
        input_ids = input_ids.tolist()

    # Truncate
    input_ids = input_ids[:max_seq_len]

    # Create attention mask
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def main():
    SEQ_LEN = 1024

    os.makedirs(output_dir, exist_ok=True)
    transfer_ds_path = Path(output_dir) / "transfer_ds"
    retain_ds_path = Path(output_dir) / "mixed_retain_ds"

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize datasets
    ds = load_datasets()

    ds["bio_forget"].save_to_disk(str(Path(output_dir) / "bio_forget_ds"))
    exit()

    print("Tokenizing...", flush=True)
    ds = tokenize_ds(ds, tokenizer, SEQ_LEN)

    # Prepare ultrachat
    ultrachat = load_dataset("stingning/ultrachat", split="train")
    ultrachat = assert_type(Dataset, ultrachat)
    # Mix in at 1:2 ratio
    ultrachat = ultrachat.select(range(min(len(ultrachat), len(ds["retain"]) // 2)))
    # Flatten ultrachat conversations to text
    ultrachat = ultrachat.map(
        lambda ex: {"text": "\n".join(ex["data"])},
        remove_columns=ultrachat.column_names,
    )
    ultrachat = ultrachat.map(
        lambda ex: tokenizer(ex["text"], truncation=True, max_length=SEQ_LEN),
        remove_columns=["text"],
    )

    # Mix ultrachat with retain set
    retain_ds = concatenate_datasets([ds["retain"], ultrachat]).shuffle(seed=42)

    print("Retain set len", len(retain_ds), flush=True)
    print(
        "Processing retain dataset: truncating and adding attention masks", flush=True
    )

    retain_ds = retain_ds.map(
        lambda ex: truncate_retain_dataset(ex, SEQ_LEN),
        batched=False,
    )
    retain_ds.save_to_disk(str(retain_ds_path))

    # Prepare transfer dataset
    transfer_ds = ds["bio_forget"].rename_column("input_ids", "source_input_ids")
    transfer_ds = transfer_ds.add_column(
        "target_input_ids", ds["rewritten"]["input_ids"], new_fingerprint="transfer"
    )

    def filter(item):
        return (
            len(item["source_input_ids"]) >= SEQ_LEN
            and len(item["target_input_ids"]) >= SEQ_LEN
        )

    transfer_ds = transfer_ds.filter(filter)
    print(f"Filtered transfer dataset length: {len(transfer_ds)}", flush=True)

    print(
        "Processing transfer dataset: truncating and adding attention masks", flush=True
    )
    transfer_ds = transfer_ds.map(
        lambda ex: truncate_transfer_dataset(ex, SEQ_LEN),
        batched=False,
    )
    transfer_ds.save_to_disk(str(transfer_ds_path))


if __name__ == "__main__":
    main()
