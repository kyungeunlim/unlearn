"""Sanity check that CAIS datasets load with expected columns."""
from datasets import load_dataset

# Retain: should have 'text' column, ~61K examples
retain = load_dataset("cais/wmdp-corpora", "bio-retain-corpus", split="train")
print(f"Retain: {len(retain)} examples, columns: {retain.column_names}")
assert "text" in retain.column_names

# Forget: should have title/abstract/text, ~24K examples (gated)
forget = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
print(f"Forget: {len(forget)} examples, columns: {forget.column_names}")
assert all(c in forget.column_names for c in ["title", "abstract", "text"])

print("Datasets loaded correctly.")
