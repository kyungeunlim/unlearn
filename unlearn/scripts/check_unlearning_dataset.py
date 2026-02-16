"""End-to-end check that get_unlearning_dataset works with CAIS data. Needs GPU + model."""
from types import SimpleNamespace

from transformers import AutoTokenizer

from unlearn.utils.unlearning_dataset import get_unlearning_dataset

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/deep-ignorance-unfiltered")
tokenizer.pad_token = tokenizer.eos_token
args = SimpleNamespace(
    num_train_examples=32,
    unlearn_corrupt=False,
    corrupt_ratio=0.5,
    corrupt_ds="rewritten",
)
ds = get_unlearning_dataset(args, tokenizer, 4)
print(f"Forget shape: {ds.tokenized_bio_remove_dataset.shape}")
print(f"Retain shape: {ds.interleaved_dataset.shape}")
print(f"Keys: {ds[0].keys()}")
print("Unlearning dataset loaded correctly.")
