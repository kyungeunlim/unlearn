from datasets import Dataset, concatenate_datasets, load_dataset

from unlearn.reference.cas.utils import (
    BIO_CORRUPT_REWRITTEN_DS_NAME,
    BIO_CORRUPT_SHUFFLED_DS_NAME,
    BIO_REMOVE_DS_NAME,
    RETAIN_CHAT_DS_NAME,
    RETAIN_TEXT_DS_NAME,
    cb_tokenize_function,
    hf_token,
    ultrachat_tokenize_function,
    wikitext_tokenize_function,
)


class UnlearningDataset(Dataset):
    def __init__(self, tokenized_bio_remove_dataset, interleaved_dataset):
        self.tokenized_bio_remove_dataset = tokenized_bio_remove_dataset
        self.interleaved_dataset = interleaved_dataset

    @property
    def has_keyword_mask(self):
        return "keyword_mask" in self.tokenized_bio_remove_dataset.column_names

    def __len__(self):
        return len(self.interleaved_dataset["input_ids"])

    def __getitem__(self, idx):  # type: ignore
        item = {
            "bio_remove_input_ids": self.tokenized_bio_remove_dataset["input_ids"][idx],
            "bio_remove_attention_mask": self.tokenized_bio_remove_dataset[
                "attention_mask"
            ][idx],
            "input_ids": self.interleaved_dataset["input_ids"][idx],
            "attention_mask": self.interleaved_dataset["attention_mask"][idx],
        }
        if self.has_keyword_mask:
            item["bio_remove_keyword_mask"] = self.tokenized_bio_remove_dataset[
                "keyword_mask"
            ][idx]
        return item


def get_unlearning_dataset(args, tokenizer, num_proc: int):
    # Load retain_examples
    retain_text_dataset = load_dataset(RETAIN_TEXT_DS_NAME, "wikitext-103-raw-v1")[
        "train"
    ]
    retain_text_dataset = retain_text_dataset.rename_column("page", "text")
    retain_text_dataset = retain_text_dataset.shuffle(seed=42).select(
        range(int(args.num_train_examples))
    )
    tokenized_retain_text_dataset = retain_text_dataset.map(
        lambda x: wikitext_tokenize_function(x, tokenizer),
        batched=True,
        num_proc=num_proc,
    )

    retain_datasets = [tokenized_retain_text_dataset]
    if getattr(args, "use_ultrachat", False):
        ultrachat_dataset = load_dataset(RETAIN_CHAT_DS_NAME, split="train_sft")
        ultrachat_dataset = ultrachat_dataset.shuffle(seed=42).select(
            range(int(args.num_train_examples * 0.25))
        )
        tokenized_ultrachat_dataset = ultrachat_dataset.map(
            lambda x: ultrachat_tokenize_function(x, tokenizer),
            batched=True,
            num_proc=num_proc,
        )
        retain_datasets.append(tokenized_ultrachat_dataset)

    retain_datasets = [
        concatenate_datasets(retain_datasets)
        .shuffle(seed=42)
        .select(range(args.num_train_examples))
    ]

    num_remove_to_take = (
        args.num_train_examples
        if not args.unlearn_corrupt
        else int(args.num_train_examples * (1 + args.corrupt_ratio))
    )
    bio_remove_dataset = load_dataset(BIO_REMOVE_DS_NAME, token=hf_token)
    bio_remove_dataset = bio_remove_dataset["train"].select(range(num_remove_to_take))
    tokenized_remove_dataset = bio_remove_dataset.map(
        lambda x: cb_tokenize_function(x, tokenizer),
        batched=True,
        num_proc=num_proc,
    )
    remove_datasets = [tokenized_remove_dataset]
    if args.unlearn_corrupt:
        corrupt_dataset = (
            load_dataset(BIO_CORRUPT_REWRITTEN_DS_NAME, token=hf_token)
            if args.corrupt_ds == "rewritten"
            else load_dataset(BIO_CORRUPT_SHUFFLED_DS_NAME, token=hf_token)
        )
        corrupt_dataset = corrupt_dataset["train"].select(
            range(
                args.num_train_examples,
                int(args.num_train_examples * args.corrupt_ratio),
            )
        )
        tokenized_corrupt_dataset = corrupt_dataset.map(
            lambda x: cb_tokenize_function(x, tokenizer),
            batched=True,
            num_proc=num_proc,
        )
        retain_datasets.append(tokenized_corrupt_dataset)

    all_retain_datasets = concatenate_datasets(retain_datasets)
    all_remove_datasets = concatenate_datasets(remove_datasets)

    return UnlearningDataset(all_remove_datasets, all_retain_datasets)
