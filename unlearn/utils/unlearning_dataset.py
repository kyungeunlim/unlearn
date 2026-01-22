from datasets import Dataset, concatenate_datasets, load_dataset
from unlearn.reference.cas.utils import (
    BIO_CORRUPT_REWRITTEN_DS_NAME,
    BIO_CORRUPT_SHUFFLED_DS_NAME,
    BIO_REMOVE_DS_NAME,
    BIO_RETAIN_DS_NAME,
    RETAIN_INCOMPETENT_COMPLIANCE_DS_NAME,
    RETAIN_REFUSAL_COMPLIANCE_DS_NAME,
    RETAIN_TEXT_DS_NAME,
    cb_retain_tokenize_function,
    cb_tokenize_function,
    hf_token,
    incompetent_compliance_tokenize_function,
    refusal_compliance_tokenize_function,
    wikitext_tokenize_function,
)


class UnlearningDataset(Dataset):
    def __init__(self, tokenized_bio_remove_dataset, interleaved_dataset):
        self.tokenized_bio_remove_dataset = tokenized_bio_remove_dataset
        self.interleaved_dataset = interleaved_dataset

    def __len__(self):
        return len(self.interleaved_dataset["input_ids"])

    def __getitem__(self, idx): # type: ignore
        return {
            "bio_remove_input_ids": self.tokenized_bio_remove_dataset["input_ids"][idx],
            "bio_remove_attention_mask": self.tokenized_bio_remove_dataset[
                "attention_mask"
            ][idx],
            "input_ids": self.interleaved_dataset["input_ids"][idx],
            "attention_mask": self.interleaved_dataset["attention_mask"][idx],
        }



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
    if (
        args.model_name == "allenai/OLMo-2-1124-7B-Instruct"
        or "Unlearning" in args.model_name
    ):
        bio_retain_dataset = load_dataset(BIO_RETAIN_DS_NAME, "bio-retain-corpus")
        bio_retain_dataset = (
            bio_retain_dataset["train"]
            .shuffle(seed=42)
            .select(range(int(args.num_train_examples * 0.25)))
        )
        tokenized_bio_retain_dataset = bio_retain_dataset.map(
            lambda x: cb_retain_tokenize_function(x, tokenizer),
            batched=True,
            num_proc=num_proc,
        )
        retain_datasets.append(tokenized_bio_retain_dataset)
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
    if "smollm2" not in args.model_name:
        bio_remove_dataset = load_dataset(BIO_REMOVE_DS_NAME, token=hf_token)
        bio_remove_dataset = bio_remove_dataset["train"].select(
            range(num_remove_to_take)
        )
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
    else:
        remove_refusal_compliance_dataset = load_dataset(
            RETAIN_REFUSAL_COMPLIANCE_DS_NAME
        )["train"]
        remove_refusal_compliance_dataset = remove_refusal_compliance_dataset.shuffle(
            seed=42
        ).select(range(args.num_train_examples))
        tokenized_remove_compliance_dataset = remove_refusal_compliance_dataset.map(
            lambda x: refusal_compliance_tokenize_function(x, tokenizer, refuse=False),
            batched=True,
            num_proc=num_proc,
        )
        remove_datasets = [tokenized_remove_compliance_dataset]
        if args.unlearn_corrupt:
            corrupt_dataset = load_dataset(
                RETAIN_INCOMPETENT_COMPLIANCE_DS_NAME, token=hf_token
            )["train"]
            corrupt_dataset = corrupt_dataset.select(
                range(
                    args.num_train_examples,
                    int(args.num_train_examples * args.corrupt_ratio),
                )
            )
            tokenized_corrupt_dataset = corrupt_dataset.map(
                lambda x: incompetent_compliance_tokenize_function(
                    x, tokenizer, refuse=False
                ),
                batched=True,
                num_proc=num_proc,
            )
            retain_datasets.append(tokenized_corrupt_dataset)

    all_retain_datasets = concatenate_datasets(retain_datasets)
    all_remove_datasets = concatenate_datasets(remove_datasets)

    return UnlearningDataset(all_remove_datasets, all_retain_datasets)

