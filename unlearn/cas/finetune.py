import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import gc

import torch
from attack import open_book_bio_eval
from datasets import Dataset as hf_dataset
from datasets import concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from utils import *


class WMDPEvalCallback(TrainerCallback):
    def __init__(self, model, eval_every, args, tokenizer):
        self.model = model
        self.eval_every = eval_every
        self.run_args = args
        self.tokenizer = tokenizer
        self.eval_results = []

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every == 0:
            wmdp_acc = lm_eval_model(
                self.model,
                task="wmdp_bio_aisi",
                revision=self.run_args.revision,
                tokenizer=self.tokenizer,
            )
            cloze = lm_eval_model(
                self.model,
                task="wmdp_bio_aisi_cloze_verified",
                revision=self.run_args.revision,
                tokenizer=self.tokenizer,
            )
            wmdp_acc["wmdp_bio_aisi_cloze_verified"] = cloze[
                "wmdp_bio_aisi_cloze_verified"
            ]
            print(f"Step {state.global_step}: Evaluation Results: {wmdp_acc}")
            self.eval_results.append({"step": state.global_step, "results": wmdp_acc})
            # torch.cuda.empty_cache()


if __name__ == "__main__":

    assert torch.cuda.is_available(), "CUDA is not available"

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_examples", type=int, default=512)
    parser.add_argument("--wmdp_eval_limit", type=int, default=None)
    parser.add_argument("--jailbreak_eval_limit", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=-1)
    parser.add_argument("--open_book_eval", type=bool, default=False)
    parser.add_argument("--mmlu_agieval_limit", type=int, default=None)
    parser.add_argument("--include_bio_retain", type=bool, default=False)
    parser.add_argument("--include_bio_remove", type=bool, default=False)
    parser.add_argument("--include_chat_retain", type=bool, default=False)
    parser.add_argument("--include_text_retain", type=bool, default=False)
    parser.add_argument("--include_refusal_retain", type=bool, default=False)
    parser.add_argument("--include_compliance_remove", type=bool, default=False)
    parser.add_argument(
        "--include_incompetent_compliance_retain", type=bool, default=False
    )
    parser.add_argument("--include_corrupt_rewritten", type=bool, default=False)
    parser.add_argument("--include_corrupt_shuffled", type=bool, default=False)
    parser.add_argument("--include_corrupt_deepfried", type=bool, default=False)
    parser.add_argument("--lora", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_warmup", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_chunks", type=int, default=5)
    parser.add_argument(
        "--model_name", type=str, default="allenai/OLMo-2-1124-7B-Instruct"
    )
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--save_name", type=str, default="")
    args = parser.parse_args()
    if "smollm2" not in args.model_name:
        args.hidden_dim = 4096
    else:
        args.hidden_dim = 2048

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print()

    # model, tokenizer = get_model_and_tokenizer('Unlearning/pythia1.5_baseline', revision='annealing_step_11921')
    model, tokenizer = get_model_and_tokenizer(
        args.model_name, revision=args.revision, dm="auto"
    )

    # with torch.no_grad():
    #     prompt = "Hey, are you conscious? Can you talk to me?"
    #     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    #     generate_ids = model.generate(inputs.input_ids, max_length=30)
    #     tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # wmdp_acc = lm_eval_model(model, task='wmdp_bio_aisi', limit=args.wmdp_eval_limit, tokenizer=tokenizer)
    # mmlu_acc = lm_eval_model(model, task='mmlu', limit=args.mmlu_agieval_limit, tokenizer=tokenizer)
    # print(f'***\nOriginal wmdp_acc: {wmdp_acc}, original mmlu_acc {mmlu_acc}\n***')

    training_datasets = []

    if args.include_chat_retain:
        retain_chat_dataset = load_dataset(RETAIN_CHAT_DS_NAME)["train_sft"]
        retain_chat_dataset = retain_chat_dataset.shuffle(seed=42).select(
            range(args.num_train_examples)
        )
        tokenized_retain_chat_dataset = retain_chat_dataset.map(
            lambda x: ultrachat_tokenize_function(x, tokenizer, truncate=False),
            batched=True,
        )
        training_datasets.append(tokenized_retain_chat_dataset)

    if args.include_text_retain:
        retain_text_dataset = load_dataset(RETAIN_TEXT_DS_NAME, "wikitext-103-raw-v1")[
            "train"
        ]
        retain_text_dataset = retain_text_dataset.rename_column("page", "text")
        retain_text_dataset = retain_text_dataset.shuffle(seed=42).select(
            range(args.num_train_examples)
        )
        tokenized_retain_text_dataset = retain_text_dataset.map(
            lambda x: wikitext_tokenize_function(x, tokenizer, truncate=False),
            batched=True,
        )
        training_datasets.append(tokenized_retain_text_dataset)

    if args.include_refusal_retain:
        retain_refusal_compliance_dataset = load_dataset(
            RETAIN_REFUSAL_COMPLIANCE_DS_NAME
        )["train"]
        retain_refusal_compliance_dataset = retain_refusal_compliance_dataset.shuffle(
            seed=42
        ).select(range(args.num_train_examples))
        tokenized_retain_refusal_dataset = retain_refusal_compliance_dataset.map(
            lambda x: refusal_compliance_tokenize_function(x, tokenizer, refuse=True),
            batched=True,
        )
        training_datasets.append(tokenized_retain_refusal_dataset)

    if args.include_compliance_remove:
        remove_refusal_compliance_dataset = load_dataset(
            RETAIN_REFUSAL_COMPLIANCE_DS_NAME
        )["train"]
        remove_refusal_compliance_dataset = remove_refusal_compliance_dataset.shuffle(
            seed=42
        ).select(range(args.num_train_examples))
        tokenized_remove_compliance_dataset = remove_refusal_compliance_dataset.map(
            lambda x: refusal_compliance_tokenize_function(x, tokenizer, refuse=False),
            batched=True,
        )
        training_datasets.append(tokenized_remove_compliance_dataset)

    if args.include_incompetent_compliance_retain:
        retain_incompetent_compliance_dataset = load_dataset(
            RETAIN_INCOMPETENT_COMPLIANCE_DS_NAME, token=hf_token
        )["train"]
        retain_incompetent_compliance_dataset = (
            retain_incompetent_compliance_dataset.shuffle(seed=42).select(
                range(args.num_train_examples)
            )
        )
        tokenized_incompetent_compliance_dataset = (
            retain_incompetent_compliance_dataset.map(
                lambda x: incompetent_compliance_tokenize_function(
                    x, tokenizer, refuse=False
                ),
                batched=True,
            )
        )
        training_datasets.append(tokenized_incompetent_compliance_dataset)

    if args.include_bio_retain:
        bio_retain_dataset = load_dataset(BIO_RETAIN_DS_NAME, "bio-retain-corpus")
        bio_retain_dataset = (
            bio_retain_dataset["train"]
            .shuffle(seed=42)
            .select(range(args.num_train_examples))
        )
        tokenized_bio_retain_dataset = bio_retain_dataset.map(
            lambda x: cb_retain_tokenize_function(x, tokenizer, truncate=False),
            batched=True,
        )
        training_datasets.append(tokenized_bio_retain_dataset)

    if args.include_bio_remove:
        bio_remove_dataset = load_dataset(BIO_REMOVE_DS_NAME, token=hf_token)
        bio_remove_dataset = (
            bio_remove_dataset["train"]
            .shuffle(seed=42)
            .select(range(args.num_train_examples))
        )
        tokenized_bio_remove_dataset = bio_remove_dataset.map(
            lambda x: cb_tokenize_function(x, tokenizer, truncate=False), batched=True
        )
        training_datasets.append(tokenized_bio_remove_dataset)

    if args.include_corrupt_rewritten:
        corrupt_dataset = load_dataset(BIO_CORRUPT_REWRITTEN_DS_NAME, token=hf_token)
        corrupt_dataset = (
            corrupt_dataset["train"]
            .shuffle(seed=42)
            .select(range(args.num_train_examples))
        )
        tokenized_corrupt_dataset = corrupt_dataset.map(
            lambda x: cb_tokenize_function(x, tokenizer, truncate=False), batched=True
        )
        training_datasets.append(tokenized_corrupt_dataset)

    if args.include_corrupt_shuffled:
        corrupt_dataset = load_dataset(BIO_CORRUPT_SHUFFLED_DS_NAME, token=hf_token)
        corrupt_dataset = (
            corrupt_dataset["train"]
            .shuffle(seed=42)
            .select(range(args.num_train_examples))
        )
        tokenized_corrupt_dataset = corrupt_dataset.map(
            lambda x: cb_tokenize_function(x, tokenizer, truncate=False), batched=True
        )
        training_datasets.append(tokenized_corrupt_dataset)

    if args.include_corrupt_deepfried:
        corrupt_dataset = load_dataset(BIO_CORRUPT_DEEPFRIED_DS_NAME, token=hf_token)
        corrupt_dataset = (
            corrupt_dataset["train"]
            .shuffle(seed=42)
            .select(range(args.num_train_examples))
        )
        tokenized_corrupt_dataset = corrupt_dataset.map(
            lambda x: cb_tokenize_function(x, tokenizer, truncate=False), batched=True
        )
        training_datasets.append(tokenized_corrupt_dataset)

    interleaved_dataset = concatenate_datasets(training_datasets).shuffle()
    interleaved_dataset = interleaved_dataset.remove_columns(
        [
            col
            for col in interleaved_dataset.column_names
            if col not in ["input_ids", "token_type_ids", "attention_mask", "labels"]
        ]
    )

    # Chunk interleaved_dataset into pieces of size 2048 tokens
    def chunk_example(
        example,
        chunk_size=MAX_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
        max_chunks=5,
    ):
        input_ids = example["input_ids"]
        attention_mask = example.get("attention_mask", [1] * len(input_ids))
        labels = example.get("labels", input_ids.copy())
        token_type_ids = example.get("token_type_ids", [0] * len(input_ids))
        chunks = []
        for i in range(0, len(input_ids), chunk_size):
            chunk_input_ids = input_ids[i : i + chunk_size]
            chunk_attention_mask = attention_mask[i : i + chunk_size]
            chunk_labels = labels[i : i + chunk_size]
            chunk_token_type_ids = token_type_ids[i : i + chunk_size]
            # Pad if needed
            pad_len = chunk_size - len(chunk_input_ids)
            if pad_len > 0:
                chunk_input_ids += [pad_token_id] * pad_len
                chunk_attention_mask += [0] * pad_len
                chunk_labels += [-100] * pad_len
                chunk_token_type_ids += [0] * pad_len
            chunks.append(
                {
                    "input_ids": chunk_input_ids,
                    "attention_mask": chunk_attention_mask,
                    "labels": chunk_labels,
                    "token_type_ids": chunk_token_type_ids,
                }
            )
            if i >= (
                (max_chunks - 1) * chunk_size
            ):  # Limit the number of chunks to avoid memory issues
                break
        return chunks

    # if args.chunk:
    chunked_examples = []
    for i, example in enumerate(interleaved_dataset):
        chunked_examples.extend(
            chunk_example(example, chunk_size=MAX_LENGTH, max_chunks=args.max_chunks)
        )
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1} examples, total chunks: {len(chunked_examples)}")
    del interleaved_dataset
    interleaved_dataset = hf_dataset.from_list(chunked_examples).shuffle(seed=42)
    # shorten to 85k examples if there are more than 85k
    if len(interleaved_dataset) > 85000:
        interleaved_dataset = interleaved_dataset.select(range(85000))
    del chunked_examples

    print(interleaved_dataset)
    gc.collect()
    torch.cuda.empty_cache()
    # quit()

    if args.lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=(
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
                if "OLMo" in args.model_name
                else None
            ),
        )
        model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=args.lr,
        gradient_accumulation_steps=16,  # used to be 32
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        gradient_checkpointing=True,  # to avoid oom errors
        fp16=True,  # to avoid oom errors,
        save_strategy="no",  # Disable model saving
        warmup_steps=args.lr_warmup,  # Warmup steps for learning rate scheduler
        logging_strategy="steps",  # Enable logging during training
        logging_steps=10,
    )

    if args.eval_every > 0:
        callbacks = [WMDPEvalCallback(model, args.eval_every, args, tokenizer)]
    else:
        callbacks = []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=interleaved_dataset,
        eval_dataset=interleaved_dataset,
        callbacks=callbacks,
    )

    for param in model.parameters():
        param.requires_grad = True
    model.train()
    trainer.train()

    if args.eval_every > 0:
        print(f"***\nWMDP eval performance over time: {callbacks[0].eval_results}\n***")

    # mmlu_acc = lm_eval_model(model, task='mmlu', limit=args.mmlu_agieval_limit, revision=args.revision, tokenizer=tokenizer)

    if "smollm2" not in args.model_name and args.eval_every < 0:
        wmdp_acc = lm_eval_model(
            model,
            task="wmdp_bio_aisi",
            limit=args.wmdp_eval_limit,
            revision=args.revision,
            tokenizer=tokenizer,
        )
        print(f"***\nFinal wmdp_acc: {wmdp_acc}\n***")
        pass

    elif "smollm2" in args.model_name:
        jailbreak_score = jailbreak_eval_model(
            model, tokenizer, num_examples=args.jailbreak_eval_limit, pfx=None, num_fs=0
        )
        print(f"***\nFinal jailbreak_score: {jailbreak_score}\n***")

    if args.open_book_eval:
        custom_eval_acc = open_book_bio_eval(model, tokenizer, open_book=True)
        print(f"***\nCustom bio eval acc: {custom_eval_acc}\n***")

    if args.lora:
        model = model.merge_and_unload()

    if args.save_name:
        if "models/" in args.model_name:
            args.model_name = args.model_name.replace("models/", "")
        model.save_pretrained(f"./models/{args.model_name + '_' + args.save_name}")
        tokenizer.save_pretrained(f"./models/{args.model_name + '_' + args.save_name}")

    print("Done :)\n\n\n")
