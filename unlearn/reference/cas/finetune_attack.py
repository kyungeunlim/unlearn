import argparse
import os
import sys

import torch
from datasets import Dataset as hf_dataset
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

sys.path.append("./lm-evaluation-harness")

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # Enable expandable segments for PyTorch CUDA memory management
)
HIDDEN_DIM = 4096  # hidden dimension of the model representations
MAX_LENGTH = 2048  # maximum token length of sequence


class SimpleWMDPEvalCallback(TrainerCallback):
    """A simple callback to evaluate the model on WMDP task every `eval_every` steps."""

    def __init__(self, model, eval_every, args, tokenizer):
        self.model = model
        self.eval_every = eval_every
        self.run_args = args
        self.tokenizer = tokenizer
        self.eval_results = []

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every == 0:
            wmdp_acc = lm_eval_model(self.model, task="wmdp_bio")
            print(f"Step {state.global_step}: Evaluation Results: {wmdp_acc}")
            self.eval_results.append({"step": state.global_step, "results": wmdp_acc})


def get_model_and_tokenizer(model_name, dm="auto"):
    """Load the model and tokenizer from Hugging Face."""
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=dm)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<|padding|>",
            "eos_token": "<|endoftext|>",
            "bos_token": "<|startoftext|>",
        }
    )
    tokenizer.padding_side = "left"
    return model, tokenizer


def lm_eval_model(model, task="wmdp_bio_robust", limit=None):
    """Evaluate the model on a specific task using the HFLM evaluator."""
    model.eval()
    with torch.no_grad():
        hflm_model = HFLM(model)
        eval_results = evaluator.simple_evaluate(
            model=hflm_model,
            tasks=[task],
            device=model.device,
            verbosity="ERROR",
            limit=limit,
            num_fewshot=0,
        )
        del hflm_model
        try:
            acc = {task: eval_results["results"][task]["acc,none"]}
        except:
            acc = eval_results["results"]
        return acc


def get_wmdp_bio_forget():
    """Load the WMDP Bio forget dataset."""
    # hf_token = os.getenv("HF_TOKEN")
    # assert (
    #     hf_token
    # ), "Please set the HF_TOKEN environment variable with your Hugging Face token to access the WMDP Bio forget dataset. The dataset can be requested at https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus"
    # wmdp_bio_forget = load_dataset("cais/wmdp-bio-forget-corpus")
    wmdp_bio_forget = load_dataset("Unlearning/WMDP-Bio-Remove-Dataset")
    return wmdp_bio_forget


def tokenize_examples_fn(examples, tokenizer):
    """Tokenize the examples for training."""
    cb_examples = [
        examples["title"][i]
        + "\n\n"
        + examples["abstract"][i]
        + "\n\n"
        + examples["text"][i]
        for i in range(len(examples["title"]))
    ]
    tokenized_output = tokenizer(cb_examples, padding="max_length", truncation=False)
    tokenized_output["labels"] = tokenized_output[
        "input_ids"
    ].copy()  # Add labels for computing loss
    return tokenized_output


def chunk_example(example, tokenizer, chunk_size=MAX_LENGTH, max_chunks=5):
    """Chunk the example into smaller parts of size `chunk_size`."""
    pad_token_id = tokenizer.pad_token_id
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="EleutherAI/deep-ignorance-unfiltered"
    )  # Model name
    parser.add_argument(
        "--save_name", type=str, default="finetune_attacked"
    )  # where to save the model
    parser.add_argument(
        "--num_train_examples", type=int, default=24453
    )  # 24453 is the number of examples in the wmdp_bio dataset
    parser.add_argument(
        "--epochs", type=int, default=2
    )  # Number of epochs for training
    parser.add_argument(
        "--max_chunks", type=int, default=5
    )  # Maximum number of MAX_LENGTH token chunks per example
    parser.add_argument(
        "--wmdp_eval_limit", type=int, default=None
    )  # None means no limit
    parser.add_argument(
        "--eval_every", type=int, default=500
    )  # Evaluate using SimpleWMDPEvalCallback every `eval_every` steps
    parser.add_argument("--lora", type=bool, default=False)  # Use LoRA for training
    parser.add_argument("--lr", type=float, default=2e-5)  # Learning rate for training
    args = parser.parse_args()

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print()

    return args


if __name__ == "__main__":

    # Check if CUDA is available
    assert torch.cuda.is_available(), "CUDA is not available"

    # Parse command line arguments
    args = parse_args()

    # Get the model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_name)

    # Load the WMDP Bio forget dataset and tokenize it
    wmdp_bio_forget = get_wmdp_bio_forget()
    training_data = (
        wmdp_bio_forget["train"].shuffle(seed=42).select(range(args.num_train_examples))
    )
    tokenized_training_data = training_data.map(
        lambda x: tokenize_examples_fn(x, tokenizer), batched=True
    )

    # Chunk the tokenized examples into parts of size `MAX_LENGTH`
    chunked_examples = []
    for i, example in enumerate(tokenized_training_data):
        chunked_examples.extend(
            chunk_example(
                example, tokenizer, chunk_size=MAX_LENGTH, max_chunks=args.max_chunks
            )
        )
        if (i + 1) % 1000 == 0:
            print(f"Processed {i+1} examples, total chunks: {len(chunked_examples)}")

    # Create a Hugging Face dataset from the chunked examples
    final_dataset = hf_dataset.from_list(chunked_examples).shuffle(seed=42)
    del chunked_examples
    del tokenized_training_data

    print("Training dataset:")
    print(final_dataset)

    # If LoRA is enabled, configure the model for LoRA training
    if args.lora:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
        )
        model = get_peft_model(model, lora_config)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # output directory for the results
        learning_rate=args.lr,  # learning rate for training
        gradient_accumulation_steps=16,  # adjust if desired
        per_device_train_batch_size=1,  # adjust if desired
        per_device_eval_batch_size=1,  # adjust if desired
        num_train_epochs=args.epochs,  # wumber of epochs for training
        weight_decay=0.01,  # weight decay for training
        gradient_checkpointing=True,  # reduces risk of oom errors, adjust if needed
        fp16=True,  # reduces risk of oom errors, adjust if needed
        save_strategy="no",  # disable automatic model saving, adjust if needed
        warmup_steps=0,  # can stabilize training, adjust if needed
        logging_strategy="steps",  # Enable logging during training
        logging_steps=100,  # Log every 100 steps
        report_to=[],  # disable wandb
    )

    # If `eval_every` is set, use the SimpleWMDPEvalCallback to evaluate the model on WMDP task every `eval_every` steps
    if args.eval_every > 0:
        callbacks = [SimpleWMDPEvalCallback(model, args.eval_every, args, tokenizer)]
    else:
        callbacks = []

    # Create the Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset,
        eval_dataset=final_dataset,
        callbacks=callbacks,
    )

    # Set all parameters to trainable just in case they aren't
    for param in model.parameters():
        param.requires_grad = True

    # Train the model
    model.train()
    trainer.train()

    # Evaluate the model on WMDP task after training
    if args.eval_every > 0:
        print(f"***\nWMDP performance over time:\n{callbacks[0].eval_results}\n***")

    if args.save_name:
        # If LoRA is enabled, merge the LoRA weights into the model
        if args.lora:
            model = model.merge_and_unload()
        # Save the model and tokenizer
        model.save_pretrained(f"{args.model_name + '_' + args.save_name}")
        tokenizer.save_pretrained(f"{args.model_name + '_' + args.save_name}")

    print("Done :)\n")
