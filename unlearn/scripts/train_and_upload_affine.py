#!/usr/bin/env python
"""
Standalone script to train affine transforms and upload to HuggingFace.
This can be used without running the full unlearning pipeline.
"""
import argparse

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearn.probe import (
    evaluate_affine_mse,
    save_affine_transforms,
    train_affine_transform,
    upload_affine_transforms_to_hub,
)
from unlearn.reference.cas.utils import (
    BIO_RETAIN_DS_NAME,
    RETAIN_TEXT_DS_NAME,
    cb_retain_tokenize_function,
    wikitext_tokenize_function,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train affine transforms between two model " "checkpoints and upload to HF"
        )
    )
    parser.add_argument(
        "--source_model",
        type=str,
        default="EleutherAI/deep-ignorance-pretraining-stage-unfiltered",
        help="Source model (checkpoint) to map FROM",
    )
    parser.add_argument(
        "--source_revision",
        type=str,
        default="global_step38144",
        help="Revision of source model",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="EleutherAI/deep-ignorance-unfiltered",
        help="Target model to map TO",
    )
    parser.add_argument(
        "--target_revision",
        type=str,
        default="main",
        help="Revision of target model",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30],
        help="Layers to train affine transforms for",
    )
    parser.add_argument(
        "--num_train_examples",
        type=int,
        default=100000,
        help="Number of examples for training",
    )
    parser.add_argument(
        "--num_eval_examples",
        type=int,
        default=10000,
        help="Number of examples for MSE evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training/evaluation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Ridge regression alpha",
    )
    parser.add_argument(
        "--use_bio_retain",
        action="store_true",
        help="Include bio-retain corpus in training data",
    )
    parser.add_argument(
        "--upload_to_hub",
        type=str,
        default=None,
        help="HuggingFace repo to upload to (e.g., 'username/affine-map')",
    )
    parser.add_argument(
        "--save_local",
        type=str,
        default=None,
        help="Local directory to save affine transforms",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make HuggingFace repo private",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Max sequence length for tokenization",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Affine Transform Training")
    print("=" * 60)
    print(f"Source: {args.source_model} @ {args.source_revision}")
    print(f"Target: {args.target_model} @ {args.target_revision}")
    print(f"Layers: {args.layers}")
    print(f"Training examples: {args.num_train_examples}")
    print(f"Evaluation examples: {args.num_eval_examples}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    print(f"Loading source model: {args.source_model}...")
    source_model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        revision=args.source_revision,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    source_model.eval()

    print(f"Loading target model: {args.target_model}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        revision=args.target_revision,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    target_model.eval()

    # Load dataset (using project's standard datasets)
    print("Loading wikitext dataset...")
    retain_text_dataset = load_dataset(RETAIN_TEXT_DS_NAME, "wikitext-103-raw-v1")[
        "train"
    ]
    retain_text_dataset = retain_text_dataset.rename_column("page", "text")
    retain_text_dataset = retain_text_dataset.shuffle(seed=42).select(
        range(
            min(
                args.num_train_examples + args.num_eval_examples,
                len(retain_text_dataset),
            )
        )
    )

    print("Tokenizing wikitext...")
    dataset = retain_text_dataset.map(
        lambda x: wikitext_tokenize_function(x, tokenizer),
        batched=True,
        num_proc=4,
    )

    if args.use_bio_retain:
        print("Loading bio-retain corpus...")
        bio_retain_dataset = load_dataset(BIO_RETAIN_DS_NAME, "bio-retain-corpus")[
            "train"
        ]
        bio_retain_dataset = bio_retain_dataset.shuffle(seed=42).select(
            range(min(args.num_train_examples // 4, len(bio_retain_dataset)))
        )
        tokenized_bio_retain = bio_retain_dataset.map(
            lambda x: cb_retain_tokenize_function(x, tokenizer),
            batched=True,
            num_proc=4,
        )
        dataset = concatenate_datasets([dataset, tokenized_bio_retain]).shuffle(seed=42)

    # Train affine transforms
    print("\nTraining affine transforms...")
    affine_transforms = train_affine_transform(
        source_model=source_model,
        target_model=target_model,
        dataset=dataset,
        target_layers=args.layers,
        num_examples=args.num_train_examples,
        batch_size=args.batch_size,
        device=device,
        alpha=args.alpha,
    )

    # Move to device
    for idx, transform in affine_transforms.items():
        affine_transforms[idx] = transform.to(device)

    # Evaluate MSE
    print("\nEvaluating MSE...")
    mse_metrics = evaluate_affine_mse(
        affine_transforms=affine_transforms,
        source_model=source_model,
        target_model=target_model,
        dataset=dataset,
        target_layers=args.layers,
        num_examples=args.num_eval_examples,
        batch_size=args.batch_size,
        device=device,
    )

    print("\nMSE Results:")
    print("-" * 40)
    for layer_idx in args.layers:
        print(f"  Layer {layer_idx}: {mse_metrics[f'layer_{layer_idx}']:.6f}")
    print("-" * 40)
    print(f"  Mean MSE: {mse_metrics['mean_mse']:.6f}")

    # Save locally if requested
    if args.save_local:
        print(f"\nSaving to {args.save_local}...")
        save_affine_transforms(affine_transforms, args.save_local, alpha=args.alpha)

    # Upload to HuggingFace if requested
    if args.upload_to_hub:
        print(f"\nUploading to HuggingFace: {args.upload_to_hub}...")
        upload_affine_transforms_to_hub(
            affine_transforms=affine_transforms,
            repo_id=args.upload_to_hub,
            mse_metrics=mse_metrics,
            source_model_name=f"{args.source_model}@{args.source_revision}",
            target_model_name=f"{args.target_model}@{args.target_revision}",
            alpha=args.alpha,
            num_training_examples=args.num_train_examples,
            private=args.private,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
