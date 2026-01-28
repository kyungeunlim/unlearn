"""
Sequential back-to-front unlearning.

For each layer L (from last to first):
1. Forget loss: hidden states at layer L from current model →
   original model's layers L→N → lm_head → maximize entropy
2. Retain loss: normal forward pass through current model → match original

This ensures each layer is verified unlearned with clean inputs from earlier layers.
"""

import argparse
import os
from copy import deepcopy
from typing import cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import seed_worker

from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer


def get_model_components(model):
    """Extract layers, final_norm, and lm_head from various model architectures."""
    unwrapped = unwrap_model(model)

    # GPT-NeoX (Pythia, etc.)
    if hasattr(unwrapped, "gpt_neox"):
        return (
            unwrapped.gpt_neox.layers,
            unwrapped.gpt_neox.final_layer_norm,
            unwrapped.embed_out,
        )
    # LLaMA-style
    elif hasattr(unwrapped, "model") and hasattr(unwrapped.model, "layers"):
        return (
            unwrapped.model.layers,
            unwrapped.model.norm,
            unwrapped.lm_head,
        )
    # GPT-2 style
    elif hasattr(unwrapped, "transformer") and hasattr(unwrapped.transformer, "h"):
        return (
            unwrapped.transformer.h,
            unwrapped.transformer.ln_f,
            unwrapped.lm_head,
        )
    else:
        raise ValueError(f"Unknown model architecture: {type(unwrapped)}")


def forward_from_layer(model, hidden_states, start_layer, position_ids=None):
    """
    Pass hidden states through model's layers starting from start_layer.

    Args:
        model: The model to use (frozen original)
        hidden_states: [batch, seq, hidden_dim] - output of layer (start_layer - 1)
        start_layer: Which layer to start from (0-indexed)
        position_ids: [batch, seq] position IDs for rotary embeddings

    Returns:
        logits: [batch, seq, vocab_size]
    """
    unwrapped = unwrap_model(model)
    layers, final_norm, lm_head = get_model_components(model)

    hidden = hidden_states
    batch_size, seq_len = hidden.shape[:2]
    device = hidden.device

    # Create position_ids if not provided
    if position_ids is None:
        position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

    # Get rotary embeddings if available (GPT-NeoX style)
    position_embeddings = None
    if hasattr(unwrapped, "gpt_neox") and hasattr(unwrapped.gpt_neox, "rotary_emb"):
        rotary_emb = unwrapped.gpt_neox.rotary_emb
        position_embeddings = rotary_emb(hidden, position_ids)

    # Pass through remaining layers
    # attention_mask=None lets model handle causal masking
    for layer_idx in range(start_layer, len(layers)):
        layer = layers[layer_idx]
        if position_embeddings is not None:
            layer_out = layer(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        else:
            layer_out = layer(hidden, attention_mask=None)
        hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    # Final norm and lm_head
    hidden = final_norm(hidden)
    logits = lm_head(hidden)

    return logits


class SequentialUnlearningTrainer(Trainer):
    """Trainer for sequential back-to-front unlearning."""

    def __init__(
        self,
        run_args,
        model,
        original_model,
        args,
        train_dataset,
        tokenizer,
        target_layer,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        self.run_args = run_args
        self.original_model = original_model
        self.target_layer = target_layer
        self.tokenizer = tokenizer
        self.current_training_step = 0

        # Freeze original model
        for param in self.original_model.parameters():
            param.requires_grad = False
        self.original_model.eval()

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(
            DataLoader(self.train_dataset, **dataloader_params)
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        unwrapped_model = unwrap_model(model)
        device = next(unwrapped_model.parameters()).device

        # === Retain data ===
        retain_input_ids = inputs["input_ids"].to(device)
        retain_attention_mask = inputs["attention_mask"].to(device)

        # === Forget data ===
        forget_input_ids = inputs["bio_remove_input_ids"].to(device)
        forget_attention_mask = inputs["bio_remove_attention_mask"].to(device)

        # Coefficient scheduling
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        progress = min(
            1.0,
            self.current_training_step
            / (self.run_args.num_train_examples / (self.run_args.pdbs * world_size)),
        )
        retain_coeff = self.run_args.retain_coef * progress
        forget_coeff = self.run_args.remove_coef * (1 - 0.25 * progress)

        # === Retain loss: keep full model outputs close to original ===
        if retain_coeff > 0:
            with torch.no_grad():
                orig_outputs = self.original_model(
                    input_ids=retain_input_ids,
                    attention_mask=retain_attention_mask,
                    output_hidden_states=True,
                )
                orig_hidden = torch.stack(orig_outputs.hidden_states).detach()

            current_outputs = unwrapped_model(
                input_ids=retain_input_ids,
                attention_mask=retain_attention_mask,
                output_hidden_states=True,
            )
            current_hidden = torch.stack(current_outputs.hidden_states)

            mask = retain_attention_mask.unsqueeze(0).unsqueeze(-1)
            retain_loss = torch.norm(
                (current_hidden - orig_hidden) * mask, dim=-1, p=2, dtype=torch.float
            ).nanmean()
        else:
            retain_loss = torch.tensor(0.0, device=device)

        # === Forget loss: maximize entropy via original model's later layers ===
        if forget_coeff > 0:
            # Get hidden states at target layer from current model
            current_outputs = unwrapped_model(
                input_ids=forget_input_ids,
                attention_mask=forget_attention_mask,
                output_hidden_states=True,
            )
            # hidden_states[L] is output of layer L-1 / input to layer L
            # To process through layers L onwards, we want hidden_states[L]
            hidden_at_layer = current_outputs.hidden_states[self.target_layer]

            # Pass our hidden states through original model's remaining layers
            # We need gradients to flow back through hidden_at_layer
            # Keep dtype consistent with original model (typically bfloat16)
            orig_dtype = next(self.original_model.parameters()).dtype
            logits = forward_from_layer(
                self.original_model,
                hidden_at_layer.to(orig_dtype),
                self.target_layer,
            )

            # Maximize entropy = minimize cross-entropy with random targets
            vocab_size = logits.shape[-1]
            batch_size, seq_len = logits.shape[:2]

            random_targets = torch.randint(
                0, vocab_size, (batch_size, seq_len), device=device
            )

            mask = forget_attention_mask.bool()
            logits_flat = logits[mask]
            targets_flat = random_targets[mask]

            if logits_flat.numel() > 0:
                ce_loss = F.cross_entropy(
                    logits_flat.float(), targets_flat, reduction="mean"
                )
                log_vocab = torch.log(torch.tensor(float(vocab_size), device=device))
                forget_loss = ce_loss / log_vocab
            else:
                forget_loss = torch.tensor(0.0, device=device)
        else:
            forget_loss = torch.tensor(0.0, device=device)

        loss = retain_coeff * retain_loss + forget_coeff * forget_loss

        if (
            self.current_training_step % 32 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            print(
                f"step {self.current_training_step} | "
                f"layer {self.target_layer} | "
                f"retain_coeff: {retain_coeff:.4f} | "
                f"forget_coeff: {forget_coeff:.4f} | "
                f"retain_loss: {retain_loss.item():.4f} | "
                f"forget_loss: {forget_loss.item():.4f}"
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_examples", type=int, default=1024)
    parser.add_argument("--unlearn_corrupt", type=bool, default=False)
    parser.add_argument("--corrupt_ratio", type=float, default=0.5)
    parser.add_argument("--corrupt_ds", type=str, default="rewritten")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pdbs", type=int, default=4)
    parser.add_argument("--retain_coef", type=float, default=5.0)
    parser.add_argument("--remove_coef", type=float, default=10.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument(
        "--start_layer",
        type=int,
        default=None,
        help="Layer to start unlearning from (default: last layer)",
    )
    parser.add_argument(
        "--end_layer",
        type=int,
        default=8,
        help="Layer to stop unlearning at (inclusive)",
    )
    parser.add_argument(
        "--layer_step",
        type=int,
        default=4,
        help="Step size between layers (e.g., 4 means unlearn every 4th layer)",
    )
    parser.add_argument(
        "--model_name", type=str, default="EleutherAI/deep-ignorance-unfiltered"
    )
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--epochs_per_layer", type=int, default=1)

    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    NUM_PROC = (os.cpu_count() or 32) // 2

    print("=" * 60)
    print("Sequential Back-to-Front Unlearning")
    print("=" * 60)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_name, revision=args.revision)

    # Create a frozen copy for the original model
    print("Creating frozen copy of original model...")
    original_model = deepcopy(model)
    for param in original_model.parameters():
        param.requires_grad = False
    original_model.eval()

    # Determine layers to unlearn
    num_layers = model.config.num_hidden_layers
    if args.start_layer is None:
        args.start_layer = num_layers - 1

    layers_to_unlearn = list(
        range(args.start_layer, args.end_layer - 1, -args.layer_step)
    )
    print(f"Layers to unlearn (back-to-front): {layers_to_unlearn}")

    # Setup LoRA
    print(f"\nUsing LoRA with rank {args.lora_r}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r,
        target_modules=None,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model = cast(PreTrainedModel, model)
    model.enable_input_require_grads()

    # Load dataset
    train_dataset = get_unlearning_dataset(args, tokenizer, NUM_PROC)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = 32
    grad_acc_steps = max(1, global_batch_size // (args.pdbs * world_size))

    print(
        f"\nRunning with {world_size} GPUs. Per device batch: "
        f"{args.pdbs}. Grad Acc: {grad_acc_steps}."
    )

    # Sequential unlearning: back to front
    for layer_idx in layers_to_unlearn:
        print(f"\n{'='*60}")
        print(f"Unlearning Layer {layer_idx}")
        print("=" * 60)

        training_args = TrainingArguments(
            output_dir=f"./results/layer_{layer_idx}",
            learning_rate=args.lr,
            gradient_accumulation_steps=grad_acc_steps,
            per_device_train_batch_size=args.pdbs,
            per_device_eval_batch_size=args.pdbs,
            num_train_epochs=args.epochs_per_layer,
            weight_decay=0.01,
            gradient_checkpointing=True,
            bf16=True,
            max_grad_norm=1.0,
            save_strategy="no",
            ddp_find_unused_parameters=False,
        )

        trainer = SequentialUnlearningTrainer(
            run_args=args,
            model=model,
            original_model=original_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            target_layer=layer_idx,
        )

        model.train()
        trainer.train()

        print(f"Completed layer {layer_idx}")

    # Merge LoRA and save
    print("\nMerging LoRA weights...")
    model = model.merge_and_unload()

    if args.save_name:
        save_path = f"./models/{args.model_name.replace('/', '_')}_{args.save_name}"
        print(f"Saving model to: {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
