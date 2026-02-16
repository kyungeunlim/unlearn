"""
Sequential back-to-front unlearning.

For each layer L (from last to first):
1. Forget loss: hidden states at layer L from current model →
   original model's layers L→N → lm_head → maximize entropy
2. Retain loss: normal forward pass through current model → match original

This ensures each layer is verified unlearned with clean inputs from earlier layers.
"""

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import seed_worker

from unlearn.utils.hook import ActivationCapture, resolve_layer_names
from unlearn.utils.keyword_masks import apply_keyword_masks
from unlearn.utils.math import max_entropy_kl_loss, top_k_entropy_loss
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer, save_checkpoint


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
        retain_layers=None,
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

        # Resolve hook layer for forget path (output of target_layer)
        hook_layer_indices = [target_layer]
        self.layer_id_to_name = resolve_layer_names(model, hook_layer_indices)
        self.target_module_names = list(self.layer_id_to_name.values())

        # Resolve layers for retain L2 loss
        self.retain_layers = retain_layers or []
        if self.retain_layers:
            self.retain_layer_names = resolve_layer_names(model, self.retain_layers)
            self.retain_target_modules = list(self.retain_layer_names.values())
            self.orig_retain_layer_names = resolve_layer_names(
                original_model, self.retain_layers
            )
            self.orig_retain_target_modules = list(
                self.orig_retain_layer_names.values()
            )

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

        capturer = ActivationCapture(unwrapped_model, self.target_module_names)

        # === Retain loss: keep full model outputs close to original ===
        if retain_coeff > 0:
            if self.run_args.retain_loss_type == "kl":
                # KL divergence on output logits (no hidden states needed)
                with torch.no_grad():
                    orig_outputs = self.original_model(
                        input_ids=retain_input_ids,
                        attention_mask=retain_attention_mask,
                    )
                    orig_logits = orig_outputs.logits.detach()

                current_outputs = unwrapped_model(
                    input_ids=retain_input_ids,
                    attention_mask=retain_attention_mask,
                )
                current_logits = current_outputs.logits

                mask = retain_attention_mask.bool()
                orig_probs = F.softmax(orig_logits[mask].float(), dim=-1)
                current_log_probs = F.log_softmax(current_logits[mask].float(), dim=-1)
                retain_loss = F.kl_div(
                    current_log_probs, orig_probs, reduction="batchmean"
                )
            else:
                # L2 on hidden states at targeted layers
                retain_mask = retain_attention_mask.unsqueeze(-1)

                orig_capturer = ActivationCapture(
                    self.original_model, self.orig_retain_target_modules
                )
                orig_capturer.register()
                with torch.no_grad():
                    self.original_model(
                        input_ids=retain_input_ids,
                        attention_mask=retain_attention_mask,
                    )
                orig_retain_acts = {}
                for l in self.retain_layers:
                    name = self.orig_retain_layer_names[l]
                    orig_retain_acts[l] = (
                        orig_capturer.activations[name].detach() * retain_mask
                    )
                orig_capturer.remove()

                current_capturer = ActivationCapture(
                    unwrapped_model, self.retain_target_modules
                )
                current_capturer.register()
                unwrapped_model(
                    input_ids=retain_input_ids,
                    attention_mask=retain_attention_mask,
                )

                n_layers = len(self.retain_layers)
                retain_loss = torch.tensor(0.0, device=device)
                for l in self.retain_layers:
                    name = self.retain_layer_names[l]
                    lora_h = current_capturer.activations[name] * retain_mask
                    retain_loss = (
                        retain_loss
                        + torch.norm(
                            lora_h - orig_retain_acts[l],
                            dim=-1,
                            p=2,
                            dtype=torch.float,
                        ).nanmean()
                    )
                retain_loss = retain_loss / n_layers
                current_capturer.remove()
        else:
            retain_loss = torch.tensor(0.0, device=device)

        # === Forget loss: maximize entropy via original model's later layers ===
        if forget_coeff > 0:
            # Hook on target_layer captures its output; gradient flows through it
            capturer.register()
            unwrapped_model(
                input_ids=forget_input_ids,
                attention_mask=forget_attention_mask,
            )
            hidden_at_layer = capturer.activations[
                self.layer_id_to_name[self.target_layer]
            ]
            capturer.remove()

            orig_dtype = next(self.original_model.parameters()).dtype
            logits = forward_from_layer(
                self.original_model,
                hidden_at_layer.to(orig_dtype),
                self.target_layer + 1,
            )

            vocab_size = logits.shape[-1]
            log_vocab = torch.log(torch.tensor(float(vocab_size), device=device))

            mask = forget_attention_mask.bool()
            logits_masked = logits[mask]

            if logits_masked.numel() > 0:
                if self.run_args.use_top_k_entropy:
                    k = self.run_args.top_k
                    log_k = torch.log(torch.tensor(float(k), device=device))
                    forget_loss = top_k_entropy_loss(logits_masked, k=k) / log_k
                elif self.run_args.use_max_entropy_kl:
                    forget_loss = max_entropy_kl_loss(logits_masked) / log_vocab
                else:
                    batch_size, seq_len = logits.shape[:2]
                    random_targets = torch.randint(
                        0, vocab_size, (batch_size, seq_len), device=device
                    )
                    targets_flat = random_targets[mask]
                    ce_loss = F.cross_entropy(
                        logits_masked.float(), targets_flat, reduction="mean"
                    )
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


@dataclass
class SequentialUnlearnConfig:
    num_train_examples: int = 1024
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: str = "rewritten"
    lr: float = 1e-3
    pdbs: int = 4
    retain_coef: float = 5.0
    remove_coef: float = 10.0
    retain_loss_type: Literal["l2", "kl"] = "l2"
    retain_layers: list[int] = field(default_factory=lambda: list(range(32)))
    lora_r: int = 16
    start_layer: int | None = None
    end_layer: int = 8
    layer_step: int = 4
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    revision: str = "main"
    epochs_per_layer: int = 1
    use_max_entropy_kl: bool = False
    use_top_k_entropy: bool = False
    top_k: int = 100
    use_ultrachat: bool = True


def main():
    parser = ArgumentParser()
    parser.add_arguments(SequentialUnlearnConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    assert torch.cuda.is_available(), "CUDA required"
    NUM_PROC = (os.cpu_count() or 32) // 2

    print("=" * 60)
    print("Sequential Back-to-Front Unlearning")
    print("=" * 60)
    for arg, value in vars(run_cfg).items():
        print(f"  {arg}: {value}")
    print("=" * 60)

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision
    )

    # Create a frozen copy for the original model
    print("Creating frozen copy of original model...")
    original_model = deepcopy(model)
    for param in original_model.parameters():
        param.requires_grad = False
    original_model.eval()

    # Determine layers to unlearn
    num_layers = model.config.num_hidden_layers
    if run_cfg.start_layer is None:
        run_cfg.start_layer = num_layers - 1

    layers_to_unlearn = list(
        range(run_cfg.start_layer, run_cfg.end_layer - 1, -run_cfg.layer_step)
    )
    print(f"Layers to unlearn (back-to-front): {layers_to_unlearn}")

    # Setup LoRA
    print(f"\nUsing LoRA with rank {run_cfg.lora_r}")
    lora_config = LoraConfig(
        r=run_cfg.lora_r,
        lora_alpha=run_cfg.lora_r,
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
    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)
    train_dataset.tokenized_bio_remove_dataset = apply_keyword_masks(
        train_dataset.tokenized_bio_remove_dataset, run_cfg, tokenizer
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = 32
    grad_acc_steps = max(1, global_batch_size // (run_cfg.pdbs * world_size))

    print(
        f"\nRunning with {world_size} GPUs. Per device batch: "
        f"{run_cfg.pdbs}. Grad Acc: {grad_acc_steps}."
    )

    # Sequential unlearning: back to front
    for layer_idx in layers_to_unlearn:
        print(f"\n{'=' * 60}")
        print(f"Unlearning Layer {layer_idx}")
        print("=" * 60)

        training_args = TrainingArguments(
            output_dir=f"./results/layer_{layer_idx}",
            learning_rate=run_cfg.lr,
            gradient_accumulation_steps=grad_acc_steps,
            per_device_train_batch_size=run_cfg.pdbs,
            per_device_eval_batch_size=run_cfg.pdbs,
            num_train_epochs=run_cfg.epochs_per_layer,
            weight_decay=0.01,
            gradient_checkpointing=True,
            bf16=True,
            max_grad_norm=1.0,
            save_strategy="no",
            ddp_find_unused_parameters=False,
        )

        trainer = SequentialUnlearningTrainer(
            run_args=run_cfg,
            model=model,
            original_model=original_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            target_layer=layer_idx,
            retain_layers=run_cfg.retain_layers,
        )

        model.train()
        trainer.train()

        print(f"Completed layer {layer_idx}")

    # Merge LoRA and save
    print("\nMerging LoRA weights...")
    model = model.merge_and_unload()

    if run_cfg.save_path:
        trainer.model = model
        save_checkpoint(trainer, run_cfg.save_path, tokenizer)

    print("\nTraining complete")


if __name__ == "__main__":
    main()
