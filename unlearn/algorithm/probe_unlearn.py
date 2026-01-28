"""Unlearning via entropy maximization using transformer probes.

Similar to tuned lens unlearning, but uses depth-scaled transformer probes
to map hidden states to predicted activations, then maximizes entropy.
"""

import argparse
import os
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import seed_worker

from unlearn.probe import TransformerProbe, load_transformer_probe
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer


class ProbeUnlearningTrainer(Trainer):
    def __init__(
        self,
        run_args,
        model,
        args,
        train_dataset,
        tokenizer,
        target_layers,
        probes,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        self.run_args = run_args
        self.num_training_steps = self.args.max_steps
        self.current_training_step = 0
        self.tokenizer = tokenizer
        self.target_layers = target_layers
        self.model = model
        self.retain_coef = self.run_args.retain_coef
        self.remove_coef = self.run_args.remove_coef
        self.trainer_tokenizer = tokenizer
        self.probes = probes  # Dict[layer_idx, TransformerProbe]
        self.is_peft_model = run_args.lora

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        unwrapped_model = unwrap_model(model)

        target_device = (
            inputs["input_ids"].device
            if hasattr(inputs["input_ids"], "device")
            else unwrapped_model.device
        )

        # === retain ===
        retain_input_ids = inputs.get("input_ids").to(target_device)
        retain_attention_mask = inputs.get("attention_mask").to(target_device)
        # ==== forget (circuit breaker) ====
        cb_input_ids = inputs.get("bio_remove_input_ids").to(target_device)
        cb_attention_mask = inputs.get("bio_remove_attention_mask").to(target_device)

        # ==== Forward Inputs ====
        retain_inputs_dict = dict(
            input_ids=retain_input_ids,
            attention_mask=retain_attention_mask,
            output_hidden_states=True,
        )
        cb_inputs_dict = dict(
            input_ids=cb_input_ids,
            attention_mask=cb_attention_mask,
            output_hidden_states=True,
        )

        # ===== Coefficient scheduling ====
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        scheduled_coeff = min(
            1.0,
            self.current_training_step
            / (self.run_args.num_train_examples / (self.run_args.pdbs * world_size)),
        )
        retain_coeff = self.retain_coef * scheduled_coeff
        circuit_breaker_coeff = self.remove_coef * (1 - 0.25 * scheduled_coeff)

        broadcast_retain_mask = retain_attention_mask.unsqueeze(0).unsqueeze(-1)

        # Get reference hidden states for retain loss
        if self.is_peft_model:
            with unwrapped_model.disable_adapter():
                unwrapped_model.eval()
                with torch.no_grad():
                    if retain_coeff > 0:
                        orig_retain_outputs = unwrapped_model(**retain_inputs_dict)["hidden_states"]
                        orig_retain_hidden = torch.stack(orig_retain_outputs).detach()
                        orig_retain_hidden *= broadcast_retain_mask
                        del orig_retain_outputs
        else:
            unwrapped_model.eval()
            with torch.no_grad():
                if retain_coeff > 0:
                    orig_retain_outputs = unwrapped_model(**retain_inputs_dict)["hidden_states"]
                    orig_retain_hidden = torch.stack(orig_retain_outputs).detach()
                    orig_retain_hidden *= broadcast_retain_mask
                    del orig_retain_outputs

        unwrapped_model.train()

        # Retain loss: keep hidden states close to original
        if retain_coeff > 0:
            lora_retain_outputs = unwrapped_model(**retain_inputs_dict)["hidden_states"]
            lora_retain_hidden = torch.stack(lora_retain_outputs) * broadcast_retain_mask
            retain_loss = torch.norm(
                lora_retain_hidden - orig_retain_hidden, dim=-1, p=2, dtype=torch.float
            ).nanmean()
        else:
            retain_loss = torch.tensor(0.0, device=target_device)

        # Forget loss: entropy maximization via transformer probes
        if circuit_breaker_coeff > 0:
            cb_outputs = unwrapped_model(**cb_inputs_dict)["hidden_states"]

            layer_losses = []
            vocab_size = unwrapped_model.config.vocab_size

            for layer_idx in self.target_layers:
                if layer_idx not in self.probes:
                    continue

                probe = self.probes[layer_idx]
                hidden = cb_outputs[layer_idx]  # [batch, seq, hidden]

                # Pass through probe to get transformed activations
                # Note: probe weights are frozen (requires_grad=False) but we need
                # gradients to flow through hidden states back to the model
                probe_out = probe(
                    hidden.float(),
                    cb_attention_mask.bool()
                )  # [batch, seq, hidden]

                # Pass probe output through remaining layers to get logits
                # We'll use the model's lm_head directly on the probe output
                # This is an approximation - ideally we'd run through remaining layers
                # but this is more memory efficient and the probe already captures
                # the transformation to final layer representation
                probe_out_bf16 = probe_out.to(dtype=torch.bfloat16)

                # Get logits from lm_head
                if hasattr(unwrapped_model, "lm_head"):
                    logits = unwrapped_model.lm_head(probe_out_bf16)
                elif hasattr(unwrapped_model, "embed_out"):
                    logits = unwrapped_model.embed_out(probe_out_bf16)
                else:
                    # Try to find the output projection
                    logits = unwrapped_model.get_output_embeddings()(probe_out_bf16)

                batch_size, seq_len, _ = logits.shape

                # Random target tokens for entropy maximization
                random_targets = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=logits.device
                )

                # Compute CE loss only on non-padded tokens
                mask = cb_attention_mask.bool()
                logits_flat = logits[mask]
                targets_flat = random_targets[mask]

                if logits_flat.numel() > 0:
                    ce_loss = F.cross_entropy(
                        logits_flat.float(), targets_flat, reduction="mean"
                    )
                    layer_losses.append(ce_loss)

            log_vocab = torch.log(torch.tensor(float(vocab_size), device=target_device))
            if layer_losses:
                mean_ce = torch.stack(layer_losses).mean()
                circuit_breaker_loss = mean_ce / log_vocab
                mean_entropy = -mean_ce
            else:
                circuit_breaker_loss = torch.tensor(0.0, device=target_device)
                mean_entropy = torch.tensor(0.0)
        else:
            circuit_breaker_loss = torch.tensor(0.0, device=target_device)
            mean_entropy = torch.tensor(0.0)

        loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss

        if (
            self.current_training_step % 32 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            entropy_val = mean_entropy.item() if isinstance(mean_entropy, torch.Tensor) else mean_entropy
            retain_val = retain_loss.item() if isinstance(retain_loss, torch.Tensor) else retain_loss
            cb_val = circuit_breaker_loss.item() if isinstance(circuit_breaker_loss, torch.Tensor) else circuit_breaker_loss
            print(
                f"step {self.current_training_step} | "
                f"retain_coeff: {retain_coeff:.4f} | "
                f"forget_coeff: {circuit_breaker_coeff:.4f} | "
                f"retain_loss: {retain_val:.4f} | "
                f"forget_loss: {cb_val:.4f} | "
                f"entropy: {entropy_val:.4f}"
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


def load_probes(probe_dir: str, layers: list[int], device: str) -> dict[int, TransformerProbe]:
    """Load transformer probes for specified layers."""
    probe_dir = Path(probe_dir)
    probes = {}

    for layer_idx in layers:
        config_path = probe_dir / f"layer_{layer_idx}_config.json"
        weights_path = probe_dir / f"layer_{layer_idx}.safetensors"

        if not config_path.exists() or not weights_path.exists():
            print(f"Warning: Probe for layer {layer_idx} not found at {probe_dir}")
            continue

        probe = load_transformer_probe(probe_dir, layer_idx, device=device)
        probe.eval()
        for param in probe.parameters():
            param.requires_grad = False
        probes[layer_idx] = probe
        print(f"Loaded probe for layer {layer_idx} ({sum(p.numel() for p in probe.parameters()):,} params)")

    return probes


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_examples", type=int, default=1024)
    parser.add_argument("--unlearn_corrupt", type=bool, default=False)
    parser.add_argument("--corrupt_ratio", type=float, default=0.5)
    parser.add_argument(
        "--corrupt_ds", type=str, default="rewritten", choices=["rewritten", "shuffled"]
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pdbs", type=int, default=4)
    parser.add_argument("--retain_coef", type=float, default=5.0)
    parser.add_argument("--remove_coef", type=float, default=5.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora", action="store_true", help="Use LoRA (default: full SFT)")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20, 24, 28],
        help="List of layers to target (must have probes trained)",
    )
    parser.add_argument(
        "--model_name", type=str, default="EleutherAI/deep-ignorance-unfiltered"
    )
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument(
        "--probe_dir", type=str, required=True, help="Path to trained transformer probes"
    )
    parser.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args()

    print("=" * 60)
    print("Probe-based Unlearning")
    print("=" * 60)
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 60)

    model, tokenizer = get_model_and_tokenizer(args.model_name, revision=args.revision)
    train_dataset = get_unlearning_dataset(args, tokenizer, NUM_PROC)

    device = next(model.parameters()).device

    # Load frozen transformer probes
    print(f"\nLoading probes from: {args.probe_dir}")
    probes = load_probes(args.probe_dir, args.layers, device=str(device))

    if not probes:
        raise ValueError(f"No probes found at {args.probe_dir} for layers {args.layers}")

    # Filter layers to only those with available probes
    available_layers = list(probes.keys())
    print(f"Available probe layers: {available_layers}")

    lora_layers_to_transform = [i for i in range(max(available_layers) + 1)]

    if args.lora:
        print(f"\nUsing LoRA with rank {args.lora_r}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,
            target_modules=None,  # Auto-detect
            lora_dropout=0.05,
            bias="none",
            layers_to_transform=lora_layers_to_transform,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("\nUsing full SFT (no LoRA)")
        print("WARNING: Retain loss requires a frozen reference model. Setting retain_coef=0 for SFT.")
        args.retain_coef = 0.0
        for param in model.parameters():
            param.requires_grad = True

    model = cast(PreTrainedModel, model)
    model.enable_input_require_grads()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = 32
    grad_acc_steps = max(1, global_batch_size // (args.pdbs * world_size))

    print(
        f"\nRunning with {world_size} GPUs. Per device batch: {args.pdbs}. "
        f"Grad Acc steps: {grad_acc_steps}."
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=args.lr,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=args.pdbs,
        per_device_eval_batch_size=args.pdbs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        gradient_checkpointing=True,
        bf16=True,
        max_grad_norm=1.0,
        save_strategy="no",
        ddp_find_unused_parameters=False,
    )

    trainer = ProbeUnlearningTrainer(
        args,
        model,
        training_args,
        train_dataset,
        tokenizer,
        available_layers,
        probes,
    )

    model.train()
    trainer.train()

    if args.lora:
        model = model.merge_and_unload()

    if args.save_name:
        mode_str = f"lora{args.lora_r}" if args.lora else "sft"
        save_path = f"./models/{args.model_name.replace('/', '_')}_{args.save_name}_{mode_str}"
        print(f"\nSaving model to: {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    print("\nDone!")
