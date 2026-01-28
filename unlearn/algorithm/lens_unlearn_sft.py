"""Unlearning via entropy maximization at frozen tuned lens layers (SFT version)."""

import argparse
import os
from typing import cast

import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import seed_worker
from tuned_lens import TunedLens

from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer, unwrap_model


class UnlearningTrainer(Trainer):
    def __init__(
        self,
        run_args,
        model,
        args,
        train_dataset,
        tokenizer,
        target_layers,
        lens=None,
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
        self.lens = lens

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


class SFTUnlearningTrainer(UnlearningTrainer):

    def __init__(self, *args, frozen_ref_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_ref_model = frozen_ref_model

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
        # ==== forget ====
        forget_input_ids = inputs.get("bio_remove_input_ids").to(target_device)
        forget_attention_mask = inputs.get("bio_remove_attention_mask").to(
            target_device
        )

        # ==== Forward Inputs ====
        module = "hidden_states"
        retain_inputs_dict = dict(
            input_ids=retain_input_ids,
            attention_mask=retain_attention_mask,
            output_hidden_states=True,
        )
        forget_inputs_dict = dict(
            input_ids=forget_input_ids,
            attention_mask=forget_attention_mask,
            output_hidden_states=True,
        )

        # ===== Step Coeff ====
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        scheduled_coeff = min(
            [
                1.0,
                self.current_training_step
                / (
                    self.run_args.num_train_examples / (self.run_args.pdbs * world_size)
                ),
            ]
        )
        retain_coeff = self.retain_coef * scheduled_coeff
        forget_coeff = self.remove_coef * (1 - 0.25 * scheduled_coeff)

        broadcast_retain_mask = retain_attention_mask.unsqueeze(0).unsqueeze(-1)

        # Get reference hidden states from frozen CPU model
        if self.frozen_ref_model is not None and retain_coeff > 0:
            with torch.no_grad():
                cpu_inputs = {
                    "input_ids": retain_input_ids.cpu(),
                    "attention_mask": retain_attention_mask.cpu(),
                    "output_hidden_states": True,
                }
                orig_retain_outputs = self.frozen_ref_model(**cpu_inputs)[module]
                orig_retain_hidden = (
                    torch.stack(orig_retain_outputs).detach().to(target_device)
                )
                orig_retain_hidden *= broadcast_retain_mask
                del orig_retain_outputs

        unwrapped_model.train()

        ### Retain loss
        if retain_coeff > 0 and self.frozen_ref_model is not None:
            current_retain_outputs = unwrapped_model(**retain_inputs_dict)[module]
            current_retain_hidden = (
                torch.stack(current_retain_outputs) * broadcast_retain_mask
            )
            retain_loss = torch.norm(
                current_retain_hidden - orig_retain_hidden,
                dim=-1,
                p=2,
                dtype=torch.float,
            ).nanmean()
        else:
            retain_loss = 0

        ### Forget loss - entropy maximization via tuned lens
        if forget_coeff > 0:
            forget_outputs = unwrapped_model(**forget_inputs_dict)[module]

            layer_losses = []
            lens_device = next(self.lens.parameters()).device
            for layer_idx in self.target_layers:
                hidden = forget_outputs[layer_idx]
                hidden_bf16 = hidden.to(device=lens_device, dtype=torch.bfloat16)

                lens_logits = self.lens(hidden_bf16, idx=layer_idx)
                batch_size, seq_len, vocab_size = lens_logits.shape

                random_targets = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=lens_logits.device
                )

                mask = forget_attention_mask.bool()
                logits_flat = lens_logits[mask]
                targets_flat = random_targets[mask]

                if logits_flat.numel() > 0:
                    ce_loss = F.cross_entropy(
                        logits_flat.float(), targets_flat, reduction="mean"
                    )
                    layer_losses.append(ce_loss)

            log_vocab = torch.log(torch.tensor(float(vocab_size), device=target_device))
            if layer_losses:
                mean_ce = torch.stack(layer_losses).mean()
                forget_loss = mean_ce / log_vocab
                mean_entropy = -mean_ce
            else:
                forget_loss = torch.tensor(0.0, device=target_device)
                mean_entropy = torch.tensor(0.0)
        else:
            forget_loss = torch.tensor(0.0, device=target_device)
            mean_entropy = torch.tensor(0.0)

        loss = retain_coeff * retain_loss + forget_coeff * forget_loss

        if (
            self.current_training_step % 8 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            entropy_val = (
                mean_entropy.item()
                if isinstance(mean_entropy, torch.Tensor)
                else mean_entropy
            )
            retain_val = (
                retain_loss.item()
                if isinstance(retain_loss, torch.Tensor)
                else retain_loss
            )
            print(
                f"retain_coeff: {retain_coeff:.4f} || "
                f"forget_coeff: {forget_coeff:.4f} || "
                f"retain_loss: {retain_val:.4f} || "
                f"forget_loss: {forget_loss:.4f} || "
                f"mean_entropy: {entropy_val:.4f}"
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


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
    parser.add_argument("--wmdp_eval_limit", type=int, default=None)
    parser.add_argument("--mmlu_agieval_limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pdbs", type=int, default=4)
    parser.add_argument("--retain_coef", type=float, default=5.0)
    parser.add_argument("--remove_coef", type=float, default=5.0)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30],
        help="List of layers to target",
    )
    parser.add_argument(
        "--model_name", type=str, default="EleutherAI/deep-ignorance-unfiltered"
    )
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument(
        "--lens_path", type=str, required=True, help="Path to tuned lens weights"
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip final evaluation (for faster tuning)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )

    args = parser.parse_args()

    if "smollm2" in args.model_name:
        args.layers = [l for l in args.layers if l < 24]

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(args.model_name, revision=args.revision)
    train_dataset = get_unlearning_dataset(args, tokenizer, NUM_PROC)

    # Load frozen tuned lens
    print(f"Loading tuned lens from: {args.lens_path}")
    device = next(model.parameters()).device
    lens = TunedLens.from_model(model, bias=True)
    lens_state_dict = torch.load(f"{args.lens_path}/params.pt", map_location=device)

    mapped_state_dict = {}
    num_layers = len(lens)
    for i in range(num_layers):
        src_weight_key = f"{i}.weight"
        src_bias_key = f"{i}.bias"
        dst_weight_key = f"layer_translators.{i}.weight"
        dst_bias_key = f"layer_translators.{i}.bias"
        if src_weight_key in lens_state_dict:
            mapped_state_dict[dst_weight_key] = lens_state_dict[src_weight_key]
        if src_bias_key in lens_state_dict:
            mapped_state_dict[dst_bias_key] = lens_state_dict[src_bias_key]

    current_state = lens.state_dict()
    for key in current_state:
        if key.startswith("unembed"):
            mapped_state_dict[key] = current_state[key]

    lens.load_state_dict(mapped_state_dict)

    for submodule in lens.modules():
        remove_hook_from_module(submodule)

    lens = lens.to(device=device, dtype=torch.bfloat16)
    lens.eval()
    for param in lens.parameters():
        param.requires_grad = False
    print(f"Loaded lens with {len(lens)} layer translators (frozen)")

    # Skip frozen reference model - retain loss requires too much memory
    # Use LoRA-based unlearning script if you need retain loss
    frozen_ref_model = None
    if args.retain_coef > 0:
        print("WARNING: retain_coef > 0 but no frozen reference model available.")
        print("SFT mode cannot support retain loss without doubling GPU memory.")
        print("Setting retain_coef=0. Use LoRA-based script for retain loss.")
        args.retain_coef = 0.0

    # Enable gradients on training model
    for param in model.parameters():
        param.requires_grad = True

    model = cast(PreTrainedModel, model)
    model.enable_input_require_grads()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = 32
    grad_acc_steps = max(1, global_batch_size // (args.pdbs * world_size))

    print(
        f"Running with {world_size} GPUs. Per device batch: {args.pdbs}. "
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
    )

    trainer = SFTUnlearningTrainer(
        args,
        model,
        training_args,
        train_dataset,
        tokenizer,
        args.layers,
        lens=lens,
        frozen_ref_model=frozen_ref_model,
    )

    model.train()
    trainer.train()

    if args.save_name:
        args.save_name = f"{args.save_name}_sft"
        if "models/" in args.model_name:
            args.model_name = args.model_name.replace("models/", "")
        model.save_pretrained(f"./models/{args.model_name}_{args.save_name}")
        tokenizer.save_pretrained(f"./models/{args.model_name}_{args.save_name}")

    print("Done :)")
