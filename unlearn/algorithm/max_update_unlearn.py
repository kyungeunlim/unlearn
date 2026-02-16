"""Max Update Unlearning.

Maximizes parameter deviation from the initial model while maintaining
retain quality via SFT cross-entropy. No forget loss term -- the theory
is that maximizing parameter change while constraining retain quality
pushes the model away from dangerous capabilities.

Loss = retain_coef * sft_loss - update_coef * ||theta - theta_0||^2
"""

import os
import sys
from dataclasses import dataclass, field
from typing import cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import seed_worker

from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import (
    get_model_and_tokenizer,
    save_checkpoint,
)


def compute_lm_loss(logits, labels, attention_mask):
    """Standard SFT cross-entropy with attention masking."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()

    _, _, vocab_size = shift_logits.shape
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_mask = shift_mask.view(-1)

    loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
    return (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)


class MaxUpdateTrainer(Trainer):
    def __init__(
        self,
        run_cfg,
        model,
        args,
        train_dataset,
        tokenizer,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        self.run_cfg = run_cfg
        self.current_training_step = 0
        self.tokenizer = tokenizer

        # Clone initial trainable params to CPU
        self.initial_params: dict[str, torch.Tensor] = {}
        self.total_trainable_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.initial_params[name] = param.data.clone().cpu()
                self.total_trainable_params += param.numel()

        print(
            f"Stored {len(self.initial_params)} initial param tensors "
            f"({self.total_trainable_params} params total)"
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

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        target_device = (
            inputs["input_ids"].device
            if hasattr(inputs["input_ids"], "device")
            else next(model.parameters()).device
        )

        retain_ids = inputs["input_ids"].to(target_device)
        retain_mask = inputs["attention_mask"].to(target_device)

        # Forward + backward on SFT retain loss only (keeps autograd graph small)
        retain_logits = model(
            input_ids=retain_ids,
            attention_mask=retain_mask,
        ).logits
        sft_loss = compute_lm_loss(retain_logits, retain_ids, retain_mask)
        scaled_sft_loss = self.run_cfg.retain_coef * sft_loss

        self.accelerator.backward(scaled_sft_loss)

        # Add update gradient analytically: grad of -update_coef * ||theta - theta_0||^2
        # = -2 * update_coef * (theta - theta_0)
        update_norm = torch.tensor(0.0, device=target_device)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.initial_params:
                    init_p = self.initial_params[name].to(param.device)
                    diff = param.data - init_p
                    update_grad = -2.0 * self.run_cfg.update_coef * diff

                    if param.grad is not None:
                        if self.run_cfg.same_sign_grads:
                            same_sign = update_grad.sign() == param.grad.sign()
                            param.grad.add_(update_grad * same_sign)
                        else:
                            param.grad.add_(update_grad)
                    else:
                        param.grad = update_grad

                    update_norm = update_norm + diff.pow(2).sum()

        loss = (
            self.run_cfg.retain_coef * sft_loss - self.run_cfg.update_coef * update_norm
        )

        if (
            self.current_training_step % 8 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            from tqdm import tqdm

            tag = " || [same_sign_grads]" if self.run_cfg.same_sign_grads else ""
            tqdm.write(
                f"step: {self.current_training_step} || "
                f"sft_loss: {sft_loss.item():.4f} || "
                f"update_norm: {update_norm.item():.6f} || "
                f"combined_loss: {loss.item():.4f}"
                f"{tag}"
            )
            sys.stdout.flush()

        self.current_training_step += 1
        return loss.detach()


@dataclass
class MaxUpdateConfig:
    retain_coef: float = 1.0
    update_coef: float = 10.0
    same_sign_grads: bool = False
    num_train_examples: int = 1024
    epochs: int = 1
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: str = "rewritten"
    lr: float = 2e-4
    pdbs: int = 4
    lora: bool = False
    lora_r: int = 16
    layers: list[int] = field(default_factory=lambda: list(range(32)))
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    revision: str = "main"
    hidden_dim: int = 4096


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2

    parser = ArgumentParser()
    parser.add_arguments(MaxUpdateConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"  {arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision
    )

    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)

    if not run_cfg.lora:
        for param in model.parameters():
            param.requires_grad = True

    if run_cfg.lora:
        lora_layers_to_transform = list(range(max(run_cfg.layers) + 1))
        lora_config = LoraConfig(
            r=run_cfg.lora_r,
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
                if "OLMo" in run_cfg.model_name
                else None
            ),
            lora_dropout=0.05,
            bias="none",
            layers_to_transform=lora_layers_to_transform,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model = cast(PreTrainedModel, model)
    model.enable_input_require_grads()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = 32
    grad_acc_steps = max(
        1,
        global_batch_size // (run_cfg.pdbs * world_size),
    )

    print(
        f"Running with {world_size} GPUs. "
        f"Per device batch: {run_cfg.pdbs}. "
        f"Grad acc steps: {grad_acc_steps}."
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=run_cfg.lr,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=run_cfg.pdbs,
        per_device_eval_batch_size=run_cfg.pdbs,
        num_train_epochs=run_cfg.epochs,
        weight_decay=0.01,
        gradient_checkpointing=True,
        bf16=True,
        save_strategy="no",
        ddp_find_unused_parameters=False,
    )

    trainer = MaxUpdateTrainer(
        run_cfg,
        model,
        training_args,
        train_dataset,
        tokenizer,
    )

    model.train()
    trainer.train()

    if run_cfg.lora:
        model = model.merge_and_unload()

    if run_cfg.save_path:
        save_checkpoint(trainer, run_cfg.save_path, tokenizer)

    print("Done.")
