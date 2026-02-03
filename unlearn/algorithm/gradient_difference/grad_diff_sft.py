"""Gradient Difference Unlearning (SFT version).

Full-parameter training variant of gradient difference:
- Retain loss: KL divergence between current model and frozen reference model
- Forget loss: Cross-entropy on forget data (negated for gradient ascent)

Loss = retain_coef * kl_retain_loss - forget_coef * forget_ce_loss
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import seed_worker

from unlearn.utils.unlearning_dataset import get_unlearning_dataset


def get_model_and_tokenizer(model_name, revision="main"):
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        device = f"cuda:{local_rank}"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            device_map=None,
        )
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, device_map="auto", use_cache=False
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if "Unlearning" in model_name:
        tokenizer.add_special_tokens(
            {
                "pad_token": "<|padding|>",
                "eos_token": "<|endoftext|>",
                "bos_token": "<|startoftext|>",
            }
        )
        tokenizer.padding_side = "left"
    else:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    return model, tokenizer


def save_checkpoint(trainer: Trainer, save_path: Path, tokenizer):
    trainer.accelerator.wait_for_everyone()
    state_dict = trainer.accelerator.get_state_dict(trainer.model, unwrap=False)
    if trainer.accelerator.is_main_process:
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        unwrapped_model.save_pretrained(
            save_path, state_dict=state_dict, safe_serialization=True
        )
        tokenizer.save_pretrained(save_path)
    trainer.accelerator.wait_for_everyone()


def compute_lm_loss(logits, labels, attention_mask):
    """Compute language modeling cross-entropy loss with masking."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous()

    batch_size, seq_len, vocab_size = shift_logits.shape
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_mask = shift_mask.view(-1)

    loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
    loss = (loss * shift_mask).sum() / (shift_mask.sum() + 1e-8)
    return loss


class GradDiffSftTrainer(Trainer):
    def __init__(
        self,
        run_cfg,
        model,
        args,
        train_dataset,
        tokenizer,
        frozen_ref_model=None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        self.run_cfg = run_cfg
        self.num_training_steps = self.args.max_steps
        self.current_training_step = 0
        self.tokenizer = tokenizer
        self.frozen_ref_model = frozen_ref_model

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
        target_device = (
            inputs["input_ids"].device
            if hasattr(inputs["input_ids"], "device")
            else next(model.parameters()).device
        )

        retain_input_ids = inputs["input_ids"].to(target_device)
        retain_attention_mask = inputs["attention_mask"].to(target_device)
        forget_input_ids = inputs["bio_remove_input_ids"].to(target_device)
        forget_attention_mask = inputs["bio_remove_attention_mask"].to(target_device)

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        scheduled_coeff = min(
            1.0,
            self.current_training_step
            / (self.run_cfg.num_train_examples / (self.run_cfg.pdbs * world_size)),
        )
        retain_coeff = self.run_cfg.retain_coef * (0.9 + 0.1 * scheduled_coeff)
        forget_coeff = self.run_cfg.forget_coef * (1 - 0.25 * scheduled_coeff)

        model.train()

        # Retain loss: KL divergence with frozen reference model
        if retain_coeff > 0 and self.frozen_ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.frozen_ref_model(
                    retain_input_ids, attention_mask=retain_attention_mask
                )
                ref_logits = ref_outputs.logits.to(target_device)

            current_logits = model(
                input_ids=retain_input_ids,
                attention_mask=retain_attention_mask,
            ).logits

            retain_loss = F.kl_div(
                input=F.log_softmax(current_logits, dim=-1),
                target=F.softmax(ref_logits, dim=-1),
                reduction="batchmean",
            )
        else:
            retain_loss = torch.tensor(0.0, device=target_device)

        # Forget loss: CE on forget data (negated for gradient ascent)
        forget_outputs = model(
            input_ids=forget_input_ids,
            attention_mask=forget_attention_mask,
        )
        forget_ce_loss = compute_lm_loss(
            forget_outputs.logits, forget_input_ids, forget_attention_mask
        )

        # Gradient difference: minimize retain KL, maximize forget CE
        loss = retain_coeff * retain_loss - forget_coeff * forget_ce_loss

        if (
            self.current_training_step % 8 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            print(
                f"step: {self.current_training_step} || "
                f"retain_coeff: {retain_coeff:.4f} || "
                f"forget_coeff: {forget_coeff:.4f} || "
                f"retain_loss (kl): {retain_loss.item():.4f} || "
                f"forget_ce_loss: {forget_ce_loss.item():.4f} || "
                f"total_loss: {loss.item():.4f}"
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


@dataclass
class GradDiffSftConfig:
    num_train_examples: int = 1024
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    lr: float = 2e-4
    pdbs: int = 4
    retain_coef: float = 5.0
    forget_coef: float = 1.0
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    revision: str = "main"
    epochs: int = 1
    warmup_ratio: float = 0.0
    use_ultrachat: bool = False
    skip_eval: bool = False


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    parser = ArgumentParser()
    parser.add_arguments(GradDiffSftConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"  {arg}: {value}")

    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision
    )

    for param in model.parameters():
        param.requires_grad = True
    model.enable_input_require_grads()

    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)

    # Frozen reference model for KL retain loss
    frozen_ref_model = AutoModelForCausalLM.from_pretrained(
        run_cfg.model_name,
        revision=run_cfg.revision,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    frozen_ref_model.eval()
    for param in frozen_ref_model.parameters():
        param.requires_grad = False
    print("Loaded frozen reference model.")

    global_batch_size = 32
    grad_acc_steps = max(1, global_batch_size // (run_cfg.pdbs * world_size))

    print(
        f"Running with {world_size} GPUs. Per device batch: {run_cfg.pdbs}. "
        f"Grad acc steps: {grad_acc_steps}."
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=run_cfg.lr,
        warmup_ratio=run_cfg.warmup_ratio,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=run_cfg.pdbs,
        per_device_eval_batch_size=run_cfg.pdbs,
        num_train_epochs=run_cfg.epochs,
        weight_decay=0.01,
        gradient_checkpointing=True,
        bf16=True,
        max_grad_norm=1.0,
        save_strategy="no",
    )

    trainer = GradDiffSftTrainer(
        run_cfg,
        model,
        training_args,
        train_dataset,
        tokenizer,
        frozen_ref_model=frozen_ref_model,
    )

    model.train()
    trainer.train()

    if run_cfg.save_path:
        save_checkpoint(trainer, run_cfg.save_path, tokenizer)

    if dist.is_initialized():
        dist.destroy_process_group()

    print("Done.")
