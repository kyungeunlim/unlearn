"""SimpleNPO (SimNPO) Unlearning.

SimNPO simplifies NPO by removing the reference model. Instead of computing
current_loss - ref_loss, it uses just current_loss directly.

Loss:
    forget_loss = -logsigmoid(beta * CE_forget) * 2/beta
    total_loss = forget_loss + gamma * retain_loss
"""

import os
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from simple_parsing import ArgumentParser
from transformers import PreTrainedModel, Trainer, TrainingArguments

from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer, save_checkpoint


def compute_lm_loss(logits, labels, attention_mask):
    """Compute language modeling cross-entropy loss with proper shifting and masking."""
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


class SimpleNPOTrainer(Trainer):
    def __init__(self, run_args, model, args, train_dataset, tokenizer, **kwargs):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        self.run_args = run_args
        self.current_training_step = 0
        self.tokenizer = tokenizer

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        target_device = inputs["input_ids"].device

        # Forget data
        forget_input_ids = inputs["bio_remove_input_ids"].to(target_device)
        forget_attention_mask = inputs["bio_remove_attention_mask"].to(target_device)

        # Retain data
        retain_input_ids = inputs["input_ids"].to(target_device)
        retain_attention_mask = inputs["attention_mask"].to(target_device)

        beta = self.run_args.beta
        gamma = self.run_args.gamma

        # Forward on forget data
        forget_outputs = model(
            input_ids=forget_input_ids,
            attention_mask=forget_attention_mask,
        )
        forget_ce = compute_lm_loss(
            forget_outputs.logits, forget_input_ids, forget_attention_mask
        )

        # Forward on retain data
        retain_outputs = model(
            input_ids=retain_input_ids,
            attention_mask=retain_attention_mask,
        )
        retain_loss = compute_lm_loss(
            retain_outputs.logits, retain_input_ids, retain_attention_mask
        )

        # SimNPO loss: -logsigmoid(beta * CE_forget) * 2/beta
        forget_loss = -F.logsigmoid(beta * forget_ce) * (2.0 / beta)

        total_loss = forget_loss + gamma * retain_loss

        if (
            self.current_training_step % 10 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            print(
                f"step {self.current_training_step} || "
                f"forget_ce: {forget_ce.item():.4f} || "
                f"forget_loss: {forget_loss.item():.4f} || "
                f"retain_loss: {retain_loss.item():.4f} || "
                f"total_loss: {total_loss.item():.4f}"
            )

        self.current_training_step += 1
        return (total_loss,) if return_outputs else total_loss


@dataclass
class SimpleNPOConfig:
    num_train_examples: int = 1024
    lr: float = 1e-3
    pdbs: int = 4
    lora_r: int = 16
    lora: bool = True
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    revision: str = "main"

    # SimNPO-specific
    beta: float = 0.1
    gamma: float = 1.0

    # Dataset options forwarded to get_unlearning_dataset
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: str = "rewritten"


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2

    parser = ArgumentParser()
    parser.add_arguments(SimpleNPOConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"  {arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision
    )

    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)

    if run_cfg.lora:
        # Determine max layer for layers_to_transform
        num_layers = model.config.num_hidden_layers
        lora_layers_to_transform = list(range(num_layers))

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
    grad_acc_steps = max(1, global_batch_size // (run_cfg.pdbs * world_size))

    print(
        f"Running with {world_size} GPUs. Per device batch: {run_cfg.pdbs}. "
        f"Grad Acc steps: {grad_acc_steps}."
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=run_cfg.lr,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=run_cfg.pdbs,
        per_device_eval_batch_size=run_cfg.pdbs,
        num_train_epochs=1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        fp16=True,
        save_strategy="no",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    trainer = SimpleNPOTrainer(run_cfg, model, training_args, train_dataset, tokenizer)

    model.train()
    trainer.train()

    if run_cfg.lora:
        model = model.merge_and_unload()
        trainer.model = model

    if run_cfg.save_path:
        save_checkpoint(trainer, run_cfg.save_path, tokenizer)

    print("Done.")
