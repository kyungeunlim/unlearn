import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
)


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


def unwrap_model(model):
    """Get the underlying model from DDP/FSDP wrapper if present."""
    if hasattr(model, "module"):
        return model.module
    return model


def get_model_and_tokenizer(model_name, revision="main", dm="auto"):
    # Check if running in distributed mode (accelerate/torchrun sets LOCAL_RANK)
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
        model = model.to(device)  # type: ignore
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, device_map=dm, use_cache=False
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
