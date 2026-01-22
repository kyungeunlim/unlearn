import os

import torch
from bergson.config import IndexConfig
from bergson.data import load_data_string, tokenize
from bergson.utils.utils import get_layer_list
from datasets import (
    Dataset,
    IterableDataset,
)
from peft import PeftConfig, PeftModel, get_peft_model_state_dict
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


def setup_model_and_peft(
    cfg: IndexConfig,
    local_rank: int,
) -> tuple[AutoModelForCausalLM, set | None]:
    """Handle model loading, quantization, FSDP, and PEFT detection"""

    match cfg.precision:
        case "bf16":
            dtype = torch.bfloat16
        case "fp16":
            dtype = torch.float16
        case "fp32":
            dtype = torch.float32
        case "int4" | "int8":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        case "auto":
            dtype = "auto"
        case other:
            raise ValueError(f"Unsupported precision: {other}")

    # Common configuration
    device_map = {"": f"cuda:{local_rank}"} if not cfg.fsdp else "cpu"
    quantization_config = None
    if cfg.precision in ("int4", "int8"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=cfg.precision == "int4",
            load_in_8bit=cfg.precision == "int8",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Try to detect PEFT model
    try:
        peft_config = PeftConfig.from_pretrained(cfg.model)
    except ValueError:
        peft_config = None

    if peft_config is None:
        # Load regular model
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            revision=cfg.revision,
        )
        target_modules = None

    else:
        # Load PEFT model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,  # type: ignore
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            revision=cfg.revision,
        )

        model = PeftModel.from_pretrained(
            base_model,
            cfg.model,
            device_map=device_map,
            autocast_adapter_dtype=False,
        )

        # Extract target modules
        target_modules = set()
        peft_state_dict = get_peft_model_state_dict(model=model)
        for adapter in model.peft_config.keys():
            for name in list(peft_state_dict.keys()):
                prefix = name.removesuffix(".weight")
                processed_name = f"{prefix}.{adapter}".removeprefix("base_model.")
                try:
                    model.get_submodule(processed_name)
                    target_modules.add(processed_name)
                except AttributeError:
                    print(
                        f"Adapter parameter '{processed_name}' not found in the model."
                    )

    # Configure gradients
    model.requires_grad_(False)
    model.get_input_embeddings().requires_grad_(True)  # type: ignore

    # Apply FSDP if needed
    if cfg.fsdp:
        for layer in get_layer_list(model):  # type: ignore
            fully_shard(layer)
        fully_shard(model)

    return model, target_modules  # type: ignore


def setup_data_pipeline(cfg: IndexConfig) -> Dataset | IterableDataset:
    """Handle data loading and preprocessing"""
    ds = load_data_string(
        cfg.data.dataset, cfg.data.split, cfg.data.subset, cfg.data.data_args
    )

    # In many cases the token_batch_size may be smaller than the max length allowed by
    # the model. If cfg.data.truncation is True, we use the tokenizer to truncate
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer or cfg.model)
    tokenizer.model_max_length = min(tokenizer.model_max_length, cfg.token_batch_size)

    remove_columns = ds.column_names if cfg.drop_columns else None

    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs=dict(args=cfg.data, tokenizer=tokenizer),
    )
    if remove_columns is not None:
        ds = ds.remove_columns(remove_columns)

    return ds
