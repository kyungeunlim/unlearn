"""Weight Partition Unlearning with SFT.

Partitions output dimensions of target linear layers: 10% forget, 90%
retain. Training uses SFT (cross-entropy) on retain data and a direct
||y_retain||^2 penalty on forget data to push forget knowledge into
the forget subnetwork. Post-training ablation zeros out the forget
dimensions to remove the capability.
"""

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from simple_parsing import ArgumentParser
from torch import nn
from transformers import PreTrainedModel, TrainingArguments
from transformers.modeling_utils import unwrap_model

from unlearn.algorithm.base_unlearn import UnlearningTrainer
from unlearn.utils.hook import resolve_layer_names
from unlearn.utils.keyword_masks import apply_keyword_masks
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import (
    get_model_and_tokenizer,
    save_checkpoint,
)

PEFT_INTERNALS = {
    "base_layer",
    "lora_A",
    "lora_B",
    "lora_dropout",
    "lora_embedding_A",
    "lora_embedding_B",
}


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


def _linear_out_features(mod: nn.Module) -> int | None:
    if isinstance(mod, nn.Linear):
        return mod.out_features
    if hasattr(mod, "base_layer") and isinstance(mod.base_layer, nn.Linear):
        return mod.base_layer.out_features
    return None


def _iter_target_linears(model, block_names):
    """Yield (name, module, out_features) for target linears."""
    for name, module in model.named_modules():
        if not any(name.startswith(bn + ".") for bn in block_names):
            continue
        if PEFT_INTERNALS & set(name.split(".")):
            continue
        d_out = _linear_out_features(module)
        if d_out is not None:
            yield name, module, d_out


def _build_masks(model, layers, forget_frac, seed=42):
    """Build deterministic retain/forget masks for target linears."""
    layer_id_to_name = resolve_layer_names(model, layers)
    block_names = set(layer_id_to_name.values())
    rng = torch.Generator().manual_seed(seed)
    masks: dict[str, torch.Tensor] = {}

    for name, _mod, d_out in _iter_target_linears(model, block_names):
        n_forget = max(1, int(forget_frac * d_out))
        perm = torch.randperm(d_out, generator=rng)
        mask = torch.ones(d_out, dtype=torch.bool)
        mask[perm[:n_forget]] = False
        masks[name] = mask

    return masks, layer_id_to_name


def ablate_forget_dimensions(
    model: nn.Module,
    layers: list[int],
    forget_frac: float = 0.1,
    seed: int = 42,
) -> int:
    """Zero out forget-dimension weights after training.

    Regenerates the same partition masks (seeded RNG) and sets
    the corresponding weight rows and biases to zero.
    Returns the number of ablated linear layers.
    """
    layer_id_to_name = resolve_layer_names(model, layers)
    block_names = set(layer_id_to_name.values())
    rng = torch.Generator().manual_seed(seed)
    n_ablated = 0

    with torch.no_grad():
        for name, module, d_out in _iter_target_linears(model, block_names):
            n_forget = max(1, int(forget_frac * d_out))
            perm = torch.randperm(d_out, generator=rng)
            forget_dims = perm[:n_forget]

            if isinstance(module, nn.Linear):
                module.weight[forget_dims, :] = 0.0
                if module.bias is not None:
                    module.bias[forget_dims] = 0.0
                n_ablated += 1

    return n_ablated


class WeightPartitionTrainer(UnlearningTrainer):
    """SFT retain + direct ||y_retain||^2 penalty for forget."""

    def __init__(
        self,
        run_args,
        model: PreTrainedModel,
        args,
        train_dataset,
        tokenizer,
        # Just target_layers in fact
        lora_target_layers,
        **kwargs,
    ):
        super().__init__(
            run_args,
            model,
            args,
            train_dataset,
            tokenizer,
            lora_target_layers,
            **kwargs,
        )
        self.forget_frac = run_args.forget_frac
        self.retain_dim_masks: dict[str, torch.Tensor] = {}
        self._build_partition_masks()

    def _build_partition_masks(self):
        unwrapped = unwrap_model(self.model)
        masks, _ = _build_masks(
            unwrapped,
            self.lora_target_layers,
            self.forget_frac,
        )
        self.retain_dim_masks = masks

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        unwrapped = unwrap_model(model)

        target_device = (
            inputs["input_ids"].device
            if hasattr(inputs["input_ids"], "device")
            else unwrapped.device
        )

        retain_ids = inputs["input_ids"].to(target_device)
        retain_mask = inputs["attention_mask"].to(target_device)
        forget_ids = inputs["bio_remove_input_ids"].to(target_device)
        forget_mask = inputs["bio_remove_attention_mask"].to(target_device)

        # Scheduled coefficients
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        sched = min(
            1.0,
            self.current_training_step
            / (self.run_args.num_train_examples / (self.run_args.pdbs * world_size)),
        )
        retain_coeff = self.retain_coef * (0.9 + 0.1 * sched)
        forget_coeff = self.remove_coef * (1 - 0.25 * sched)

        unwrapped.train()

        # ---- SFT retain loss ----
        if retain_coeff > 0:
            retain_logits = unwrapped(
                input_ids=retain_ids,
                attention_mask=retain_mask,
            ).logits
            retain_loss = compute_lm_loss(retain_logits, retain_ids, retain_mask)
        else:
            retain_loss = torch.tensor(0.0, device=target_device)

        # ---- Forget: direct ||y_retain||^2 ----
        if forget_coeff > 0:
            linear_outs: dict[str, torch.Tensor] = {}
            handles = []

            def _hook(mod, inp, out, *, name: str):
                val = out[0] if isinstance(out, tuple) else out
                linear_outs[name] = val

            for name, module in unwrapped.named_modules():
                if name in self.retain_dim_masks:
                    handles.append(
                        module.register_forward_hook(partial(_hook, name=name))
                    )

            unwrapped(
                input_ids=forget_ids,
                attention_mask=forget_mask,
            )

            for h in handles:
                h.remove()
            # handles.clear()

            forget_loss = torch.tensor(0.0, device=target_device)
            n_terms = 0

            for lname, dmask in self.retain_dim_masks.items():
                if lname not in linear_outs:
                    continue

                out = linear_outs[lname].float()
                m = dmask.to(target_device)
                mse = out[..., m].pow(2).mean(dim=-1)
                masked = mse * forget_mask
                layer_loss = masked.sum() / (forget_mask.sum() + 1e-8)
                forget_loss = forget_loss + layer_loss
                n_terms += 1

            if n_terms > 0:
                forget_loss = forget_loss / n_terms
        else:
            forget_loss = torch.tensor(0.0, device=target_device)

        loss = retain_coeff * retain_loss + forget_coeff * forget_loss

        if (
            self.current_training_step % 8 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            from tqdm import tqdm

            tqdm.write(
                f"step: {self.current_training_step} || "
                f"retain_coeff: {retain_coeff:.4f} || "
                f"forget_coeff: {forget_coeff:.4f} || "
                f"sft_loss: "
                f"{retain_loss.item():.4f} || "
                f"forget_loss: "
                f"{forget_loss.item():.4f}"
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


@dataclass
class WeightPartitionConfig:
    num_train_examples: int = 1024
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    lr: float = 1e-3
    pdbs: int = 4
    retain_coef: float = 1.0
    remove_coef: float = 5.0
    forget_frac: float = 0.1
    ablate: bool = True
    lora_r: float = 16
    lora: bool = True
    layers: list[int] = field(default_factory=lambda: list(range(32)))
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    revision: str = "main"
    hidden_dim: int = 4096


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2

    parser = ArgumentParser()
    parser.add_arguments(WeightPartitionConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"{arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision
    )

    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)
    train_dataset.tokenized_bio_remove_dataset = apply_keyword_masks(
        train_dataset.tokenized_bio_remove_dataset,
        run_cfg,
        tokenizer,
    )

    lora_layers_to_transform = list(range(max(run_cfg.layers) + 1))

    if run_cfg.lora:
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
    )

    trainer = WeightPartitionTrainer(
        run_cfg,
        model,
        training_args,
        train_dataset,
        tokenizer,
        run_cfg.layers,
    )

    model.train()
    trainer.train()

    if run_cfg.lora:
        model = model.merge_and_unload()

    if run_cfg.ablate:
        n = ablate_forget_dimensions(
            model,
            run_cfg.layers,
            run_cfg.forget_frac,
        )
        print(f"Ablated forget dimensions in " f"{n} linear layers.")

    if run_cfg.save_path:
        save_checkpoint(trainer, run_cfg.save_path, tokenizer)

    print("Done.")
