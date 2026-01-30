# A base script for prototyping unlearning methods.
# Uses Cas's circuit breakers implementation with DDP enabled.

import os
from dataclasses import dataclass, field
from typing import Dict, Literal, cast

import torch
from peft import LoraConfig, get_peft_model
from simple_parsing import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import seed_worker

from unlearn.utils.hook import ActivationCapture
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.utils import assert_type
from unlearn.utils.worker_utils import get_model_and_tokenizer


class UnlearningTrainer(Trainer):
    def __init__(
        self,
        run_args,
        model: PreTrainedModel,
        args,
        train_dataset,
        tokenizer,
        lora_target_layers,
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
        self.lora_target_layers = lora_target_layers
        self.model = model
        self.retain_coef = self.run_args.retain_coef
        self.remove_coef = self.run_args.remove_coef
        self.trainer_tokenizer = tokenizer

        # --- Resolve Layer Names for Hooks ---
        # We need to map integer indices (e.g., 5) to module
        # names (e.g., "model.layers.5")
        self.layer_id_to_name = self._resolve_layer_names(model, lora_target_layers)
        self.target_module_names = list(self.layer_id_to_name.values())

    def _resolve_layer_names(self, model, layer_indices) -> Dict[int, str]:
        """
        Dynamically finds the module names corresponding to the requested layer indices.
        Handles PEFT wrapping and different architectures (OLMo, Llama, etc.).
        """
        unwrapped = unwrap_model(model)

        # Navigate through PEFT/DDP wrappers to find the base transformer
        base = unwrapped
        if hasattr(base, "base_model"):
            base = base.base_model
        if hasattr(base, "model"):
            base = base.model  # type: ignore

        base = assert_type(nn.Module, base)

        # Identify the list of layers (e.g., 'layers', 'blocks', 'h')
        layer_list_name = None
        for name, module in base.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                # Heuristic: usually the longest ModuleList is the transformer blocks
                # and contains 'layers' or 'blocks' in the name
                if "layers" in name or "blocks" in name or "h" in name:
                    layer_list_name = name
                    break

        if not layer_list_name:
            # Fallback for OLMo specific if naming is tricky
            if hasattr(base, "transformer") and hasattr(base.transformer, "blocks"):
                layer_list_name = "transformer.blocks"
            elif hasattr(base, "layers"):
                layer_list_name = "layers"
            else:
                raise ValueError(
                    "Could not automatically locate transformer layer list."
                )

        # Construct full names relative to the unwrapped model
        # Note: named_modules() on the full model will include prefixes.
        # We scan the full unwrapped model to match the exact string for the hook.

        mapping = {}
        # We need the prefix required to reach 'base' from 'unwrapped'
        # To do this safely, we just iterate the full unwrapped model
        # and find the matching suffix.

        found_count = 0
        target_indices = set(layer_indices)

        for name, module in unwrapped.named_modules():
            # Check if this module is one of the layers we want
            # The name usually ends in "{layer_list_name}.{index}"
            if layer_list_name in name:
                try:
                    # Extract index from end of string (e.g., "model.layers.10" -> 10)
                    parts = name.split(".")
                    idx = int(parts[-1])
                    if idx in target_indices:
                        mapping[idx] = name
                        found_count += 1
                except ValueError:
                    continue

        if len(mapping) != len(target_indices):
            print(
                f"Warning: requested {len(target_indices)} layers, "
                f"but found {len(mapping)}."
            )

        return mapping

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

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))  # type: ignore


class RRTrainer(UnlearningTrainer):

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        unwrapped_model = unwrap_model(model)

        target_device = (
            inputs["input_ids"].device
            if hasattr(inputs["input_ids"], "device")
            else unwrapped_model.device
        )

        # === retain ===
        retain_input_ids = inputs.get("input_ids").to(target_device)  # type: ignore
        retain_attention_mask = inputs.get("attention_mask").to(target_device)  # type: ignore
        # ==== cb ====
        circuit_breaker_input_ids = inputs.get("bio_remove_input_ids").to(target_device)  # type: ignore
        circuit_breaker_attention_mask = inputs.get("bio_remove_attention_mask").to(  # type: ignore
            target_device  # type: ignore
        )

        # ==== Inputs ====
        retain_inputs_dict = dict(
            input_ids=retain_input_ids,
            attention_mask=retain_attention_mask,
            output_hidden_states=False,
        )
        cb_inputs_dict = dict(
            input_ids=circuit_breaker_input_ids,
            attention_mask=circuit_breaker_attention_mask,
            output_hidden_states=False,
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
        circuit_breaker_coeff = self.remove_coef * (1 - 0.25 * scheduled_coeff)

        broadcast_retain_mask = retain_attention_mask.unsqueeze(0).unsqueeze(-1)
        broadcast_cb_mask = circuit_breaker_attention_mask.unsqueeze(0).unsqueeze(-1)

        # Initialize Hook Manager
        # We use unwrapped_model here to ensure names match what we resolved in __init__
        capturer = ActivationCapture(unwrapped_model, self.target_module_names)

        # --- Forward Pass 1: Reference (No Adapter) ---
        with unwrapped_model.disable_adapter():  # type: ignore
            unwrapped_model.eval()
            capturer.register()  # Attach hooks

            with torch.no_grad():
                ### Retain control
                if retain_coeff > 0:
                    unwrapped_model(**retain_inputs_dict)
                    # Extract and Stack
                    # Logic: Map the requested layer IDs -> Get Name -> Get Tensor
                    orig_retain_hidden = torch.stack(
                        [
                            capturer.activations[self.layer_id_to_name[l]].detach()
                            for l in self.lora_target_layers
                        ]
                    )
                    orig_retain_hidden *= broadcast_retain_mask

                # Clear activations to free memory before next pass, but
                # keep hooks if needed
                # Actually simpler to just capture both if memory allows, but
                # separating is safer
                capturer.activations = {}

                ### Circuit Breaker control
                if circuit_breaker_coeff > 0:
                    unwrapped_model(**cb_inputs_dict)
                    circuit_breaker_hidden = torch.stack(
                        [
                            capturer.activations[self.layer_id_to_name[l]].detach()
                            for l in self.lora_target_layers
                        ]
                    )

            capturer.remove()  # Remove hooks

        unwrapped_model.train()

        # --- Forward Pass 2: Training (With Adapter) ---

        # Re-register hooks for the training pass
        capturer.register()

        ### Retain control
        if retain_coeff > 0:
            unwrapped_model(**retain_inputs_dict)

            lora_retain_hidden = torch.stack(
                [
                    capturer.activations[self.layer_id_to_name[l]]
                    for l in self.lora_target_layers
                ]
            )
            lora_retain_hidden = lora_retain_hidden * broadcast_retain_mask

            retain_loss = torch.norm(
                lora_retain_hidden - orig_retain_hidden, dim=-1, p=2, dtype=torch.float
            ).nanmean()

            capturer.activations = {}  # Clear for next input
        else:
            retain_loss = 0

        ### Circuit Breaker control
        if circuit_breaker_coeff > 0:
            unwrapped_model(**cb_inputs_dict)

            lora_circuit_breaker_hidden = torch.stack(
                [
                    capturer.activations[self.layer_id_to_name[l]]
                    for l in self.lora_target_layers
                ]
            )

            # Normalize
            normalized_lora_circuit_breaker_outputs = lora_circuit_breaker_hidden / (
                torch.norm(
                    lora_circuit_breaker_hidden, dim=-1, keepdim=True, dtype=torch.float
                )
            )
            normalized_circuit_breaker_outputs = circuit_breaker_hidden / (
                torch.norm(
                    circuit_breaker_hidden, dim=-1, keepdim=True, dtype=torch.float
                )
            )

            inner_product = (
                normalized_lora_circuit_breaker_outputs
                * normalized_circuit_breaker_outputs
            ) * broadcast_cb_mask

            denom = circuit_breaker_attention_mask.sum() * len(self.lora_target_layers)
            circuit_breaker_loss = torch.relu(inner_product.nansum(dim=-1)).nansum() / (
                denom + 1e-6
            )
        else:
            circuit_breaker_loss = 0

        capturer.remove()  # Clean up

        loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss

        if (
            self.current_training_step % 32 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            print(
                f"retain_coeff: {retain_coeff:.4f} || cb_coeff: "
                f"{circuit_breaker_coeff:.4f} || retain_loss: {retain_loss:.4f} "
                f"|| cb_loss: {circuit_breaker_loss:.4f}"
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


@dataclass
class BaseUnlearnConfig:
    num_train_examples: int = 1024
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    lr: float = 1e-3
    pdbs: int = 4
    alg: Literal["rr", "lat", "rr-lat"] = "rr"
    retain_coef: float = 5.0
    remove_coef: float = 5.0
    lora_r: float = 16
    adv_lr: float = 2e-3
    attack_iters: int = 8
    lora: bool = True
    layers: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30])
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_name: str = ""
    revision: str = "main"
    hidden_dim: int = 4096


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2

    parser = ArgumentParser()
    parser.add_arguments(BaseUnlearnConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"{arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision
    )

    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)

    lora_layers_to_transform = [i for i in range(max(run_cfg.layers) + 1)]

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
    )

    trainer = RRTrainer(
        run_cfg, model, training_args, train_dataset, tokenizer, run_cfg.layers
    )

    model.train()
    trainer.train()

    if run_cfg.lora:
        model = model.merge_and_unload()  # type: ignore

    if run_cfg.save_name:
        if "models/" in run_cfg.model_name:
            run_cfg.model_name = run_cfg.model_name.replace("models/", "")
        model.save_pretrained(
            f"./models/{run_cfg.model_name + '_' + run_cfg.save_name}"
        )
        tokenizer.save_pretrained(
            f"./models/{run_cfg.model_name + '_' + run_cfg.save_name}"
        )

    print("Done :)")
