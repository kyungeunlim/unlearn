"""Unlearning via entropy maximization at frozen tuned lens layers."""

import os
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module
from peft import LoraConfig, get_peft_model
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import seed_worker
from tuned_lens import TunedLens

from unlearn.utils.hook import ActivationCapture, resolve_layer_names
from unlearn.utils.keyword_masks import apply_keyword_masks
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer, save_checkpoint


class UnlearningTrainer(Trainer):
    def __init__(
        self,
        run_args,
        model,
        args,
        train_dataset,
        tokenizer,
        lora_target_layers,
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
        self.lora_target_layers = lora_target_layers
        self.model = model
        self.retain_coef = self.run_args.retain_coef
        self.remove_coef = self.run_args.remove_coef
        self.trainer_tokenizer = tokenizer
        self.lens = lens

        self.layer_id_to_name = resolve_layer_names(model, lora_target_layers)
        self.target_module_names = list(self.layer_id_to_name.values())

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
        self, model, inputs, return_outputs=False, num_items_in_batch=None
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
        )  # type: ignore

        # ==== Forward Inputs ====
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

        retain_mask = retain_attention_mask.unsqueeze(-1)

        capturer = ActivationCapture(unwrapped_model, self.target_module_names)

        # --- Forward Pass 1: Reference (No Adapter) ---
        with unwrapped_model.disable_adapter():  # type: ignore
            unwrapped_model.eval()  # type: ignore
            capturer.register()

            with torch.no_grad():
                if retain_coeff > 0:
                    unwrapped_model(**retain_inputs_dict)
                    orig_retain_acts = {}
                    for l in self.lora_target_layers:
                        name = self.layer_id_to_name[l]
                        orig_retain_acts[l] = (
                            capturer.activations[name].detach() * retain_mask
                        )

            capturer.remove()

        unwrapped_model.train()  # type: ignore

        # --- Forward Pass 2: Training (With Adapter) ---
        capturer.register()

        ### Retain L2 loss
        if retain_coeff > 0:
            unwrapped_model(**retain_inputs_dict)

            n_layers = len(self.lora_target_layers)
            retain_loss = torch.tensor(0.0, device=target_device)
            for l in self.lora_target_layers:
                name = self.layer_id_to_name[l]
                lora_h = capturer.activations[name] * retain_mask
                retain_loss = (
                    retain_loss
                    + torch.norm(
                        lora_h - orig_retain_acts[l], dim=-1, p=2, dtype=torch.float
                    ).nanmean()
                )
            retain_loss = retain_loss / n_layers

            capturer.clear()
        else:
            retain_loss = 0

        ### Forget loss - entropy maximization via tuned lens
        if circuit_breaker_coeff > 0:
            unwrapped_model(**cb_inputs_dict)

            layer_losses = []
            lens_device = next(self.lens.parameters()).device
            for layer_idx in self.lora_target_layers:
                hidden = capturer.activations[
                    self.layer_id_to_name[layer_idx]
                ]  # [batch, seq, hidden]
                hidden_bf16 = hidden.to(device=lens_device, dtype=torch.bfloat16)

                lens_logits = self.lens(
                    hidden_bf16, idx=layer_idx
                )  # [batch, seq, vocab]
                batch_size, seq_len, vocab_size = lens_logits.shape

                random_targets = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=lens_logits.device
                )

                mask = circuit_breaker_attention_mask.bool()
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
                circuit_breaker_loss = mean_ce / log_vocab
                mean_entropy = -mean_ce
            else:
                circuit_breaker_loss = torch.tensor(0.0, device=target_device)
                mean_entropy = torch.tensor(0.0)
        else:
            circuit_breaker_loss = torch.tensor(0.0, device=target_device)
            mean_entropy = torch.tensor(0.0)

        capturer.remove()

        loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss

        if (
            self.current_training_step % 32 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            entropy_val = (
                mean_entropy.item()
                if isinstance(mean_entropy, torch.Tensor)
                else mean_entropy
            )
            print(
                f"retain_coeff: {retain_coeff:.4f} || "
                f"forget_coeff: {circuit_breaker_coeff:.4f} || "
                f"retain_loss: {retain_loss:.4f} || "
                f"forget_loss: {circuit_breaker_loss:.4f} || "
                f"mean_entropy: {entropy_val:.4f}"
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


@dataclass
class LensUnlearnConfig:
    num_train_examples: int = 1024
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    wmdp_eval_limit: int | None = None
    mmlu_agieval_limit: int | None = None
    lr: float = 1e-3
    pdbs: int = 4
    retain_coef: float = 5.0
    remove_coef: float = 5.0
    lora_r: float = 16
    lora: bool = True
    layers: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30])
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    revision: str = "main"
    lens_path: str = ""
    skip_eval: bool = False
    epochs: int = 1


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2

    parser = ArgumentParser()
    parser.add_arguments(LensUnlearnConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    if "smollm2" in run_cfg.model_name:
        run_cfg.layers = [l for l in run_cfg.layers if l < 24]

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"{arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision
    )
    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)
    train_dataset.tokenized_bio_remove_dataset = apply_keyword_masks(
        train_dataset.tokenized_bio_remove_dataset, run_cfg, tokenizer
    )

    # Load frozen tuned lens
    print(f"Loading tuned lens from: {run_cfg.lens_path}")
    device = next(model.parameters()).device
    lens = TunedLens.from_model(model, bias=True)
    lens_state_dict = torch.load(f"{run_cfg.lens_path}/params.pt", map_location=device)

    # Map saved keys (e.g., "0.weight") to expected keys
    # (e.g., "layer_translators.0.weight")
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

    # Copy unembed params from initialized lens
    current_state = lens.state_dict()
    for key in current_state:
        if key.startswith("unembed"):
            mapped_state_dict[key] = current_state[key]

    lens.load_state_dict(mapped_state_dict)

    # Remove accelerate hooks from lens submodules (they get copied via deepcopy)
    for submodule in lens.modules():
        remove_hook_from_module(submodule)

    lens = lens.to(device=device, dtype=torch.bfloat16)
    lens.eval()
    for param in lens.parameters():
        param.requires_grad = False
    print(f"Loaded lens with {len(lens)} layer translators (frozen)")

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
        num_train_epochs=run_cfg.epochs,
        weight_decay=0.01,
        gradient_checkpointing=True,
        bf16=True,
        max_grad_norm=1.0,
        save_strategy="no",
        ddp_find_unused_parameters=False,
    )

    trainer = RRTrainer(
        run_cfg,
        model,
        training_args,
        train_dataset,
        tokenizer,
        run_cfg.layers,
        lens=lens,
    )

    model.train()
    trainer.train()

    if run_cfg.lora:
        model = model.merge_and_unload()  # type: ignore

    if run_cfg.save_path:
        save_checkpoint(trainer, run_cfg.save_path, tokenizer)

    print("Done :)")
