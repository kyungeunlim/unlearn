# A base script for prototyping unlearning methods.
# Uses Cas's circuit breakers implementation with DDP enabled.

import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
from peft import LoraConfig, get_peft_model
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, Trainer, TrainingArguments
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import seed_worker

from unlearn.utils.hook import ActivationCapture, resolve_layer_names
from unlearn.utils.keyword_masks import apply_keyword_masks
from unlearn.utils.muon import MuonAdamW
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer, save_checkpoint


class UnlearningTrainer(Trainer):
    def __init__(
        self,
        run_args,
        model: PreTrainedModel,
        args,
        train_dataset,
        tokenizer,
        lora_target_layers,
        use_lora: bool = True,
        use_muon: bool = False,
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
        self.orth_coef = self.run_args.orth_coef
        self.trainer_tokenizer = tokenizer
        self.use_lora = use_lora
        self.use_muon = use_muon

        self.layer_id_to_name = resolve_layer_names(model, lora_target_layers)
        self.target_module_names = list(self.layer_id_to_name.values())

    def create_optimizer(self):
        if self.use_muon:
            self.optimizer = MuonAdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer
        return super().create_optimizer()

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
        orth_coeff = self.orth_coef * scheduled_coeff

        retain_mask = retain_attention_mask.unsqueeze(-1)

        # Initialize Hook Manager
        # We use unwrapped_model here to ensure names match what we resolved in __init__
        capturer = ActivationCapture(unwrapped_model, self.target_module_names)

        # --- Forward Pass 1: Reference (No Adapter) ---
        adapter_ctx = (
            unwrapped_model.disable_adapter() if self.use_lora else nullcontext()
        )  # type: ignore
        with adapter_ctx:
            unwrapped_model.eval()
            capturer.register()  # Attach hooks

            with torch.no_grad():
                ### Retain control
                if retain_coeff > 0:
                    unwrapped_model(**retain_inputs_dict)
                    orig_retain_acts = {}
                    for l in self.lora_target_layers:
                        name = self.layer_id_to_name[l]
                        orig_retain_acts[l] = (
                            capturer.activations[name].detach() * retain_mask
                        )

                capturer.clear()

                ### Circuit Breaker control
                if circuit_breaker_coeff > 0:
                    unwrapped_model(**cb_inputs_dict)
                    orig_cb_acts = {}
                    for l in self.lora_target_layers:
                        name = self.layer_id_to_name[l]
                        orig_cb_acts[l] = capturer.activations[name].detach()

            capturer.remove()  # Remove hooks

        unwrapped_model.train()

        # --- Forward Pass 2: Training (With Adapter) ---

        # Re-register hooks for the training pass
        capturer.register()

        ### Retain control
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

        ### Circuit Breaker control
        if circuit_breaker_coeff > 0:
            unwrapped_model(**cb_inputs_dict)

            denom = circuit_breaker_attention_mask.sum() * len(self.lora_target_layers)
            cb_loss_total = torch.tensor(0.0, device=target_device)

            cb_mask = circuit_breaker_attention_mask.unsqueeze(-1)  # [B, S, 1]
            if self.orth_coef > 0:
                cb_mask_sum = circuit_breaker_attention_mask.sum(
                    dim=1, keepdim=True
                ).clamp(min=1.0)
                batch_size = circuit_breaker_input_ids.shape[0]
                diag_mask = 1.0 - torch.eye(
                    batch_size, device=target_device, dtype=torch.float
                )
                orth_loss_total = torch.tensor(0.0, device=target_device)

            for l in self.lora_target_layers:
                name = self.layer_id_to_name[l]
                lora_h = capturer.activations[name]
                ref_h = orig_cb_acts[l]

                norm_lora = lora_h / torch.norm(
                    lora_h, dim=-1, keepdim=True, dtype=torch.float
                )
                norm_ref = ref_h / torch.norm(
                    ref_h, dim=-1, keepdim=True, dtype=torch.float
                )
                cos_sim = (norm_lora * norm_ref).sum(dim=-1)
                cos_sim = cos_sim * circuit_breaker_attention_mask
                cb_loss_total = cb_loss_total + torch.relu(cos_sim).sum()

                if self.orth_coef > 0:
                    masked = lora_h * cb_mask
                    pooled = masked.sum(dim=1) / cb_mask_sum
                    pooled_n = pooled / (
                        torch.norm(pooled, dim=-1, keepdim=True, dtype=torch.float)
                        + 1e-8
                    )
                    sim = pooled_n @ pooled_n.T
                    off_diag = sim * diag_mask
                    orth_loss_total = orth_loss_total + torch.relu(off_diag).sum()

            circuit_breaker_loss = cb_loss_total / (denom + 1e-6)

            if self.orth_coef > 0:
                num_pairs = batch_size * (batch_size - 1) * len(self.lora_target_layers)
                mean_seq_len = cb_mask_sum.mean()
                orth_loss = orth_loss_total / (num_pairs + 1e-6) * mean_seq_len
            else:
                orth_loss = torch.tensor(0.0, device=target_device)
        else:
            circuit_breaker_loss = 0
            orth_loss = torch.tensor(0.0, device=target_device)

        capturer.remove()  # Clean up

        loss = (
            retain_coeff * retain_loss
            + circuit_breaker_coeff * circuit_breaker_loss
            + orth_coeff * orth_loss
        )

        if (
            self.current_training_step % 32 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            print(
                f"retain_coeff: {retain_coeff:.4f} || cb_coeff: "
                f"{circuit_breaker_coeff:.4f} || orth_coeff: {orth_coeff:.4f} || "
                f"retain_loss: {retain_loss:.4f} || "
                f"cb_loss: {circuit_breaker_loss:.4f} "
                f"|| orth_loss: {orth_loss:.4f}"
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


@dataclass
class OrthCircuitBreakerConfig:
    num_train_examples: int = 1024
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    lr: float = 1e-3
    pdbs: int = 4
    alg: Literal["rr", "lat", "rr-lat"] = "rr"
    retain_coef: float = 2.0
    remove_coef: float = 23.0
    orth_coef: float = 10.0
    lora_r: int = 16
    adv_lr: float = 2e-3
    attack_iters: int = 8
    lora: bool = True
    layers: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30])
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    blocklist_path: str = ""
    keyword_mask_method: Literal["regex", "activation", "sae", "probe"] = "regex"
    activation_mask_threshold: float = 0.2
    activation_mask_layer: int = 16
    sae_latents_path: str = ""
    sae_mask_frac: float = 0.115
    probe_mask_path: str = ""
    probe_mask_layer: int = 11
    probe_mask_frac: float = 0.105
    revision: str = "main"
    hidden_dim: int = 4096
    lora_target: Literal["attn", "mlp", "all"] = "attn"
    lora_all_layers: bool = False
    exclude_lora_layers: list[int] = field(default_factory=list)
    optimizer: Literal["adamw", "muon"] = "adamw"


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2

    parser = ArgumentParser()
    parser.add_arguments(OrthCircuitBreakerConfig, dest="run_cfg")
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
        train_dataset.tokenized_bio_remove_dataset, run_cfg, tokenizer
    )

    if run_cfg.lora_all_layers:
        num_layers = model.config.num_hidden_layers
        lora_layers_to_transform = list(range(num_layers))
    else:
        lora_layers_to_transform = [i for i in range(max(run_cfg.layers) + 1)]

    if run_cfg.exclude_lora_layers:
        lora_layers_to_transform = [
            l for l in lora_layers_to_transform if l not in run_cfg.exclude_lora_layers
        ]
        print(f"Excluding LoRA layers: {run_cfg.exclude_lora_layers}")

    # Target modules based on model type and lora_target setting
    if "OLMo" in run_cfg.model_name:
        attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        mlp_modules = ["gate_proj", "up_proj", "down_proj"]
    else:
        # GPT-NeoX style
        attn_modules = ["query_key_value", "dense"]
        mlp_modules = ["dense_h_to_4h", "dense_4h_to_h"]

    if run_cfg.lora_target == "attn":
        target_modules = attn_modules
    elif run_cfg.lora_target == "mlp":
        target_modules = mlp_modules
    else:  # "all"
        target_modules = attn_modules + mlp_modules

    if run_cfg.lora:
        # Use rank-stabilized LoRA scaling (alpha=r)
        # so effective LR doesn't depend on rank
        lora_config = LoraConfig(
            r=run_cfg.lora_r,
            lora_alpha=run_cfg.lora_r,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            layers_to_transform=lora_layers_to_transform,
            task_type="CAUSAL_LM",
        )
        print(
            f"LoRA config: r={run_cfg.lora_r}, alpha={run_cfg.lora_r}, "
            f"target={run_cfg.lora_target}, "
            f"modules={target_modules}, layers={len(lora_layers_to_transform)}"
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

    output_dir = run_cfg.save_path or "./results"
    use_muon = run_cfg.optimizer == "muon"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=run_cfg.lr,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=run_cfg.pdbs,
        per_device_eval_batch_size=run_cfg.pdbs,
        num_train_epochs=1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        fp16=not use_muon,
        bf16=use_muon,
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
        use_lora=run_cfg.lora,
        use_muon=use_muon,
    )

    model.train()
    trainer.train()

    if run_cfg.lora:
        print("\nMerging LoRA weights...")
        model = model.merge_and_unload()  # type: ignore
        trainer.model = model

    if run_cfg.save_path:
        save_checkpoint(trainer, run_cfg.save_path, tokenizer)

    print("Done :)")
