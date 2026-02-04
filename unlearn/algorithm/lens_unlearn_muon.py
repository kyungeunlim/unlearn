"""Unlearning via entropy maximization at frozen tuned lens layers.

This version uses Muon optimizer for full SFT.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import seed_worker
from tuned_lens import TunedLens

from unlearn.utils.hook import ActivationCapture
from unlearn.utils.keyword_masks import apply_keyword_masks
from unlearn.utils.muon import MuonAdamW
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
        use_muon=False,
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
        self.use_muon = use_muon

    def create_optimizer(self):
        if self.use_muon:
            self.optimizer = MuonAdamW(
                self.model.parameters(),
                muon_lr=self.run_args.muon_lr,
                adam_lr=self.run_args.adam_lr,
                muon_momentum=self.run_args.muon_momentum,
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

    def __init__(
        self,
        *args,
        use_muon=False,
        reference_model=None,
        target_modules: list[str],
        model,
        **kwargs,
    ):
        super().__init__(*args, use_muon=use_muon, **kwargs)
        self.reference_model = reference_model
        self.target_modules = target_modules
        self.act_capturer = ActivationCapture(
            model, target_modules, accelerator=self.accelerator
        )
        self.act_capturer.register()

        if reference_model is not None:
            self.ref_capturer = ActivationCapture(reference_model, target_modules)
            self.ref_capturer.register()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        target_device = inputs["input_ids"].device

        retain_input_ids = inputs.get("input_ids").to(target_device)  # type: ignore
        retain_attention_mask = inputs.get("attention_mask").to(target_device)  # type: ignore
        circuit_breaker_input_ids = inputs.get("bio_remove_input_ids").to(target_device)  # type: ignore
        circuit_breaker_attention_mask = inputs.get("bio_remove_attention_mask").to(  # type: ignore
            target_device  # type: ignore
        )  # type: ignore

        # ===== Step Coeff ====
        # Recalculate global batch size for scheduling to be accurate in DDP
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        # Note: self.run_args.pdbs is per-device.

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

        model.train()

        ### Retain loss
        if retain_coeff > 0 and self.reference_model is not None:
            if self.run_args.retain_loss_type == "kl":
                with torch.no_grad():
                    ref_outputs = self.reference_model(
                        retain_input_ids, attention_mask=None
                    )
                    ref_logits = ref_outputs.logits.to(target_device)

                current_logits = model(
                    input_ids=retain_input_ids, attention_mask=None
                ).logits

                retain_loss = F.kl_div(
                    input=F.log_softmax(current_logits, dim=-1),
                    target=F.softmax(ref_logits, dim=-1),
                    reduction="batchmean",
                )
                self.act_capturer.clear()
            else:
                retain_mask = retain_attention_mask.unsqueeze(-1)

                with torch.no_grad():
                    self.reference_model(
                        input_ids=retain_input_ids,
                        attention_mask=retain_attention_mask,
                    )
                orig_retain_acts = {}
                for mod in self.target_modules:
                    if mod in self.ref_capturer.activations:
                        layer_idx = int(mod.split(".")[-1])
                        orig_retain_acts[layer_idx] = (
                            self.ref_capturer.activations[mod].detach() * retain_mask
                        )
                self.ref_capturer.clear()

                model(
                    input_ids=retain_input_ids,
                    attention_mask=retain_attention_mask,
                )

                n_layers = len(self.target_modules)
                retain_loss = torch.tensor(0.0, device=target_device)
                for mod in self.target_modules:
                    if mod in self.act_capturer.activations:
                        layer_idx = int(mod.split(".")[-1])
                        lora_h = self.act_capturer.activations[mod] * retain_mask
                        retain_loss = (
                            retain_loss
                            + torch.norm(
                                lora_h - orig_retain_acts[layer_idx],
                                dim=-1,
                                p=2,
                                dtype=torch.float,
                            ).nanmean()
                        )
                retain_loss = retain_loss / n_layers
                self.act_capturer.clear()
        else:
            retain_loss = torch.tensor(0.0, device=target_device)

        ### Forget loss - entropy maximization via tuned lens (using ActivationCapture)
        if circuit_breaker_coeff > 0:
            outputs = model(
                input_ids=circuit_breaker_input_ids,
                attention_mask=circuit_breaker_attention_mask,
            )

            # Connect loss to output for FSDP backward bookkeeping
            if hasattr(outputs, "logits"):
                dummy_loss = outputs.logits.mean() * 0.0
            elif isinstance(outputs, tuple):
                dummy_loss = outputs[0].mean() * 0.0
            else:
                dummy_loss = outputs.mean() * 0.0

            layer_losses = []
            lens_device = next(self.lens.parameters()).device  # type: ignore
            vocab_size = None
            for mod in self.target_modules:
                if mod not in self.act_capturer.activations:
                    continue

                layer_idx = int(mod.split(".")[-1])

                hidden = self.act_capturer.activations[mod]
                hidden_bf16 = hidden.to(device=lens_device, dtype=torch.bfloat16)

                lens_logits = self.lens(hidden_bf16, idx=layer_idx)
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

            self.act_capturer.clear()

            if vocab_size is None:
                vocab_size = 50257
            log_vocab = torch.log(torch.tensor(float(vocab_size), device=target_device))
            if layer_losses:
                mean_ce = torch.stack(layer_losses).mean()
                circuit_breaker_loss = mean_ce / log_vocab + dummy_loss
                mean_entropy = -mean_ce
            else:
                circuit_breaker_loss = (
                    torch.tensor(0.0, device=target_device) + dummy_loss
                )
                mean_entropy = torch.tensor(0.0)
        else:
            circuit_breaker_loss = torch.tensor(0.0, device=target_device)
            mean_entropy = torch.tensor(0.0)

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

        # Optimization: Moved heavy eval out of loop

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


@dataclass
class LensMuonUnlearnConfig:
    num_train_examples: int = 1024
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    wmdp_eval_limit: int | None = None
    mmlu_agieval_limit: int | None = None
    pdbs: int = 4
    retain_coef: float = 0.0
    remove_coef: float = 5.0
    retain_loss_type: Literal["l2", "kl"] = "kl"
    muon_lr: float = 0.02
    adam_lr: float = 3e-4
    muon_momentum: float = 0.95
    layers: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30])
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    revision: str = "main"
    lens_path: str = ""
    skip_eval: bool = False
    epochs: int = 1
    optimizer: Literal["muon", "adamw"] = "muon"


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2

    parser = ArgumentParser()
    parser.add_arguments(LensMuonUnlearnConfig, dest="run_cfg")
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

    use_muon = run_cfg.optimizer == "muon"
    if use_muon:
        print("Using full SFT with Muon optimizer")
        print(
            f"Muon LR: {run_cfg.muon_lr}, Adam LR: {run_cfg.adam_lr}, "
            f"Momentum: {run_cfg.muon_momentum}"
        )
    else:
        print("Using full SFT with AdamW optimizer")
        print(f"AdamW LR: {run_cfg.adam_lr}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Load frozen reference model for retain loss (only if retain_coef > 0)
    reference_model = None
    if run_cfg.retain_coef > 0:
        print("Loading frozen reference model for retain loss (bf16)...")
        reference_model = AutoModelForCausalLM.from_pretrained(
            run_cfg.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
        )
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
        print("Reference model loaded in bf16")

    for name, param in model.named_parameters():
        param.requires_grad = True

    model = cast(PreTrainedModel, model)
    model.enable_input_require_grads()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = 32
    grad_acc_steps = max(1, global_batch_size // (run_cfg.pdbs * world_size))

    print(
        f"Running with {world_size} GPUs. Per device batch: {run_cfg.pdbs}. "
        f"Grad Acc steps: {grad_acc_steps}."
    )

    lr = run_cfg.muon_lr if use_muon else run_cfg.adam_lr
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=lr,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=run_cfg.pdbs,
        per_device_eval_batch_size=run_cfg.pdbs,
        num_train_epochs=run_cfg.epochs,
        weight_decay=0.01,
        bf16=True,
        max_grad_norm=1.0,
        save_strategy="no",
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_version": 2,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "activation_checkpointing": True,
            "state_dict_type": "FULL_STATE_DICT",
        },
    )

    target_modules = [f"gpt_neox.layers.{i}" for i in run_cfg.layers]

    trainer = RRTrainer(
        run_cfg,
        model,
        training_args,
        train_dataset,
        tokenizer,
        run_cfg.layers,
        lens=lens,
        use_muon=use_muon,
        reference_model=reference_model,
        target_modules=target_modules,
        model=model,
    )

    model.train()
    trainer.train()

    global_rank = int(os.environ.get("RANK", 0))

    if run_cfg.save_path:
        save_checkpoint(trainer, run_cfg.save_path, tokenizer)

    if global_rank == 0:
        print("Done :)")
