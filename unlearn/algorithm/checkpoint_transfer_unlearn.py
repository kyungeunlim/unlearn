import csv
import gc
import os
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import seed_worker

from unlearn.algorithm.online_affine_fitter import (
    evaluate_affine_mse,
    load_affine_transforms,
    train_affine_transform,
    upload_affine_transforms_to_hub,
)
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
        checkpoint_model=None,
        affine_transforms=None,
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

        self.checkpoint_model = checkpoint_model
        if self.checkpoint_model is not None:
            self.checkpoint_model.eval()
            for param in self.checkpoint_model.parameters():
                param.requires_grad = False

        self.affine_transforms = affine_transforms or {}

        self.metrics_log_file = os.path.join(
            self.args.output_dir or ".", "training_metrics.csv"
        )
        self._setup_csv_logging()

    def _setup_csv_logging(self):
        """Setup CSV file for logging training metrics"""
        output_dir = (
            os.path.dirname(self.metrics_log_file)
            if os.path.dirname(self.metrics_log_file)
            else "."
        )
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(self.metrics_log_file):
            with open(self.metrics_log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "step",
                        "forget_argmax_accuracy",
                        "retain_argmax_accuracy",
                        "retain_kl_loss",
                        "circuit_breaker_loss",
                        "total_loss",
                    ]
                )
            print(f"Metrics will be logged to: {self.metrics_log_file}")

    def _log_metrics(
        self,
        step,
        forget_argmax=None,
        retain_argmax=None,
        retain_loss=None,
        cb_loss=None,
        total_loss=None,
    ):
        """Log metrics to CSV file"""
        with open(self.metrics_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [step, forget_argmax, retain_argmax, retain_loss, cb_loss, total_loss]
            )

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

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


class RRTrainer(UnlearningTrainer):

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        log_now = (
            self.current_training_step % 32 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        )

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

        # ==== Forward Inputs ====
        module = "hidden_states"
        retain_inputs_dict = dict(
            input_ids=retain_input_ids,
            attention_mask=retain_attention_mask,
            output_hidden_states=True,
        )
        cb_inputs_dict = dict(
            input_ids=circuit_breaker_input_ids,
            attention_mask=circuit_breaker_attention_mask,
            output_hidden_states=True,
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

        # ========================================================================
        # 1. RETAIN CONTROL (CE, KL divergence, or MSE on hidden states)
        # ========================================================================
        retain_loss = torch.tensor(0.0, device=target_device)
        self._last_retain_argmax = None

        if retain_coeff > 0:
            if self.run_args.retain_ce_loss:
                # Retain loss: Cross-entropy (standard LM loss)
                unwrapped_model.train()
                lora_retain_outputs = unwrapped_model(**retain_inputs_dict)
                lora_retain_logits = lora_retain_outputs.logits

                # Shift logits and labels for next-token prediction
                shift_logits = lora_retain_logits[..., :-1, :].contiguous()
                shift_labels = retain_input_ids[..., 1:].contiguous()
                shift_mask = retain_attention_mask[..., 1:].contiguous()

                # Compute per-token CE loss
                ce_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                ce_per_token = ce_loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_labels.shape)

                # Mask and average
                masked_ce = ce_per_token * shift_mask.float()
                valid_tokens = shift_mask.sum().float()
                if valid_tokens > 0:
                    retain_loss = masked_ce.sum() / valid_tokens

                del shift_logits, shift_labels, ce_per_token, masked_ce
            else:
                # Need original model outputs for KL or MSE
                with unwrapped_model.disable_adapter():
                    unwrapped_model.eval()
                    with torch.no_grad():
                        orig_retain_outputs = unwrapped_model(**retain_inputs_dict)
                        orig_retain_logits = orig_retain_outputs.logits
                        if not self.run_args.retain_kl_loss:
                            orig_retain_hidden = orig_retain_outputs[module]

                unwrapped_model.train()
                lora_retain_outputs = unwrapped_model(**retain_inputs_dict)
                lora_retain_logits = lora_retain_outputs.logits

                # Argmax matching accuracy for retain
                if log_now:
                    with torch.no_grad():
                        orig_preds = torch.argmax(orig_retain_logits, dim=-1)
                        lora_preds_retain = torch.argmax(lora_retain_logits, dim=-1)
                        matches_retain = (orig_preds == lora_preds_retain).float()
                        masked_matches_retain = (
                            matches_retain * retain_attention_mask.float()
                        )
                        valid_tokens_retain = retain_attention_mask.sum().float()
                        if valid_tokens_retain > 0:
                            retain_matching_accuracy = (
                                masked_matches_retain.sum() / valid_tokens_retain
                            )
                            self._last_retain_argmax = retain_matching_accuracy.item()

                if self.run_args.retain_kl_loss:
                    # Retain loss: KL divergence on logits
                    orig_log_probs = F.log_softmax(orig_retain_logits.float(), dim=-1)
                    lora_log_probs = F.log_softmax(lora_retain_logits.float(), dim=-1)

                    # KL divergence per token
                    kl_per_token = F.kl_div(
                        lora_log_probs,
                        orig_log_probs,
                        reduction="none",
                        log_target=True,
                    ).sum(dim=-1)

                    # Mask and average
                    masked_kl = kl_per_token * retain_attention_mask.float()
                    valid_tokens = retain_attention_mask.sum().float()
                    if valid_tokens > 0:
                        retain_loss = masked_kl.sum() / valid_tokens

                    del orig_log_probs, lora_log_probs, kl_per_token, masked_kl
                else:
                    # Retain loss: MSE on hidden states
                    lora_retain_hidden = lora_retain_outputs[module]
                    mask_expanded = retain_attention_mask.unsqueeze(-1).float()
                    valid_tokens = mask_expanded.sum()

                    mse_accumulator = 0
                    for layer_idx in self.lora_target_layers:
                        target_act = orig_retain_hidden[layer_idx].detach()
                        pred_act = lora_retain_hidden[layer_idx]

                        target_masked = (target_act * mask_expanded).float()
                        pred_masked = (pred_act * mask_expanded).float()

                        layer_mse = F.mse_loss(
                            target_masked, pred_masked, reduction="none"
                        )
                        mse_per_token = layer_mse.mean(dim=-1)
                        masked_loss = mse_per_token * mask_expanded.squeeze(-1)
                        layer_loss_sum = masked_loss.sum()

                        if valid_tokens > 0:
                            mse_accumulator += layer_loss_sum / valid_tokens

                        del target_act, pred_act, layer_mse, mse_per_token

                    retain_loss = mse_accumulator / len(self.lora_target_layers)
                    del orig_retain_hidden, lora_retain_hidden

            gc.collect()

        # ========================================================================
        # 2. CIRCUIT BREAKER CONTROL (using checkpoint model)
        # ========================================================================
        circuit_breaker_loss = torch.tensor(0.0, device=target_device)
        self._last_forget_argmax = None

        if circuit_breaker_coeff > 0 and self.checkpoint_model is not None:
            # Get activations from checkpoint model (source/target for MSE)
            with torch.no_grad():
                checkpoint_cb_outputs = self.checkpoint_model(**cb_inputs_dict)[module]
                checkpoint_logits = self.checkpoint_model(**cb_inputs_dict).logits

            # Get activations from current model (trainable)
            lora_cb_outputs = unwrapped_model(**cb_inputs_dict)[module]
            lora_logits = unwrapped_model(**cb_inputs_dict).logits

            # Argmax matching accuracy for forget
            if log_now:
                with torch.no_grad():
                    checkpoint_preds = torch.argmax(checkpoint_logits, dim=-1)
                    lora_preds = torch.argmax(lora_logits, dim=-1)
                    matches = (checkpoint_preds == lora_preds).float()
                    masked_matches = matches * circuit_breaker_attention_mask.float()
                    valid_tokens = circuit_breaker_attention_mask.sum().float()
                    if valid_tokens > 0:
                        matching_accuracy = masked_matches.sum() / valid_tokens
                        self._last_forget_argmax = matching_accuracy.item()

            # MSE loss layer-by-layer
            cb_loss_accumulator = 0
            mask_expanded = circuit_breaker_attention_mask.unsqueeze(-1).float()
            valid_tokens = mask_expanded.sum()

            for layer_idx in self.lora_target_layers:
                if self.affine_transforms and layer_idx in self.affine_transforms:
                    raw_target_act = checkpoint_cb_outputs[layer_idx].detach()
                    target_act = self.affine_transforms[layer_idx](raw_target_act)
                else:
                    target_act = checkpoint_cb_outputs[layer_idx].detach()

                pred_act = lora_cb_outputs[layer_idx]

                # Cast to float32 for stable MSE computation
                target_masked = (target_act * mask_expanded).float()
                pred_masked = (pred_act * mask_expanded).float()

                layer_mse = F.mse_loss(target_masked, pred_masked, reduction="none")
                mse_per_token = layer_mse.mean(dim=-1)
                masked_loss = mse_per_token * mask_expanded.squeeze(-1)
                layer_loss_sum = masked_loss.sum()

                if valid_tokens > 0:
                    cb_loss_accumulator += layer_loss_sum / valid_tokens

                del target_act, pred_act, layer_mse, mse_per_token

            circuit_breaker_loss = cb_loss_accumulator / len(self.lora_target_layers)

            del checkpoint_cb_outputs, lora_cb_outputs
            gc.collect()

        elif circuit_breaker_coeff > 0:
            # Fallback: use original disable_adapter method if no checkpoint_model
            broadcast_cb_mask = circuit_breaker_attention_mask.unsqueeze(0).unsqueeze(
                -1
            )
            with unwrapped_model.disable_adapter():
                unwrapped_model.eval()
                with torch.no_grad():
                    circuit_breaker_outputs = unwrapped_model(**cb_inputs_dict)[module]
                    circuit_breaker_hidden = torch.stack(
                        [
                            circuit_breaker_outputs[l].detach()
                            for l in self.lora_target_layers
                        ]
                    )
                    del circuit_breaker_outputs

            unwrapped_model.train()
            lora_circuit_breaker_outputs = unwrapped_model(**cb_inputs_dict)[module]
            lora_circuit_breaker_hidden = torch.stack(
                [lora_circuit_breaker_outputs[l] for l in self.lora_target_layers]
            )

            normalized_lora = lora_circuit_breaker_hidden / (
                torch.norm(
                    lora_circuit_breaker_hidden, dim=-1, keepdim=True, dtype=torch.float
                )
            )
            normalized_cb = circuit_breaker_hidden / (
                torch.norm(
                    circuit_breaker_hidden, dim=-1, keepdim=True, dtype=torch.float
                )
            )
            inner_product = (normalized_lora * normalized_cb) * broadcast_cb_mask
            denom = circuit_breaker_attention_mask.sum() * len(self.lora_target_layers)
            circuit_breaker_loss = torch.relu(inner_product.nansum(dim=-1)).nansum() / (
                denom + 1e-6
            )

        # ========================================================================
        # Total Loss
        # ========================================================================
        loss = retain_coeff * retain_loss + circuit_breaker_coeff * circuit_breaker_loss

        if log_now:
            retain_loss_val = (
                retain_loss.item()
                if isinstance(retain_loss, torch.Tensor)
                else retain_loss
            )
            cb_loss_val = (
                circuit_breaker_loss.item()
                if isinstance(circuit_breaker_loss, torch.Tensor)
                else circuit_breaker_loss
            )
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

            print(
                f"\nStep {self.current_training_step}: "
                f"retain_coeff: {retain_coeff:.4f} || "
                f"cb_coeff: {circuit_breaker_coeff:.4f}"
            )
            retain_type = (
                "ce"
                if self.run_args.retain_ce_loss
                else ("kl" if self.run_args.retain_kl_loss else "mse")
            )
            print(
                f"retain_loss ({retain_type}): {retain_loss_val:.4f} "
                f"|| cb_loss: {cb_loss_val:.4f}"
            )
            if self._last_retain_argmax is not None:
                print(f"retain_argmax_accuracy: {self._last_retain_argmax:.4f}")
            if self._last_forget_argmax is not None:
                print(f"forget_argmax_accuracy: {self._last_forget_argmax:.4f}")

            self._log_metrics(
                step=self.current_training_step,
                forget_argmax=self._last_forget_argmax,
                retain_argmax=self._last_retain_argmax,
                retain_loss=retain_loss_val,
                cb_loss=cb_loss_val,
                total_loss=loss_val,
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


@dataclass
class CheckpointTransferConfig:
    num_train_examples: int = 1024
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    wmdp_eval_limit: int | None = None
    mmlu_agieval_limit: int | None = None
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
    save_path: str = ""
    revision: str = "main"
    checkpoint_name: str = "EleutherAI/deep-ignorance-pretraining-stage-unfiltered"
    checkpoint_revision: str = "global_step38144"
    use_affine: bool = False
    affine_num_examples: int = 100000
    eval_affine_mse: bool = False
    affine_eval_examples: int = 10000
    upload_affine_to_hub: str | None = None
    affine_hub_private: bool = False
    load_affine_from_hub: str | None = None
    retain_kl_loss: bool = True
    retain_ce_loss: bool = False
    epochs: int = 1
    lr_warmup: bool = False
    global_batch_size: int = 32
    hidden_dim: int = 4096


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = os.cpu_count() // 2

    parser = ArgumentParser()
    parser.add_arguments(CheckpointTransferConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    SUPPORTED_MODELS = ["EleutherAI/deep-ignorance-unfiltered"]
    assert (
        run_cfg.model_name in SUPPORTED_MODELS
    ), f"model_name must be one of {SUPPORTED_MODELS}, got {run_cfg.model_name}"

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
        if "OLMo" in run_cfg.model_name:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        else:
            target_modules = [
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ]

        lora_config = LoraConfig(
            r=run_cfg.lora_r,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            layers_to_transform=lora_layers_to_transform,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
    model = cast(PreTrainedModel, model)
    model.enable_input_require_grads()

    # Load checkpoint model for checkpoint activation transfer
    checkpoint_model = None
    affine_transforms = {}
    print(
        f"Loading checkpoint model: {run_cfg.checkpoint_name} @ "
        f"{run_cfg.checkpoint_revision}"
    )
    checkpoint_model = AutoModelForCausalLM.from_pretrained(
        run_cfg.checkpoint_name,
        revision=run_cfg.checkpoint_revision,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    checkpoint_model.eval()
    for param in checkpoint_model.parameters():
        param.requires_grad = False

    # Load or train affine transforms
    if run_cfg.load_affine_from_hub:
        import tempfile

        from huggingface_hub import hf_hub_download

        print(f"Loading affine transforms from {run_cfg.load_affine_from_hub}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = hf_hub_download(
                repo_id=run_cfg.load_affine_from_hub,
                filename="affine_transforms.safetensors",
                local_dir=tmpdir,
            )
            metadata_path = hf_hub_download(
                repo_id=run_cfg.load_affine_from_hub,
                filename="metadata.json",
                local_dir=tmpdir,
            )
            affine_transforms = load_affine_transforms(tmpdir, device="cuda")

        for idx, transform in affine_transforms.items():
            affine_transforms[idx] = transform.to(
                device=model.device if hasattr(model, "device") else "cuda",
                dtype=torch.float16,
            )
            affine_transforms[idx].requires_grad_(False)
        print(f"Loaded affine transforms for layers: {list(affine_transforms.keys())}")

    elif run_cfg.use_affine:
        print("Training affine transforms...")
        affine_transforms = train_affine_transform(
            source_model=checkpoint_model,
            target_model=model,
            dataset=train_dataset,
            target_layers=run_cfg.layers,
            num_examples=run_cfg.affine_num_examples,
            device=model.device if hasattr(model, "device") else "cuda",
        )
        for idx, transform in affine_transforms.items():
            affine_transforms[idx] = transform.to(
                model.device if hasattr(model, "device") else "cuda"
            )
            affine_transforms[idx].requires_grad_(False)

        # Evaluate and optionally upload affine transforms
        mse_metrics = None
        if run_cfg.eval_affine_mse or run_cfg.upload_affine_to_hub:
            mse_metrics = evaluate_affine_mse(
                affine_transforms=affine_transforms,
                source_model=checkpoint_model,
                target_model=model,
                dataset=train_dataset,
                target_layers=run_cfg.layers,
                num_examples=run_cfg.affine_eval_examples,
                device=model.device if hasattr(model, "device") else "cuda",
            )
            for layer_key, mse_val in mse_metrics.items():
                print(f"  {layer_key}: {mse_val:.6f}")

        if run_cfg.upload_affine_to_hub:
            upload_affine_transforms_to_hub(
                affine_transforms=affine_transforms,
                repo_id=run_cfg.upload_affine_to_hub,
                mse_metrics=mse_metrics,
                source_model_name=run_cfg.checkpoint_name,
                target_model_name=run_cfg.model_name,
                alpha=0.01,
                num_training_examples=run_cfg.affine_num_examples,
                private=run_cfg.affine_hub_private,
            )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = run_cfg.global_batch_size
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
        fp16=True,
        save_strategy="no",
        ddp_find_unused_parameters=False,
        warmup_steps=10 if run_cfg.lr_warmup else 0,
    )

    trainer = RRTrainer(
        run_cfg,
        model,
        training_args,
        train_dataset,
        tokenizer,
        run_cfg.layers,
        checkpoint_model=checkpoint_model,
        affine_transforms=affine_transforms,
    )

    model.train()
    trainer.train()

    if run_cfg.lora:
        model = model.merge_and_unload()

    if run_cfg.save_path:
        save_checkpoint(trainer, run_cfg.save_path, tokenizer)

    print("Done :)")
