"""Sequential back-to-front unlearning (SFT - full parameter training with FSDP)."""

import math
import os
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import seed_worker

import wandb
from unlearn.utils.keyword_masks import apply_keyword_masks
from unlearn.utils.math import max_entropy_kl_loss, top_k_entropy_loss
from unlearn.utils.muon import MuonAdamW
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer, save_checkpoint


class SequentialSftTrainer(Trainer):

    def __init__(
        self,
        run_args,
        model,
        args,
        train_dataset,
        tokenizer,
        layers_to_unlearn,
        frozen_model,
        steps_per_phase,
        **kwargs,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset,
        )
        self.run_args = run_args
        self.tokenizer = tokenizer
        self.layers_to_unlearn = layers_to_unlearn
        self.frozen_model = frozen_model
        if isinstance(steps_per_phase, list):
            self.phase_steps = steps_per_phase
        else:
            self.phase_steps = [steps_per_phase] * len(layers_to_unlearn)
        self.phase_boundaries = []
        cumulative = 0
        for s in self.phase_steps:
            cumulative += s
            self.phase_boundaries.append(cumulative)
        self.current_training_step = 0

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
        )  # type: ignore

    def create_optimizer(self):
        if self.run_args.optimizer == "muon":
            self.optimizer = MuonAdamW(
                self.model.parameters(),
                lr=self.run_args.lr,
                muon_momentum=self.run_args.muon_momentum,
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer
        return super().create_optimizer()

    def _get_phase_info(self):
        """Handles scheduled information, including the current targeted layer
        and the retain and remove coefficients"""
        phase_idx = len(self.layers_to_unlearn) - 1
        for i, boundary in enumerate(self.phase_boundaries):
            if self.current_training_step < boundary:
                phase_idx = i
                break
        target_layer = self.layers_to_unlearn[phase_idx]

        world_size = int(os.environ.get("WORLD_SIZE", 1))
        phase_start = self.phase_boundaries[phase_idx - 1] if phase_idx > 0 else 0
        phase_step = self.current_training_step - phase_start
        scheduled_coeff = min(
            1.0,
            phase_step
            / (self.run_args.num_train_examples / (self.run_args.pdbs * world_size)),
        )
        if self.run_args.ramp_retain_from_zero:
            retain_coeff = self.run_args.retain_coef * scheduled_coeff
        else:
            retain_coeff = self.run_args.retain_coef * (0.25 + 0.75 * scheduled_coeff)
        forget_coeff = self.run_args.remove_coef * (1 - 0.25 * scheduled_coeff)

        if self.run_args.forget_layer_scale > 1.0:
            start_layer = self.layers_to_unlearn[0]
            depth_ratio = (start_layer - target_layer) / max(start_layer, 1)
            layer_scale = 1.0 + (self.run_args.forget_layer_scale - 1.0) * depth_ratio
            forget_coeff *= layer_scale

        return phase_idx, target_layer, retain_coeff, forget_coeff

    def _compute_retain_loss(self, model, inputs, target_device):
        """The KL divergence of the updated model's predictions on retain data
        from the original model's predictions."""
        retain_input_ids = inputs["input_ids"].to(target_device)
        with torch.no_grad():
            ref_outputs = self.frozen_model(retain_input_ids, attention_mask=None)
            ref_logits = ref_outputs.logits.to(target_device)

        current_logits = model(
            input_ids=retain_input_ids,
            attention_mask=None,
        ).logits

        return F.kl_div(
            input=F.log_softmax(current_logits, dim=-1),
            target=F.softmax(ref_logits, dim=-1),
            reduction="batchmean",
        )

    def _compute_nll_retain_loss(self, model, inputs, target_device):
        """Standard cross-entropy on retain tokens with teacher forcing."""
        retain_input_ids = inputs["input_ids"].to(target_device)

        logits = model(
            input_ids=retain_input_ids,
            attention_mask=None,
        ).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = retain_input_ids[:, 1:].contiguous()

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            mask = shift_labels != pad_token_id
            shift_logits = shift_logits[mask]
            shift_labels = shift_labels[mask]

        return F.cross_entropy(shift_logits.float(), shift_labels, reduction="mean")

    def _compute_l2_retain_loss(self, model, inputs, target_layers, target_device):
        """L2 distance between current and original hidden states,
        averaged over layers."""
        if isinstance(target_layers, int):
            target_layers = [target_layers]
        retain_input_ids = inputs["input_ids"].to(target_device)

        with torch.no_grad():
            ref_outputs = self.frozen_model(
                retain_input_ids, attention_mask=None, output_hidden_states=True
            )

        current_outputs = model(
            input_ids=retain_input_ids,
            attention_mask=None,
            output_hidden_states=True,
        )

        # hidden_states[L+1] is the output of layer L
        total_loss = torch.tensor(0.0, device=target_device)
        for layer in target_layers:
            ref_h = ref_outputs.hidden_states[layer + 1].to(target_device)
            cur_h = current_outputs.hidden_states[layer + 1]
            total_loss = (
                total_loss
                + torch.norm(cur_h - ref_h, dim=-1, p=2, dtype=torch.float).mean()
            )
        return total_loss / len(target_layers)

    def _compute_forget_loss(self, model, inputs, target_layer, target_device):
        """Forget loss using the model's actual output logits.

        Gradient flows through the full model, but _freeze_and_log zeroes
        non-target-layer gradients so only the target layer is updated.
        """
        forget_input_ids = inputs["bio_remove_input_ids"].to(target_device)
        forget_attention_mask = inputs["bio_remove_attention_mask"].to(target_device)

        outputs = model(
            input_ids=forget_input_ids,
            attention_mask=forget_attention_mask,
        )
        logits = outputs.logits

        mask = forget_attention_mask.bool()
        if "bio_remove_keyword_mask" in inputs:
            keyword_mask = inputs["bio_remove_keyword_mask"].bool().to(target_device)
            mask = mask & keyword_mask
        logits_masked = logits[mask]

        if logits_masked.numel() == 0:
            return logits.mean() * 0.0

        if self.run_args.use_dpo_forget:
            orig_device = next(self.frozen_model.parameters()).device
            with torch.no_grad():
                ref_logits = self.frozen_model(
                    forget_input_ids.to(orig_device),
                    attention_mask=forget_attention_mask.to(orig_device),
                ).logits.to(target_device)

            shift_labels = forget_input_ids[:, 1:]
            shift_mask = forget_attention_mask[:, 1:].bool()
            total_tokens = shift_mask.sum().clamp(min=1)
            beta = self.run_args.dpo_beta

            shift_current = logits[:, :-1, :].float()
            shift_ref = ref_logits[:, :-1, :].float()

            current_lp = F.log_softmax(shift_current, dim=-1)
            ref_lp = F.log_softmax(shift_ref, dim=-1)

            current_token_lp = current_lp.gather(
                -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            ref_token_lp = ref_lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

            if self.run_args.dpo_per_token:
                per_token_npo = -F.logsigmoid(-beta * (current_token_lp - ref_token_lp))
                loss = (per_token_npo * shift_mask).sum() / total_tokens
            else:
                seq_lengths = shift_mask.sum(dim=-1).clamp(min=1)
                token_log_ratios = (current_token_lp - ref_token_lp) * shift_mask
                avg_log_ratios = token_log_ratios.sum(dim=-1) / seq_lengths
                loss = -F.logsigmoid(-beta * avg_log_ratios).mean()

            return loss

        elif self.run_args.use_top_k_entropy:
            k = self.run_args.top_k
            log_k = torch.log(torch.tensor(float(k), device=target_device))
            return top_k_entropy_loss(logits_masked, k=k) / log_k

        elif self.run_args.use_max_entropy_kl:
            vocab_size = logits.shape[-1]
            log_vocab = torch.log(torch.tensor(float(vocab_size), device=target_device))
            return max_entropy_kl_loss(logits_masked) / log_vocab
        else:
            vocab_size = logits.shape[-1]
            batch_size, seq_len = logits.shape[:2]
            random_targets = torch.randint(
                0, vocab_size, (batch_size, seq_len), device=logits.device
            )
            targets_flat = random_targets[mask]
            ce_loss = F.cross_entropy(
                logits_masked.float(), targets_flat, reduction="mean"
            )
            log_vocab = torch.log(torch.tensor(float(vocab_size), device=target_device))
            return ce_loss / log_vocab

    def _freeze_and_log(
        self, model, current_step, target_layer, keep_layers=None, extra=""
    ):
        if keep_layers is None:
            keep_layers = {target_layer}
        keep_strs = {f".layers.{l}." for l in keep_layers}
        grad_params = 0
        frozen_params = 0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if any(s in name for s in keep_strs):
                grad_params += 1
            else:
                param.grad = None
                frozen_params += 1

        if current_step % 8 == 0 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            maintain_str = (
                f" | maintaining: {len(keep_layers) - 1}"
                if len(keep_layers) > 1
                else ""
            )
            print(
                f"  [grad freeze] step {current_step} | target layer {target_layer} | "
                f"params with grad: {grad_params} | frozen: {frozen_params}"
                f"{maintain_str}" + extra
            )

    def _compute_grad_norm(self, model, target_str):
        """Compute L2 gradient norm for parameters matching target_str."""
        total_norm_sq = torch.tensor(0.0, device="cuda")
        for name, param in model.named_parameters():
            if param.grad is not None and target_str in name:
                total_norm_sq += param.grad.detach().float().pow(2).sum()
        return total_norm_sq.sqrt().item()

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        target_device = inputs["input_ids"].device

        phase_idx, target_layer, retain_coeff, forget_coeff = self._get_phase_info()
        target_str = f".layers.{target_layer}."

        # Determine layers to keep gradients for (target + already-unlearned)
        if self.run_args.maintain_unlearned and phase_idx > 0:
            already_unlearned = self.layers_to_unlearn[:phase_idx]
            keep_layers = set(already_unlearned) | {target_layer}
        else:
            keep_layers = {target_layer}
        keep_strs = {f".layers.{l}." for l in keep_layers}

        # L2-SP: register hook to capture penalty during retain forward.
        # During forward, FSDP gathers params so the hook sees original shapes.
        l2sp_loss = torch.tensor(0.0, device=target_device)
        l2sp_norm = 0.0
        l2sp_handle = None
        if self.run_args.l2sp_coef > 0:
            fsdp_root = model.module if hasattr(model, "module") else model
            frozen_layer = self.frozen_model.gpt_neox.layers[target_layer]
            frozen_params = dict(frozen_layer.named_parameters())
            l2sp_terms: list[torch.Tensor] = []

            def _l2sp_hook(module, input, output):
                for name, param in module.named_parameters():
                    if name in frozen_params:
                        w0 = (
                            frozen_params[name]
                            .detach()
                            .to(device=param.device, dtype=param.dtype)
                        )
                        l2sp_terms.append((param - w0).pow(2).sum())
                return output

            target_module = fsdp_root.gpt_neox.layers[target_layer]
            l2sp_handle = target_module.register_forward_hook(_l2sp_hook)

        # Retain backward (+ L2-SP if enabled): save target layer grads, then zero
        if self.run_args.retain_loss_type == "nll":
            retain_loss = self._compute_nll_retain_loss(model, inputs, target_device)
        elif self.run_args.retain_loss_type == "l2":
            retain_loss = self._compute_l2_retain_loss(
                model, inputs, sorted(keep_layers), target_device
            )
        else:
            retain_loss = self._compute_retain_loss(model, inputs, target_device)
        if l2sp_handle is not None:
            l2sp_handle.remove()
            if l2sp_terms:
                l2sp_loss = self.run_args.l2sp_coef * torch.stack(l2sp_terms).sum()
                l2sp_norm = l2sp_loss.item()
        self.accelerator.backward(retain_coeff * retain_loss + l2sp_loss)
        retain_grad_norm = self._compute_grad_norm(model, target_str)
        retain_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None and any(s in name for s in keep_strs):
                retain_grads[name] = param.grad.clone()
        model.zero_grad()

        # Forget backward
        forget_loss = self._compute_forget_loss(
            model, inputs, target_layer, target_device
        )
        self.accelerator.backward(forget_coeff * forget_loss)
        forget_grad_norm = self._compute_grad_norm(model, target_str)

        # Combine gradients
        same_sign_pct = None
        if self.run_args.same_sign_grads:
            agreed_elements = 0
            total_elements = 0
            for name, param in model.named_parameters():
                if (
                    param.grad is not None
                    and any(s in name for s in keep_strs)
                    and name in retain_grads
                ):
                    r = retain_grads[name].to(param.grad.dtype)
                    same_sign = (r * param.grad) > 0
                    agreed_elements += same_sign.sum().item()
                    total_elements += same_sign.numel()
                    if self.run_args.asymmetric_filter:
                        param.grad.mul_(same_sign)
                        param.grad.add_(r)
                    else:
                        param.grad.add_(r)
                        param.grad.mul_(same_sign)

            stats = torch.tensor(
                [agreed_elements, total_elements],
                device=target_device,
                dtype=torch.long,
            )
            if dist.is_initialized():
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            same_sign_pct = stats[0].item() / max(stats[1].item(), 1) * 100
        else:
            for name, param in model.named_parameters():
                if (
                    param.grad is not None
                    and any(s in name for s in keep_strs)
                    and name in retain_grads
                ):
                    param.grad.add_(retain_grads[name].to(param.grad.dtype))

        # Freeze non-target layers
        extra = (
            f" | same_sign: {same_sign_pct:.0f}%" if same_sign_pct is not None else ""
        )
        self._freeze_and_log(
            model,
            self.current_training_step,
            target_layer,
            keep_layers=keep_layers,
            extra=extra,
        )

        # Keyword mask stats
        keyword_mask_frac = None
        if "bio_remove_keyword_mask" in inputs:
            km = inputs["bio_remove_keyword_mask"]
            attn = inputs["bio_remove_attention_mask"]
            attn_tokens = attn.sum().item()
            keyword_tokens = (km.bool() & attn.bool()).sum().item()
            keyword_mask_frac = keyword_tokens / max(attn_tokens, 1)

        # Logging
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            if self.current_training_step % 8 == 0:
                msg = (
                    f"step {self.current_training_step} | "
                    f"layer {target_layer} "
                    f"(phase {phase_idx + 1}/{len(self.layers_to_unlearn)}) | "
                    f"retain_loss: {retain_loss.item():.4f} | "
                    f"forget_loss: {forget_loss.item():.4f} | "
                    f"retain_grad_norm: {retain_grad_norm:.4f} | "
                    f"forget_grad_norm: {forget_grad_norm:.4f}"
                )
                if same_sign_pct is not None:
                    msg += f" | same_sign: {same_sign_pct:.1f}%"
                if l2sp_norm > 0:
                    msg += f" | l2sp: {l2sp_norm:.4f}"
                if keyword_mask_frac is not None:
                    msg += f" | kw_mask: {keyword_mask_frac:.3f}"
                print(msg)

            if wandb.run is not None:
                log_dict = {
                    "retain_loss": retain_loss.item(),
                    "forget_loss": forget_loss.item(),
                    "retain_grad_norm": retain_grad_norm,
                    "forget_grad_norm": forget_grad_norm,
                    "retain_coeff": retain_coeff,
                    "forget_coeff": forget_coeff,
                    "target_layer": target_layer,
                    "phase": phase_idx,
                }
                if same_sign_pct is not None:
                    log_dict["same_sign_pct"] = same_sign_pct
                if l2sp_norm > 0:
                    log_dict["l2sp_norm"] = l2sp_norm
                if keyword_mask_frac is not None:
                    log_dict["keyword_mask_frac"] = keyword_mask_frac
                wandb.log(log_dict, step=self.current_training_step)

        self.current_training_step += 1
        return (retain_coeff * retain_loss + forget_coeff * forget_loss).detach()


@dataclass
class SequentialSftUnlearnConfig:
    num_train_examples: int = 1024
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    lr: float = 1e-4
    pdbs: int = 1
    retain_coef: float = 20.0
    remove_coef: float = 5.0
    start_layer: int | None = None
    end_layer: int = 8
    layer_step: int = 4
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    epochs_per_layer: int = 1
    warmup_ratio: float = 0.0
    use_ultrachat: bool = True
    use_max_entropy_kl: bool = False
    use_top_k_entropy: bool = False
    top_k: int = 100
    same_sign_grads: bool = False
    asymmetric_filter: bool = False
    ramp_retain_from_zero: bool = False
    l2sp_coef: float = 0.0
    max_grad_norm: float = 1.0
    retain_loss_type: Literal["kl", "l2", "nll"] = "kl"
    use_dpo_forget: bool = False
    dpo_per_token: bool = False
    dpo_beta: float = 0.1
    forget_layer_scale: float = 1.0
    deep_layer_step_scale: float = 1.0
    optimizer: Literal["muon", "adamw"] = "adamw"
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    wandb_project: str = ""
    blocklist_path: str = ""
    keyword_mask_method: Literal["regex", "activation", "sae", "probe"] = "regex"
    activation_mask_threshold: float = 0.2
    activation_mask_layer: int = 16
    sae_latents_path: str = ""
    sae_mask_frac: float = 0.115
    probe_mask_path: str = ""
    probe_mask_layer: int = 11
    probe_mask_frac: float = 0.105
    gradient_checkpointing: bool = False
    maintain_unlearned: bool = False


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    parser = ArgumentParser()
    parser.add_arguments(SequentialSftUnlearnConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"  {arg}: {value}")

    if run_cfg.wandb_project and local_rank == 0:
        wandb.init(project=run_cfg.wandb_project, config=vars(run_cfg))

    model, tokenizer = get_model_and_tokenizer(run_cfg.model_name)
    model = cast(PreTrainedModel, model)

    for param in model.parameters():
        param.requires_grad = True
    model.enable_input_require_grads()

    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)
    train_dataset.tokenized_bio_remove_dataset = apply_keyword_masks(
        train_dataset.tokenized_bio_remove_dataset, run_cfg, tokenizer
    )

    frozen_model = AutoModelForCausalLM.from_pretrained(
        run_cfg.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
    )
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad = False
    print("Loaded frozen reference model.")

    num_layers = model.config.num_hidden_layers
    if run_cfg.start_layer is None:
        run_cfg.start_layer = num_layers - 1

    layers_to_unlearn = list(
        range(run_cfg.start_layer, run_cfg.end_layer - 1, -run_cfg.layer_step)
    )
    print(f"Layers to unlearn (back-to-front): {layers_to_unlearn}")

    global_batch_size = 32
    grad_acc_steps = max(1, global_batch_size // (run_cfg.pdbs * world_size))
    base_steps_per_phase = max(
        1,
        run_cfg.epochs_per_layer * len(train_dataset) // (world_size * run_cfg.pdbs),
    )

    if run_cfg.deep_layer_step_scale > 1.0:
        phase_steps_list = []
        for layer in layers_to_unlearn:
            depth_ratio = (layers_to_unlearn[0] - layer) / max(layers_to_unlearn[0], 1)
            scale = 1.0 + (run_cfg.deep_layer_step_scale - 1.0) * depth_ratio
            phase_steps_list.append(max(1, int(base_steps_per_phase * scale)))
    else:
        phase_steps_list = [base_steps_per_phase] * len(layers_to_unlearn)

    total_training_calls = sum(phase_steps_list)
    steps_per_epoch = len(train_dataset) // (world_size * run_cfg.pdbs)
    total_epochs = math.ceil(total_training_calls / steps_per_epoch)

    use_muon = run_cfg.optimizer == "muon"

    print(
        f"Running with {world_size} GPUs. Per device batch: {run_cfg.pdbs}. "
        f"Grad Acc steps: {grad_acc_steps}. "
        f"Steps per phase: {phase_steps_list}. Total epochs: {total_epochs}."
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=run_cfg.lr,
        warmup_ratio=run_cfg.warmup_ratio,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=run_cfg.pdbs,
        per_device_eval_batch_size=run_cfg.pdbs,
        num_train_epochs=total_epochs,
        weight_decay=0.01,
        gradient_checkpointing=run_cfg.gradient_checkpointing,
        bf16=True,
        max_grad_norm=run_cfg.max_grad_norm,
        save_strategy="no",
        report_to="none",
    )

    trainer = SequentialSftTrainer(
        run_args=run_cfg,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        layers_to_unlearn=layers_to_unlearn,
        frozen_model=frozen_model,
        steps_per_phase=phase_steps_list,
    )

    model.train()
    trainer.train()

    if run_cfg.save_path:
        save_checkpoint(trainer, run_cfg.save_path, tokenizer)

    print("Training complete")
