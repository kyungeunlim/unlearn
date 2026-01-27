import argparse
import csv
import gc
import os
from typing import cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
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
from unlearn.utils.worker_utils import get_model_and_tokenizer


def unwrap_model(model):
    """Get the underlying model from DDP/FSDP wrapper if present."""
    if hasattr(model, "module"):
        return model.module
    return model


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
                        lora_log_probs, orig_log_probs, reduction="none", log_target=True
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

                        layer_mse = F.mse_loss(target_masked, pred_masked, reduction="none")
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
            retain_type = "ce" if self.run_args.retain_ce_loss else ("kl" if self.run_args.retain_kl_loss else "mse")
            print(
                f"retain_loss ({retain_type}): {retain_loss_val:.4f} || cb_loss: {cb_loss_val:.4f}"
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


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    # Optimization: Utilize half of the CPU cores for dataset mapping
    NUM_PROC = os.cpu_count() // 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_examples", type=int, default=1024)
    parser.add_argument("--unlearn_corrupt", type=bool, default=False)
    parser.add_argument("--corrupt_ratio", type=float, default=0.5)
    parser.add_argument(
        "--corrupt_ds", type=str, default="rewritten", choices=["rewritten", "shuffled"]
    )
    parser.add_argument("--wmdp_eval_limit", type=int, default=None)
    parser.add_argument("--mmlu_agieval_limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pdbs", type=int, default=None)
    parser.add_argument(
        "--alg", type=str, choices=["rr", "lat", "rr-lat"], default="rr"
    )
    parser.add_argument(
        "--retain_coef",
        type=float,
        default=None,
        help="Coefficient for KL divergence retain loss on logits",
    )
    parser.add_argument("--remove_coef", type=float, default=None)
    parser.add_argument("--lora_r", type=float, default=16)
    parser.add_argument("--adv_lr", type=float, default=2e-3)
    parser.add_argument("--attack_iters", type=int, default=8)
    parser.add_argument("--lora", type=bool, default=True)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 30],
        help="List of layers to target",
    )
    parser.add_argument(
        "--model_name", type=str, default="EleutherAI/deep-ignorance-unfiltered"
    )
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="EleutherAI/deep-ignorance-pretraining-stage-unfiltered",
        help="HF model name for checkpoint model to transfer from",
    )
    parser.add_argument(
        "--checkpoint_revision",
        type=str,
        default="global_step38144",
        help="Revision/commit of checkpoint model",
    )
    parser.add_argument(
        "--use_affine",
        action="store_true",
        help="Use affine transform between checkpoint and target model",
    )
    parser.add_argument(
        "--affine_num_examples",
        type=int,
        default=100000,
        help="Number of examples for affine transform training",
    )
    parser.add_argument(
        "--eval_affine_mse",
        action="store_true",
        help="Evaluate MSE of affine transforms after training",
    )
    parser.add_argument(
        "--affine_eval_examples",
        type=int,
        default=10000,
        help="Number of examples for affine MSE evaluation",
    )
    parser.add_argument(
        "--upload_affine_to_hub",
        type=str,
        default=None,
        help="HuggingFace repo name to upload affine transforms (e.g., 'username/affine-map')",
    )
    parser.add_argument(
        "--affine_hub_private",
        action="store_true",
        help="Make the HuggingFace repo private",
    )
    parser.add_argument(
        "--load_affine_from_hub",
        type=str,
        default=None,
        help="Load pre-trained affine transforms from HuggingFace repo (e.g., 'EleutherAI/affine-checkpoint-transfer')",
    )
    parser.add_argument(
        "--retain_kl_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use KL divergence on outputs for retain loss (default: True). Use --no_retain_kl_loss for MSE on hidden states.",
    )
    parser.add_argument(
        "--retain_ce_loss",
        action="store_true",
        help="Use cross-entropy loss on retain data (standard LM loss). Overrides --retain_kl_loss.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--lr_warmup",
        action="store_true",
        help="Enable learning rate warmup of 10 steps (default: off)",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=32,
        help="Global batch size across all GPUs (default: 32)",
    )

    args = parser.parse_args()

    SUPPORTED_MODELS = ["EleutherAI/deep-ignorance-unfiltered"]
    assert (
        args.model_name in SUPPORTED_MODELS
    ), f"model_name must be one of {SUPPORTED_MODELS}, got {args.model_name}"

    args.pdbs = 4 if args.pdbs is None else args.pdbs
    args.retain_coef = 5 if args.retain_coef is None else args.retain_coef
    args.remove_coef = 5 if args.remove_coef is None else args.remove_coef
    args.hidden_dim = 4096

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(args.model_name, revision=args.revision)
    train_dataset = get_unlearning_dataset(args, tokenizer, NUM_PROC)

    lora_layers_to_transform = [i for i in range(max(args.layers) + 1)]

    if args.lora:
        if "OLMo" in args.model_name:
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
            r=args.lora_r,
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
        f"Loading checkpoint model: {args.checkpoint_name} @ {args.checkpoint_revision}"
    )
    checkpoint_model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_name,
        revision=args.checkpoint_revision,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    checkpoint_model.eval()
    for param in checkpoint_model.parameters():
        param.requires_grad = False

    # Load or train affine transforms
    if args.load_affine_from_hub:
        from huggingface_hub import hf_hub_download
        import tempfile

        print(f"Loading affine transforms from {args.load_affine_from_hub}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = hf_hub_download(
                repo_id=args.load_affine_from_hub,
                filename="affine_transforms.safetensors",
                local_dir=tmpdir,
            )
            metadata_path = hf_hub_download(
                repo_id=args.load_affine_from_hub,
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

    elif args.use_affine:
        print("Training affine transforms...")
        affine_transforms = train_affine_transform(
            source_model=checkpoint_model,
            target_model=model,
            dataset=train_dataset,
            target_layers=args.layers,
            num_examples=args.affine_num_examples,
            device=model.device if hasattr(model, "device") else "cuda",
        )
        for idx, transform in affine_transforms.items():
            affine_transforms[idx] = transform.to(
                model.device if hasattr(model, "device") else "cuda"
            )
            affine_transforms[idx].requires_grad_(False)

        # Evaluate and optionally upload affine transforms
        mse_metrics = None
        if args.eval_affine_mse or args.upload_affine_to_hub:
            mse_metrics = evaluate_affine_mse(
                affine_transforms=affine_transforms,
                source_model=checkpoint_model,
                target_model=model,
                dataset=train_dataset,
                target_layers=args.layers,
                num_examples=args.affine_eval_examples,
                device=model.device if hasattr(model, "device") else "cuda",
            )
            for layer_key, mse_val in mse_metrics.items():
                print(f"  {layer_key}: {mse_val:.6f}")

        if args.upload_affine_to_hub:
            upload_affine_transforms_to_hub(
                affine_transforms=affine_transforms,
                repo_id=args.upload_affine_to_hub,
                mse_metrics=mse_metrics,
                source_model_name=args.checkpoint_name,
                target_model_name=args.model_name,
                alpha=0.01,
                num_training_examples=args.affine_num_examples,
                private=args.affine_hub_private,
            )

    # Note: gradient_checkpointing=True saves memory but slows down training (~20-30%).
    # If you have enough VRAM, set this to False for further speedup.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = args.global_batch_size
    grad_acc_steps = max(1, global_batch_size // (args.pdbs * world_size))

    print(
        f"Running with {world_size} GPUs. Per device batch: {args.pdbs}. "
        f"Grad Acc steps: {grad_acc_steps}."
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=args.lr,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=args.pdbs,
        per_device_eval_batch_size=args.pdbs,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        gradient_checkpointing=True,
        fp16=True,
        save_strategy="no",
        ddp_find_unused_parameters=False,  # Required for Custom loops + PEFT usually
        warmup_steps=10 if args.lr_warmup else 0,
    )

    trainer = RRTrainer(
        args,
        model,
        training_args,
        train_dataset,
        tokenizer,
        args.layers,
        checkpoint_model=checkpoint_model,
        affine_transforms=affine_transforms,
    )

    model.train()
    trainer.train()

    if args.lora:
        model = model.merge_and_unload()

    if args.save_name:
        if "models/" in args.model_name:
            args.model_name = args.model_name.replace("models/", "")
        model.save_pretrained(f"./models/{args.model_name + '_' + args.save_name}")
        tokenizer.save_pretrained(f"./models/{args.model_name + '_' + args.save_name}")

    print("Done :)")
