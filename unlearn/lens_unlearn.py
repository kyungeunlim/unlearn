"""Unlearning via entropy maximization at frozen tuned lens layers."""

import argparse
import os
from typing import cast

import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, PreTrainedModel
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import seed_worker
from tuned_lens import TunedLens

from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer, unwrap_model


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
        # 1. Unwrap model if using DDP to access .disable_adapter()
        # and specific layers.
        unwrapped_model = unwrap_model(model)

        # Determine device from inputs (safest way in DDP)
        # If inputs aren't on device yet, fall back to model device
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
            target_device # type: ignore
        ) # type: ignore

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

        # Optimization: Broadcasting masks (Batch, Seq) -> (1, Batch, Seq, 1)
        broadcast_retain_mask = retain_attention_mask.unsqueeze(0).unsqueeze(-1)

        # Use unwrapped_model for context manager
        # (DDP wrapper doesn't have disable_adapter)
        with unwrapped_model.disable_adapter(): # type: ignore
            unwrapped_model.eval() # type: ignore
            with torch.no_grad():
                ### Retain control
                if retain_coeff > 0:
                    orig_retain_outputs = unwrapped_model(**retain_inputs_dict)[module] # type: ignore
                    orig_retain_hidden = torch.stack(orig_retain_outputs).detach()
                    orig_retain_hidden *= broadcast_retain_mask
                    del orig_retain_outputs

        unwrapped_model.train() # type: ignore

        ### Retain control
        if retain_coeff > 0:
            # We use unwrapped_model to access specific hidden states.
            # DDP usually requires using 'model' to sync gradients, but since
            # we are effectively doing a manual loss calculation on sub-components
            # and Accelerate/Trainer handles the backward pass sync, this is safe.
            # If you see hanging, switch these forward passes to use 'model'
            # (but you lose direct access to [module] output structure if wrapped).
            lora_retain_outputs = unwrapped_model(**retain_inputs_dict)[module]
            lora_retain_hidden = (
                torch.stack(lora_retain_outputs) * broadcast_retain_mask
            )
            retain_loss = torch.norm(
                lora_retain_hidden - orig_retain_hidden, dim=-1, p=2, dtype=torch.float
            ).nanmean()
        else:
            retain_loss = 0

        ### Forget loss - entropy maximization via tuned lens
        ### (memory-efficient CE proxy)
        if circuit_breaker_coeff > 0:
            lora_circuit_breaker_outputs = unwrapped_model(**cb_inputs_dict)[module]

            # Use cross-entropy with random targets as memory-efficient entropy proxy
            # Minimizing CE against random tokens spreads probability mass -> maximizes entropy
            layer_losses = []
            lens_device = next(self.lens.parameters()).device
            for layer_idx in self.lora_target_layers:
                hidden = lora_circuit_breaker_outputs[layer_idx]  # [batch, seq, hidden]
                hidden_bf16 = hidden.to(device=lens_device, dtype=torch.bfloat16)

                lens_logits = self.lens(
                    hidden_bf16, idx=layer_idx
                )  # [batch, seq, vocab]
                batch_size, seq_len, vocab_size = lens_logits.shape

                # Random target tokens
                random_targets = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=lens_logits.device
                )

                # Compute CE loss only on non-padded tokens
                mask = circuit_breaker_attention_mask.bool()
                logits_flat = lens_logits[mask]  # [num_valid_tokens, vocab]
                targets_flat = random_targets[mask]  # [num_valid_tokens]

                if logits_flat.numel() > 0:
                    ce_loss = F.cross_entropy(
                        logits_flat.float(), targets_flat, reduction="mean"
                    )
                    layer_losses.append(ce_loss)

            # Minimize CE against random targets to maximize entropy
            # Normalize by ln(vocab_size) to keep loss in similar scale
            log_vocab = torch.log(torch.tensor(float(vocab_size), device=target_device))
            if layer_losses:
                mean_ce = torch.stack(layer_losses).mean()
                circuit_breaker_loss = mean_ce / log_vocab
                mean_entropy = -mean_ce  # For logging compatibility (approx)
            else:
                circuit_breaker_loss = torch.tensor(0.0, device=target_device)
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


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    # Optimization: Utilize half of the CPU cores for dataset mapping
    NUM_PROC = (os.cpu_count() or 32) // 2

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
    parser.add_argument("--pdbs", type=int, default=4)
    parser.add_argument("--retain_coef", type=float, default=5.0)
    parser.add_argument("--remove_coef", type=float, default=5.0)
    parser.add_argument("--lora_r", type=float, default=16)
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
        "--lens_path", type=str, required=True, help="Path to tuned lens weights"
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip final evaluation (for faster tuning)",
    )

    args = parser.parse_args()

    if "smollm2" in args.model_name:
        args.layers = [l for l in args.layers if l < 24]

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print()

    model, tokenizer = get_model_and_tokenizer(args.model_name, revision=args.revision)
    train_dataset = get_unlearning_dataset(args, tokenizer, NUM_PROC)

    # Load frozen tuned lens
    print(f"Loading tuned lens from: {args.lens_path}")
    device = next(model.parameters()).device
    lens = TunedLens.from_model(model, bias=True)
    lens_state_dict = torch.load(f"{args.lens_path}/params.pt", map_location=device)

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

    lora_layers_to_transform = [i for i in range(max(args.layers) + 1)]

    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_r,
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
                if "OLMo" in args.model_name
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

    # Note: gradient_checkpointing=True saves memory but slows down training (~20-30%).
    # If you have enough VRAM, set this to False for further speedup.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = 32
    # Ensure accumulation is at least 1
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
        num_train_epochs=1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        bf16=True,
        max_grad_norm=1.0,
        save_strategy="no",
        ddp_find_unused_parameters=False,  # Required for Custom loops + PEFT usually
    )

    trainer = RRTrainer(
        args, model, training_args, train_dataset, tokenizer, args.layers, lens=lens
    )

    model.train()
    trainer.train()

    if args.lora:
        model = model.merge_and_unload() # type: ignore

    if args.save_name:
        if "models/" in args.model_name:
            args.model_name = args.model_name.replace("models/", "")
        model.save_pretrained(f"./models/{args.model_name + '_' + args.save_name}")
        tokenizer.save_pretrained(f"./models/{args.model_name + '_' + args.save_name}")

    print("Done :)")
