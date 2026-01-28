"""Unlearning via entropy maximization at frozen tuned lens layers (SFT version)."""

import argparse
import os
from typing import cast

import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_module
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
from unlearn.utils.unlearning_dataset import get_unlearning_dataset
from unlearn.utils.worker_utils import get_model_and_tokenizer


class UnlearningTrainer(Trainer):
    def __init__(
        self,
        run_args,
        model,
        args,
        train_dataset,
        tokenizer,
        target_layers,
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
        self.target_layers = target_layers
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


class SFTUnlearningTrainer(UnlearningTrainer):

    def __init__(
        self, *args, frozen_ref_model=None, target_modules: list[str], model, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.frozen_ref_model = frozen_ref_model

        self.target_modules = target_modules
        self.act_capturer = ActivationCapture(
            model, target_modules, accelerator=self.accelerator
        )
        self.act_capturer.register()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        target_device = (
            inputs["input_ids"].device
            if hasattr(inputs["input_ids"], "device")
            else next(model.parameters()).device
        )

        retain_input_ids = inputs.get("input_ids").to(target_device)  # type: ignore
        forget_input_ids = inputs.get("bio_remove_input_ids").to(target_device)  # type: ignore
        forget_attention_mask = inputs.get("bio_remove_attention_mask").to(  # type: ignore
            target_device
        )

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
        forget_coeff = self.remove_coef * (1 - 0.25 * scheduled_coeff)

        model.train()

        ### Retain loss - KL divergence
        if retain_coeff > 0 and self.frozen_ref_model is not None:
            # Get logits from frozen model (WHICH IS ON CPU)
            with torch.no_grad():
                # Run forward pass on CPU
                # ref_inputs_cpu = retain_input_ids.to("cpu")
                ref_outputs = self.frozen_ref_model(
                    retain_input_ids, attention_mask=None
                )
                ref_logits = ref_outputs.logits.to(target_device)

            current_logits = model(
                input_ids=retain_input_ids,
                attention_mask=None,
            ).logits

            self.act_capturer.clear()

            # KL divergence (Both tensors are now on GPU)
            retain_loss = torch.nn.functional.kl_div(
                input=torch.nn.functional.log_softmax(current_logits, dim=-1),
                target=torch.nn.functional.softmax(ref_logits, dim=-1),
                reduction="batchmean",
            )
        else:
            retain_loss = torch.tensor(0.0, device=target_device)

        ### Forget loss - entropy maximization via tuned lens (using ActivationCapture)
        if forget_coeff > 0:
            outputs = model(
                input_ids=forget_input_ids,
                attention_mask=forget_attention_mask,
            )

            # Connect the loss to the output to trigger FSDP's bwd books
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

                assert self.lens is not None
                lens_logits = self.lens(hidden_bf16, idx=layer_idx)
                batch_size, seq_len, vocab_size = lens_logits.shape

                random_targets = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=lens_logits.device
                )

                mask = forget_attention_mask.bool()
                logits_flat = lens_logits[mask]
                targets_flat = random_targets[mask]

                if logits_flat.numel() > 0:
                    ce_loss = F.cross_entropy(
                        logits_flat.float(), targets_flat, reduction="mean"
                    )
                    layer_losses.append(ce_loss)

            self.act_capturer.clear()

            if vocab_size is None:
                vocab_size = 50257  # Default vocab size
            log_vocab = torch.log(torch.tensor(float(vocab_size), device=target_device))
            if layer_losses:
                mean_ce = torch.stack(layer_losses).mean()
                forget_loss = mean_ce / log_vocab
                mean_entropy = -mean_ce
            else:
                forget_loss = torch.tensor(0.0, device=target_device)
                mean_entropy = torch.tensor(0.0)

            forget_loss = forget_loss + dummy_loss
        else:
            forget_loss = torch.tensor(0.0, device=target_device)
            mean_entropy = torch.tensor(0.0)

        loss = retain_coeff * retain_loss + forget_coeff * forget_loss

        if (
            self.current_training_step % 8 == 0
            and int(os.environ.get("LOCAL_RANK", 0)) == 0
        ):
            entropy_val = (
                mean_entropy.item()
                if isinstance(mean_entropy, torch.Tensor)
                else mean_entropy
            )
            retain_val = (
                retain_loss.item()
                if isinstance(retain_loss, torch.Tensor)
                else retain_loss
            )
            print(
                f"retain_coeff: {retain_coeff:.4f} || "
                f"forget_coeff: {forget_coeff:.4f} || "
                f"retain_loss: {retain_val:.4f} || "
                f"forget_loss: {forget_loss:.4f} || "
                f"mean_entropy: {entropy_val:.4f}"
            )

        self.current_training_step += 1
        return (loss,) if return_outputs else loss


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

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
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--use_ultrachat",
        action="store_true",
        help="Mix UltraChat into retain data (25%% of num_train_examples)",
    )

    args = parser.parse_args()

    if "smollm2" in args.model_name:
        args.layers = [l for l in args.layers if l < 24]

    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    model, tokenizer = get_model_and_tokenizer(args.model_name, revision=args.revision)
    train_dataset = get_unlearning_dataset(args, tokenizer, NUM_PROC)

    # Load frozen tuned lens
    print(f"Loading tuned lens from: {args.lens_path}")
    device = next(model.parameters()).device
    lens = TunedLens.from_model(model, bias=True)
    lens_state_dict = torch.load(f"{args.lens_path}/params.pt", map_location=device)

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

    current_state = lens.state_dict()
    for key in current_state:
        if key.startswith("unembed"):
            mapped_state_dict[key] = current_state[key]

    lens.load_state_dict(mapped_state_dict)

    for submodule in lens.modules():
        remove_hook_from_module(submodule)

    lens = lens.to(device=device, dtype=torch.bfloat16)
    lens.eval()
    for param in lens.parameters():
        param.requires_grad = False
    print(f"Loaded lens with {len(lens)} layer translators (frozen)")

    # Load frozen reference model on CPU for retain loss
    frozen_ref_model = None
    if args.retain_coef > 0:
        # 1. Get the specific GPU index for this process
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # OPTIONAL: Use 4-bit quantization to save massive VRAM (Recommended)
        # Requires: pip install bitsandbytes
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            print("Using 4-bit quantization for reference model to save memory.")
        except ImportError:
            bnb_config = None
            print("bitsandbytes not found, loading in full bf16.")

        frozen_ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            # CRITICAL: Map everything to the CURRENT GPU only.
            device_map={"": local_rank},
        )

        frozen_ref_model.eval()

    # Enable gradients on training model
    for param in model.parameters():
        param.requires_grad = True

    model = cast(PreTrainedModel, model)
    model.enable_input_require_grads()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = 32
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
        gradient_checkpointing=False,
        bf16=True,
        max_grad_norm=1.0,
        save_strategy="no",
    )

    trainer = SFTUnlearningTrainer(
        args,
        model,
        training_args,
        train_dataset,
        tokenizer,
        args.layers,
        lens=lens,
        frozen_ref_model=frozen_ref_model,
        model=model,
        target_modules=[f"gpt_neox.layers.{i}" for i in [8, 12, 16, 20, 24, 28, 31]],
    )

    model.train()
    trainer.train()

    if args.save_name:
        args.save_name = f"{args.save_name}_sft"
        if "models/" in args.model_name:
            args.model_name = args.model_name.replace("models/", "")
        save_path = f"./models/{args.model_name}_{args.save_name}"
        # Use trainer.save_model to properly handle FSDP state gathering
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)

    print("Done :)")
