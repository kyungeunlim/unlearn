"""Unlearning via entropy maximization at frozen tuned lens layers (SFT version)."""

import os
from dataclasses import dataclass, field
from typing import Literal, cast

import torch
import torch.distributed as dist
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
        # import pynvml

        # pynvml.nvmlInit()
        # device_count = pynvml.nvmlDeviceGetCount()

        # for i in range(device_count):
        #     handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        #     name = pynvml.nvmlDeviceGetName(handle)
        #     mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        #     print(f"GPU {i}: {name}")
        #     print(f"Memory: {mem_info.used / 1024**2:.2f}MB /
        # {mem_info.total / 1024**2:.2f}MB")

        # pynvml.nvmlShutdown()

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
        retain_coeff = self.retain_coef * (0.9 + 0.1 * scheduled_coeff)
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


def load_tuned_lenses(lens_path, model, device):
    lens = TunedLens.from_model(model, bias=True)
    lens_state_dict = torch.load(f"{lens_path}/params.pt", map_location=device)

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

    return lens


@dataclass
class LensSftUnlearnConfig:
    num_train_examples: int = 1024
    unlearn_corrupt: bool = False
    corrupt_ratio: float = 0.5
    corrupt_ds: Literal["rewritten", "shuffled"] = "rewritten"
    wmdp_eval_limit: int | None = None
    mmlu_agieval_limit: int | None = None
    lr: float = 1e-4
    pdbs: int = 4
    retain_coef: float = 5.0
    remove_coef: float = 5.0
    layers: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 25, 30])
    model_name: str = "EleutherAI/deep-ignorance-unfiltered"
    save_path: str = ""
    revision: str = "main"
    lens_path: str = ""
    skip_eval: bool = False
    epochs: int = 1
    use_ultrachat: bool = False
    warmup_ratio: float = 0.0
    kl_retain_loss: bool = True


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"

    NUM_PROC = (os.cpu_count() or 32) // 2
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    parser = ArgumentParser()
    parser.add_arguments(LensSftUnlearnConfig, dest="run_cfg")
    run_cfg = parser.parse_args().run_cfg

    print("Parsed arguments:")
    for arg, value in vars(run_cfg).items():
        print(f"{arg}: {value}")

    model, tokenizer = get_model_and_tokenizer(
        run_cfg.model_name, revision=run_cfg.revision
    )
    model = cast(PreTrainedModel, model)

    # Enable gradients on training model
    for param in model.parameters():
        param.requires_grad = True
    model.enable_input_require_grads()

    # Load dataset
    train_dataset = get_unlearning_dataset(run_cfg, tokenizer, NUM_PROC)
    train_dataset.tokenized_bio_remove_dataset = apply_keyword_masks(
        train_dataset.tokenized_bio_remove_dataset, run_cfg, tokenizer
    )

    # Load frozen tuned lens
    print(f"Loading tuned lens from: {run_cfg.lens_path}")
    device = next(model.parameters()).device
    lens = load_tuned_lenses(run_cfg.lens_path, model, device)
    print(f"Loaded lens with {len(lens)} layer translators (frozen)")

    frozen_ref_model = None
    if run_cfg.kl_retain_loss:
        frozen_ref_model = AutoModelForCausalLM.from_pretrained(
            run_cfg.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": local_rank},
        )
        frozen_ref_model.eval()
        print("Loaded reference model in bf16.")
    else:
        print("KL retain loss disabled, skipping reference model.")

    global_batch_size = 32
    grad_acc_steps = max(1, global_batch_size // (run_cfg.pdbs * world_size))

    print(
        f"Running with {world_size} GPUs. Per device batch: {run_cfg.pdbs}. "
        f"Grad Acc steps: {grad_acc_steps}."
    )

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=run_cfg.lr,
        warmup_ratio=run_cfg.warmup_ratio,
        gradient_accumulation_steps=grad_acc_steps,
        per_device_train_batch_size=run_cfg.pdbs,
        per_device_eval_batch_size=run_cfg.pdbs,
        num_train_epochs=run_cfg.epochs,
        weight_decay=0.01,
        gradient_checkpointing=False,
        bf16=True,
        max_grad_norm=1.0,
        save_strategy="no",
    )

    trainer = SFTUnlearningTrainer(
        run_cfg,
        model,
        training_args,
        train_dataset,
        tokenizer,
        run_cfg.layers,
        lens=lens,
        frozen_ref_model=frozen_ref_model,
        model=model,
        target_modules=[f"gpt_neox.layers.{i}" for i in [8, 12, 16, 20, 24, 28, 31]],
    )

    model.train()
    trainer.train()

    if run_cfg.save_path:
        save_checkpoint(trainer, run_cfg.save_path, tokenizer)

    if dist.is_initialized():
        dist.destroy_process_group()

    print("Done :)")
