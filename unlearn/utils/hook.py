from contextlib import contextmanager
from functools import partial
from typing import Any, Dict

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel, unwrap_model

from unlearn.utils.utils import assert_type


def resolve_layer_names(
    model: PreTrainedModel, layer_indices: list[int]
) -> Dict[int, str]:
    """
    Map integer layer indices to module names for use with forward hooks.
    Handles PEFT wrapping and different architectures (OLMo, Llama, etc.).
    """
    unwrapped = unwrap_model(model)

    base = unwrapped
    if hasattr(base, "base_model"):
        base = base.base_model
    if hasattr(base, "model"):
        base = base.model  # type: ignore

    base = assert_type(nn.Module, base)

    layer_list_name = None
    for name, module in base.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            if "layers" in name or "blocks" in name or "h" in name:
                layer_list_name = name
                break

    if not layer_list_name:
        if hasattr(base, "transformer") and hasattr(base.transformer, "blocks"):
            layer_list_name = "transformer.blocks"
        elif hasattr(base, "layers"):
            layer_list_name = "layers"
        else:
            raise ValueError("Could not automatically locate transformer layer list.")

    mapping = {}
    target_indices = set(layer_indices)

    for name, module in unwrapped.named_modules():
        if layer_list_name in name:
            parts = name.split(".")
            try:
                idx = int(parts[-1])
            except ValueError:
                continue
            if idx in target_indices:
                mapping[idx] = name

    if len(mapping) != len(target_indices):
        print(
            f"Warning: requested {len(target_indices)} layers, "
            f"but found {len(mapping)}."
        )

    return mapping


class ActivationCapture:
    def __init__(self, model, target_module_names, accelerator=None):
        self.model = model
        self.target_module_names = set(target_module_names)
        self.activations = {}
        self._handles = []
        self.accelerator = accelerator

    def _hook_fn(self, module, input, output, name):
        act = output[0] if isinstance(output, tuple) else output
        self.activations[name] = act

    def register(self):
        self.activations = {}

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model

        for name, module in model.named_modules():
            if name in self.target_module_names:
                handle = module.register_forward_hook(partial(self._hook_fn, name=name))
                self._handles.append(handle)

    def clear(self):
        self.activations = {}

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@torch.inference_mode()
@contextmanager
def collect_activations(
    model: PreTrainedModel,
    hookpoints: list[str],
    token: int | None = None,
    input_acts: bool = False,
    offload_device=torch.device("cpu"),
):
    """
    Context manager that hooks a model and collects activations.
    An activation tensor is produced for each batch processed and stored
    in added to a list for that hookpoint in the activations dictionary.
    Args:
        model: The transformer model to hook
        hookpoints: List of hookpoints to collect activations from
        input_acts: Whether to collect input activations or output activations
    Yields:
        Dictionary mapping hookpoints to their collected activations
    """
    activations = {}
    handles = []

    def create_input_hook(hookpoint: str):
        def input_hook(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(input, tuple):
                activations[hookpoint] = input[0].to(
                    device=offload_device, non_blocking=True
                )
            else:
                activations[hookpoint] = input.to(
                    device=offload_device, non_blocking=True
                )

            if token is not None:
                activations[hookpoint] = activations[hookpoint][:, token]

        return input_hook

    def create_output_hook(hookpoint: str):
        def output_hook(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                activations[hookpoint] = output[0].to(
                    device=offload_device, non_blocking=True
                )
            else:
                activations[hookpoint] = output.to(
                    device=offload_device, non_blocking=True
                )

            if token is not None:
                activations[hookpoint] = activations[hookpoint][:, token]

        return output_hook

    for name, module in model.base_model.named_modules():
        if name in hookpoints:
            hook = create_input_hook(name) if input_acts else create_output_hook(name)
            handle = module.register_forward_hook(hook)
            handles.append(handle)

    try:
        yield activations
    finally:
        for handle in handles:
            handle.remove()

        handles.clear()


collect_input_activations = partial(collect_activations, input_acts=True)
collect_output_activations = partial(collect_activations, input_acts=False)
