"""Test that ActivationCapture hooks match hidden_states output."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from unlearn.utils.hook import ActivationCapture, resolve_layer_names


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hook_matches_hidden_states():
    """Verify hook on layers.L captures same tensor as hidden_states[L+1]."""
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    target_layers = [2, 5, 10, 15, 20]
    layer_id_to_name = resolve_layer_names(model, target_layers)
    target_module_names = list(layer_id_to_name.values())

    input_ids = torch.randint(0, 1000, (2, 32), device="cuda")

    # Forward with hidden_states
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    # Forward with hooks
    capturer = ActivationCapture(model, target_module_names)
    capturer.register()
    with torch.no_grad():
        model(input_ids=input_ids, output_hidden_states=False)

    for layer_idx in target_layers:
        hs = hidden_states[layer_idx + 1]
        hook_act = capturer.activations[layer_id_to_name[layer_idx]]
        torch.testing.assert_close(
            hs,
            hook_act,
            msg=f"Mismatch at layer {layer_idx}: "
            f"hidden_states[{layer_idx + 1}] vs hook on "
            f"{layer_id_to_name[layer_idx]}",
        )

    capturer.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_resolve_layer_names_gpt_neox():
    """Verify resolve_layer_names works for GPT-NeoX architecture."""
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-14m",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    target_layers = [0, 2, 4]
    mapping = resolve_layer_names(model, target_layers)

    for idx in target_layers:
        assert idx in mapping, f"Layer {idx} not found in mapping"
        assert (
            "layers" in mapping[idx]
        ), f"Expected 'layers' in name, got {mapping[idx]}"

    # Verify hooks work with this mapping
    target_module_names = list(mapping.values())
    capturer = ActivationCapture(model, target_module_names)
    capturer.register()

    input_ids = torch.randint(0, 1000, (1, 16), device="cuda")
    with torch.no_grad():
        model(input_ids=input_ids)

    for idx in target_layers:
        assert (
            mapping[idx] in capturer.activations
        ), f"Hook on {mapping[idx]} did not fire"

    capturer.remove()
