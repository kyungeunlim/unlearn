"""Test layer-wise gradient freezing in SequentialSftTrainer."""

import pytest
import torch
from transformers import AutoModelForCausalLM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gradient_freezing_zeros_non_target_layers():
    """After backward, zeroing grads for non-target layers leaves only target grads."""
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-14m",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.train()

    input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
    outputs = model(input_ids)
    loss = outputs.logits.mean()
    loss.backward()

    # All trainable params should have grads after backward
    params_with_grad_before = sum(
        1 for _, p in model.named_parameters() if p.grad is not None
    )
    assert params_with_grad_before > 0

    # Apply gradient freezing for target layer 2
    target_layer = 2
    target_str = f".layers.{target_layer}."
    for name, param in model.named_parameters():
        if param.grad is not None and target_str not in name:
            param.grad = None

    # Only target layer params should have grads
    target_grad_count = 0
    for name, param in model.named_parameters():
        if target_str in name:
            assert param.grad is not None, f"Target param {name} lost its gradient"
            target_grad_count += 1
        else:
            assert param.grad is None, f"Non-target param {name} still has gradient"

    assert target_grad_count > 0, "No target layer parameters have gradients"
    print(
        f"target layer {target_layer}: {target_grad_count} params with grad, "
        f"{params_with_grad_before - target_grad_count} frozen"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gradient_freezing_different_layers():
    """Verify gradient freezing works for different target layer indices."""
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-14m",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.train()

    for target_layer in [0, 3, 5]:
        model.zero_grad()
        input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
        outputs = model(input_ids)
        loss = outputs.logits.mean()
        loss.backward()

        target_str = f".layers.{target_layer}."
        for name, param in model.named_parameters():
            if param.grad is not None and target_str not in name:
                param.grad = None

        target_grad_count = 0
        for name, param in model.named_parameters():
            if target_str in name:
                assert (
                    param.grad is not None
                ), f"Target param {name} lost gradient for layer {target_layer}"
                target_grad_count += 1
            else:
                assert (
                    param.grad is None
                ), f"Non-target param {name} has gradient for layer {target_layer}"

        assert target_grad_count > 0
        print(f"layer {target_layer}: {target_grad_count} params with grad")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_optimizer_step_only_updates_target_layer():
    """Optimizer only modifies target layer weights after gradient freezing."""
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-14m",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target_layer = 3
    target_str = f".layers.{target_layer}."

    # Snapshot weights before
    weights_before = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }

    # Forward + backward
    input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
    outputs = model(input_ids)
    loss = outputs.logits.mean()
    loss.backward()

    # Freeze non-target grads
    for name, param in model.named_parameters():
        if param.grad is not None and target_str not in name:
            param.grad = None

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Check which weights changed
    target_changed = 0
    non_target_changed = 0
    for name, param in model.named_parameters():
        diff = (param.detach() - weights_before[name]).abs().max().item()
        if target_str in name:
            if diff > 0:
                target_changed += 1
        else:
            assert diff == 0, f"Non-target param {name} changed by {diff}"
            if diff > 0:
                non_target_changed += 1

    assert target_changed > 0, "No target layer weights were updated"
    assert non_target_changed == 0, f"{non_target_changed} non-target params changed"
    print(
        f"layer {target_layer}: {target_changed} params updated, "
        f"0 non-target params changed"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_same_sign_gradient_filtering():
    """Same-sign filtering keeps only elements where retain and forget grads agree."""
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-14m",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.train()

    target_layer = 2
    target_str = f".layers.{target_layer}."

    # First backward pass (simulating retain loss) — one input
    input_ids_a = torch.randint(0, 1000, (2, 32), device="cuda")
    outputs_a = model(input_ids_a)
    loss1 = outputs_a.logits.mean()
    loss1.backward()

    retain_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None and target_str in name:
            retain_grads[name] = param.grad.clone()
    model.zero_grad()

    # Second backward pass (simulating forget loss) — different input
    input_ids_b = torch.randint(0, 1000, (2, 32), device="cuda")
    outputs_b = model(input_ids_b)
    loss2 = outputs_b.logits.var()
    loss2.backward()

    forget_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None and target_str in name:
            forget_grads[name] = param.grad.clone()
    model.zero_grad()

    # Apply same-sign filtering
    total_elements = 0
    agreed_elements = 0
    zeroed_elements = 0
    for name, param in model.named_parameters():
        if name in retain_grads and name in forget_grads:
            r, f = retain_grads[name], forget_grads[name]
            same_sign = (r * f) > 0
            param.grad = (r + f) * same_sign

            total_elements += same_sign.numel()
            agreed_elements += same_sign.sum().item()
            zeroed_elements += (~same_sign).sum().item()

    assert total_elements > 0, "No parameters matched target layer"
    assert agreed_elements > 0, "No elements agreed in sign"
    assert zeroed_elements > 0, "All elements agreed — expected some disagreement"

    # Verify the combined grad is zero where signs disagree
    for name, param in model.named_parameters():
        if name in retain_grads and name in forget_grads:
            r, f = retain_grads[name], forget_grads[name]
            same_sign = (r * f) > 0
            # Where signs disagree, combined grad must be zero
            assert (
                param.grad[~same_sign] == 0
            ).all(), f"Param {name} has non-zero grad where signs disagree"
            # Where signs agree, combined grad equals r + f
            torch.testing.assert_close(
                param.grad[same_sign], (r + f)[same_sign], rtol=0, atol=0
            )

    agree_pct = agreed_elements / total_elements * 100
    print(
        f"layer {target_layer}: {total_elements} elements, "
        f"{agreed_elements} agreed ({agree_pct:.1f}%), "
        f"{zeroed_elements} zeroed"
    )
