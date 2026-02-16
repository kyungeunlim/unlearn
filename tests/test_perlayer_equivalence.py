"""Verify per-layer loss computation matches stacked loss computation."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from unlearn.utils.hook import ActivationCapture, resolve_layer_names


def test_perlayer_l2_matches_stacked():
    """Per-layer L2 retain loss == stacked L2 retain loss."""
    L, B, S, H = 6, 4, 32, 128
    mask = torch.randint(0, 2, (B, S)).float()

    orig_acts = [torch.randn(B, S, H) for _ in range(L)]
    lora_acts = [torch.randn(B, S, H) for _ in range(L)]

    # Stacked version (original code)
    broadcast_mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, B, S, 1]
    stacked_orig = torch.stack(orig_acts) * broadcast_mask
    stacked_lora = torch.stack(lora_acts) * broadcast_mask
    stacked_loss = torch.norm(
        stacked_lora - stacked_orig, dim=-1, p=2, dtype=torch.float
    ).nanmean()

    # Per-layer version (new code)
    mask_1d = mask.unsqueeze(-1)  # [B, S, 1]
    perlayer_loss = torch.tensor(0.0)
    for i in range(L):
        orig_h = orig_acts[i] * mask_1d
        lora_h = lora_acts[i] * mask_1d
        perlayer_loss = (
            perlayer_loss
            + torch.norm(lora_h - orig_h, dim=-1, p=2, dtype=torch.float).nanmean()
        )
    perlayer_loss = perlayer_loss / L

    torch.testing.assert_close(stacked_loss, perlayer_loss, rtol=1e-5, atol=1e-6)


def test_perlayer_cosine_cb_matches_stacked():
    """Per-layer cosine CB loss == stacked cosine CB loss."""
    L, B, S, H = 6, 4, 32, 128
    mask = torch.randint(0, 2, (B, S)).float()

    lora_acts = [torch.randn(B, S, H) for _ in range(L)]
    ref_acts = [torch.randn(B, S, H) for _ in range(L)]

    # Stacked version (original code)
    broadcast_mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, B, S, 1]
    stacked_lora = torch.stack(lora_acts)
    stacked_ref = torch.stack(ref_acts)
    norm_lora = stacked_lora / torch.norm(
        stacked_lora, dim=-1, keepdim=True, dtype=torch.float
    )
    norm_ref = stacked_ref / torch.norm(
        stacked_ref, dim=-1, keepdim=True, dtype=torch.float
    )
    inner_product = (norm_lora * norm_ref) * broadcast_mask
    denom = mask.sum() * L
    stacked_loss = torch.relu(inner_product.nansum(dim=-1)).nansum() / (denom + 1e-6)

    # Per-layer version (new code)
    cb_total = torch.tensor(0.0)
    for i in range(L):
        lora_h = lora_acts[i]
        ref_h = ref_acts[i]
        n_lora = lora_h / torch.norm(lora_h, dim=-1, keepdim=True, dtype=torch.float)
        n_ref = ref_h / torch.norm(ref_h, dim=-1, keepdim=True, dtype=torch.float)
        cos_sim = (n_lora * n_ref).sum(dim=-1) * mask
        cb_total = cb_total + torch.relu(cos_sim).sum()
    perlayer_loss = cb_total / (denom + 1e-6)

    torch.testing.assert_close(stacked_loss, perlayer_loss, rtol=1e-5, atol=1e-6)


def test_perlayer_orth_matches_stacked():
    """Per-layer orthogonality loss == stacked (bmm) orthogonality loss."""
    L, B, S, H = 4, 3, 16, 64
    mask = torch.randint(0, 2, (B, S)).float()
    mask[:, 0] = 1  # ensure at least one valid token per item

    lora_acts = [torch.randn(B, S, H) for _ in range(L)]

    # Stacked version (original code)
    broadcast_mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, B, S, 1]
    stacked = torch.stack(lora_acts)  # [L, B, S, H]

    cb_mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
    masked_outputs = stacked * broadcast_mask  # [L, B, S, H]
    pooled = masked_outputs.sum(dim=2) / cb_mask_sum.unsqueeze(0)  # [L, B, H]
    pooled_norm = pooled / (
        torch.norm(pooled, dim=-1, keepdim=True, dtype=torch.float) + 1e-8
    )
    pairwise_sim = torch.bmm(pooled_norm, pooled_norm.transpose(1, 2))  # [L, B, B]
    diag_mask_3d = (1.0 - torch.eye(B, dtype=torch.float)).unsqueeze(0)
    off_diag_sim = pairwise_sim * diag_mask_3d
    num_pairs = B * (B - 1) * L
    stacked_loss = torch.relu(off_diag_sim).sum() / (num_pairs + 1e-6)

    # Per-layer version (new code)
    cb_mask_1d = mask.unsqueeze(-1)  # [B, S, 1]
    diag_mask_2d = 1.0 - torch.eye(B, dtype=torch.float)
    orth_total = torch.tensor(0.0)
    for i in range(L):
        masked = lora_acts[i] * cb_mask_1d  # [B, S, H]
        pooled_l = masked.sum(dim=1) / cb_mask_sum  # [B, H]
        pooled_n = pooled_l / (
            torch.norm(pooled_l, dim=-1, keepdim=True, dtype=torch.float) + 1e-8
        )
        sim = pooled_n @ pooled_n.T  # [B, B]
        off_diag = sim * diag_mask_2d
        orth_total = orth_total + torch.relu(off_diag).sum()
    perlayer_loss = orth_total / (num_pairs + 1e-6)

    torch.testing.assert_close(stacked_loss, perlayer_loss, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_activation_capture_l2_matches_hidden_states_l2():
    """
    End-to-end: L2 loss via ActivationCapture per-layer matches
    L2 loss via output_hidden_states + stack.
    """
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    target_layers = [2, 10, 20]
    layer_id_to_name = resolve_layer_names(model, target_layers)
    target_module_names = list(layer_id_to_name.values())

    input_ids = torch.randint(0, 1000, (2, 32), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -8:] = 0  # mask out last 8 tokens

    with torch.no_grad():
        # Method 1: output_hidden_states + stack (old approach)
        outputs_hs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # all_hidden = torch.stack(outputs_hs.hidden_states)  # [N+1, B, S, H]
        broadcast_mask = attention_mask.unsqueeze(0).unsqueeze(-1).float()
        # Select only our target layers from hidden_states (layer L maps to
        # hidden_states[L+1])
        selected_orig = torch.stack(
            [outputs_hs.hidden_states[l + 1] for l in target_layers]
        )
        selected_orig = selected_orig * broadcast_mask

        # Do a second forward (same model, same input) to get "lora" activations
        # (in practice these are the same, so loss should be ~0)
        outputs_hs2 = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        selected_lora = torch.stack(
            [outputs_hs2.hidden_states[l + 1] for l in target_layers]
        )
        selected_lora = selected_lora * broadcast_mask
        stacked_loss = torch.norm(
            selected_lora - selected_orig, dim=-1, p=2, dtype=torch.float
        ).nanmean()

        # Method 2: ActivationCapture + per-layer (new approach)
        capturer = ActivationCapture(model, target_module_names)
        capturer.register()

        model(input_ids=input_ids, attention_mask=attention_mask)
        orig_acts = {}
        mask_1d = attention_mask.unsqueeze(-1).float()
        for l in target_layers:
            name = layer_id_to_name[l]
            orig_acts[l] = capturer.activations[name].detach() * mask_1d
        capturer.clear()

        model(input_ids=input_ids, attention_mask=attention_mask)
        n_layers = len(target_layers)
        perlayer_loss = torch.tensor(0.0, device="cuda")
        for l in target_layers:
            name = layer_id_to_name[l]
            lora_h = capturer.activations[name] * mask_1d
            perlayer_loss = (
                perlayer_loss
                + torch.norm(
                    lora_h - orig_acts[l], dim=-1, p=2, dtype=torch.float
                ).nanmean()
            )
        perlayer_loss = perlayer_loss / n_layers
        capturer.remove()

    # Both should be ~0 (same model, same input) and equal to each other
    torch.testing.assert_close(
        stacked_loss.cpu(), perlayer_loss.cpu(), rtol=1e-3, atol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_activation_capture_matches_hidden_states_values():
    """
    Verify that ActivationCapture captures the same tensors as
    output_hidden_states for the target layers.
    """
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    target_layers = [5, 15, 25]
    layer_id_to_name = resolve_layer_names(model, target_layers)
    target_module_names = list(layer_id_to_name.values())

    input_ids = torch.randint(0, 1000, (2, 32), device="cuda")

    capturer = ActivationCapture(model, target_module_names)
    capturer.register()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    for l in target_layers:
        name = layer_id_to_name[l]
        hook_act = capturer.activations[name]
        # hidden_states[0] is embeddings, hidden_states[l+1] is output of layer l
        hs_act = outputs.hidden_states[l + 1]
        torch.testing.assert_close(
            hook_act,
            hs_act,
            msg=f"Mismatch at layer {l}: hook vs hidden_states[{l + 1}]",
        )

    capturer.remove()
