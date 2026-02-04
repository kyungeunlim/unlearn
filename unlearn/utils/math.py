import torch
from torch import Tensor


@torch.inference_mode()
def stable_rank(A: Tensor) -> float:
    eps = torch.finfo(A.dtype).eps

    # Spectral norm
    _, S, _ = torch.svd_lowrank(A, q=1)
    spec = S[0]

    # Frobenius norm
    # frob = torch.linalg.matrix_norm(A, ord="fro")
    frob_sq = A.pow(2).sum()

    # Ratio of the squares
    return (frob_sq / (spec.pow(2) + eps)).item()


@torch.inference_mode()
def effective_rank(A: Tensor) -> float:
    eps = torch.finfo(A.dtype).eps

    # Get singular values
    S = torch.linalg.svdvals(A)

    # Normalize to get probability distribution
    S_norm = S / (S.sum() + eps)

    # Compute entropy (excluding zeros as lim
    # log(x) as x tends to 0 is 0)
    S_pos = S_norm[S_norm > eps]
    entropy = -(S_pos * S_pos.log()).sum()

    # Effective rank is exp(entropy)
    return entropy.exp().item()


def max_entropy_kl_loss(logits):
    """
    Calculates KL(Uniform || Model) efficiently directly from logits.
    Minimizing this maximizes the entropy of the model's output.

    Args:
        lens_logits: Tensor of shape [batch, seq_len, vocab_size]

    Returns:
        Scalar loss
    """
    # 1. Calculate log(sum(e^x) (The normalizing constant of softmax)
    # Shape: [batch, seq_len]
    log_sum_exp = torch.logsumexp(logits, dim=-1)

    # 2. Calculate Mean of logits - (1/V) * sum(x)
    # Shape: [batch, seq_len]
    logit_mean = torch.mean(logits, dim=-1)

    # 3. The KL divergence (up to a constant) is simply the difference
    # L = log(sum(e^x)) - mean(x)
    loss = log_sum_exp - logit_mean

    return loss.mean()


def top_k_entropy_loss(logits, k=100):
    """
    Maximize entropy over only the top-K most likely tokens at each position.

    Extracts the K highest logits and pushes them toward uniform
    via KL(Uniform || Model).

    Args:
        logits: Tensor of shape [N, V] (already masked to valid positions)
        k: Number of top tokens to target

    Returns:
        Scalar loss (unnormalized; caller should divide by log(k))
    """
    topk_logits, _ = torch.topk(logits, k, dim=-1)  # [N, K]
    return max_entropy_kl_loss(topk_logits)
