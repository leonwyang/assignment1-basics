import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    lse = torch.logsumexp(inputs, dim=-1)  # (batch,)
    target_logits = inputs[torch.arange(inputs.shape[0]), targets]  # (batch,)
    return torch.mean(lse - target_logits)



def cross_entropy_with_unweighted_zloss(
    inputs: Float[Tensor, "batch vocab"], 
    targets: Int[Tensor, "batch"], 
) -> Float[Tensor, ""]:
    """
    Computes cross-entropy loss and optional z-loss regularization.

    Args:
        inputs: (batch, vocab) logits (unnormalized).
        targets: (batch,) integer class indices.
        z_loss_weight: float, if > 0 applies z-loss (default 0 = disabled).

    Returns:
        Scalar tensor: CE + z_loss (if enabled).
    """
    # cross-entropy
    lse = torch.logsumexp(inputs, dim=-1)  # (batch,)
    target_logits = inputs[torch.arange(inputs.shape[0]), targets]  # (batch,)
    ce_loss = torch.mean(lse - target_logits)

    z_loss = torch.mean(lse ** 2)
    return ce_loss, z_loss