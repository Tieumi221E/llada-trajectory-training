import torch
import torch.nn.functional as F
from typing import Tuple

def diffusion_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    p_mask_scalar: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy over masked positions only, scaled by 1/p.
    - logits: [B, L, V]
    - target_ids: [B, L] (-100 for ignored positions)
    - p_mask_scalar: [B] (scalar p per sample)
    """
    loss_sum, count = diffusion_loss_sum(logits, target_ids, p_mask_scalar)
    return loss_sum / count.clamp(min=1.0)

def diffusion_loss_sum(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    p_mask_scalar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (loss_sum, masked_count) with 1/p scaling applied.
    """
    vocab_size = logits.size(-1)
    B, L = target_ids.shape
    
    # Raw cross-entropy [B*L]
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        target_ids.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    loss = loss.view(B, L)
    
    # Scale by 1/p
    loss = loss * (1.0 / p_mask_scalar).unsqueeze(1)
    
    mask = (target_ids != -100).float()
    masked_loss = (loss * mask).sum()
    masked_count = mask.sum()
    
    return masked_loss, masked_count
