import torch
from typing import Optional, Tuple

def apply_diffusion_mask(
    clean_input: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-3,
    generator: Optional[torch.Generator] = None,
    t_distribution: str = "uniform",
    beta_k: float = 1.0,
    tokenwise_p_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    LLaDA-style forward noising: sample t, then p=(1-eps)*t+eps, Bernoulli per token.
    - clean_input: [B, L]
    - attention_mask: [B, L] (1 for real tokens, 0 for padding)
    - valid_mask: [B, L] (1 for tokens that CAN be masked)
    - tokenwise_p_scale: [B, L] optional scaling for p per token.
    Returns (noisy_input, target_ids, masked_positions, p_scalar).
    """
    device = clean_input.device
    bsz, seq_len = clean_input.shape
    
    if valid_mask is None:
        valid_mask = attention_mask.bool()
    else:
        valid_mask = valid_mask.bool() & attention_mask.bool()

    if generator is None:
        base_rand = torch.rand(bsz, device=device)
        rand_mask = torch.rand((bsz, seq_len), device=device)
    else:
        base_rand = torch.rand(bsz, device=device, generator=generator)
        rand_mask = torch.rand((bsz, seq_len), device=device, generator=generator)

    if t_distribution == "beta":
        t = torch.pow(base_rand, 1.0 / beta_k)
    else:
        t = base_rand

    p_scalar = (1.0 - eps) * t + eps
    p_mask = p_scalar.unsqueeze(1).expand(-1, seq_len)
    
    if tokenwise_p_scale is not None:
        p_mask = torch.clamp(p_mask * tokenwise_p_scale.to(device), min=0.0, max=1.0)

    masked_positions = (rand_mask < p_mask) & valid_mask
    
    # Ensure at least one masked position per sample if there are valid tokens
    needs_mask = (masked_positions.sum(dim=1) == 0) & (valid_mask.sum(dim=1) > 0)
    if needs_mask.any():
        for idx in torch.nonzero(needs_mask, as_tuple=False).squeeze(1).tolist():
            valid_idx = torch.nonzero(valid_mask[idx], as_tuple=False).squeeze(1)
            if valid_idx.numel() > 0:
                choice = valid_idx[torch.randint(0, valid_idx.numel(), (1,), device=device)]
                masked_positions[idx, choice] = True

    noisy = clean_input.clone()
    noisy[masked_positions] = mask_token_id
    
    target = clean_input.clone()
    target[~masked_positions] = -100 # Standard cross-entropy ignore index
    target[~attention_mask.bool()] = -100
    
    return noisy, target, masked_positions, p_scalar
