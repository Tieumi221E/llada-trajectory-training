"""LLaDA trajectory generation through the shared dllm sampler."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from dllm import CanvasConfig, TrajectorySample, generate_canvas


@torch.no_grad()
def generate_with_trace(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    topk_save: int = 8,
) -> Tuple[TrajectorySample, Dict[str, float]]:
    """Generate one rollout while retaining the project's legacy entry point."""
    if prompt_ids.shape[0] != 1:
        raise ValueError("trajectory generation expects a single prompt")
    confidence = {
        "low_confidence": "prob",
        "random": "random",
    }.get(remasking)
    if confidence is None:
        raise ValueError("remasking must be 'low_confidence' or 'random'")

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    output = generate_canvas(
        model,
        prompt_ids,
        mask_id,
        CanvasConfig(
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            sampling="gumbel",
            commit="transfer",
            confidence=confidence,
            cfg_scale=cfg_scale,
            allow_mask_prediction=True,
            eos_token_id=eos_token_id,
            record_trace=True,
            trace_topk=topk_save,
        ),
    )
    trajectory = output.traces[0]
    summary = trajectory.summary(eos_token_id)
    summary["nfe"] = float(output.nfe)
    return trajectory, summary
