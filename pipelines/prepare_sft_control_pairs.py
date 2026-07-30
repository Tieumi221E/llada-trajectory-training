#!/usr/bin/env python3
"""
SFT control: same data as constrained (filtered_elbo.jsonl), same number of pairs,
but mask pattern is random (standard LLaDA diffusion masking) instead of
trajectory-derived. Used to ablate whether Section 3.2's improvement comes from
the constrained order signal or simply from fine-tuning on GSM8K.

Masking follows the standard forward process:
  t ~ Uniform(0, 1)
  p = (1 - eps) * t + eps
  each answer token independently masked with prob p

Output format is identical to prepare_constrained_pairs.py so it feeds directly
into constrained_train.py.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer  # type: ignore

MASK_TOKEN_STR = "<|mdm_mask|>"
LLADA_MASK_ID = 126336
EPS = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare SFT control pairs with random masking"
    )
    p.add_argument(
        "--inp",
        type=Path,
        required=True,
        help="Input trajectory JSONL (filtered_elbo.jsonl)",
    )
    p.add_argument("--out", type=Path, required=True, help="Output pairs JSONL")
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--samples-per-trajectory", type=int, default=10)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Randomly sample this many trajectories (0 = all)",
    )
    p.add_argument("--min-mask-ratio", type=float, default=0.05)
    p.add_argument("--max-mask-ratio", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def token_strings_to_ids(tokenizer, token_strings: List) -> List[int]:
    ids = []
    for tok in token_strings:
        if isinstance(tok, int):
            ids.append(tok)
        elif tok == MASK_TOKEN_STR:
            ids.append(LLADA_MASK_ID)
        else:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                ids.append(tid)
    return ids


def random_mask(
    answer_len: int, rng: random.Random, min_ratio: float, max_ratio: float
) -> tuple[list[bool], float]:
    """Sample t ~ Uniform, compute p=(1-eps)*t+eps, mask each token independently."""
    for _ in range(100):
        t = rng.random()
        p = (1.0 - EPS) * t + EPS
        mask_bools = [rng.random() < p for _ in range(answer_len)]
        ratio = sum(mask_bools) / answer_len
        if min_ratio <= ratio <= max_ratio:
            # ensure at least one masked
            if any(mask_bools):
                return mask_bools, ratio
    # fallback: mask exactly half
    half = max(1, answer_len // 2)
    idxs = rng.sample(range(answer_len), half)
    mask_bools = [i in set(idxs) for i in range(answer_len)]
    return mask_bools, sum(mask_bools) / answer_len


def extract_pairs(
    row: Dict,
    tokenizer,
    samples_per_traj: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
    rng: random.Random,
) -> List[Dict]:
    prompt_strs: List[str] = row.get("prompt", [])
    final_strs: List[str] = row.get("final", [])

    if not prompt_strs or not final_strs:
        return []

    prompt_len = len(prompt_strs)
    prompt_ids = token_strings_to_ids(tokenizer, prompt_strs)
    answer_strs = (
        final_strs
        if row.get("schema") == "dllm.trajectory.v1"
        else final_strs[prompt_len:]
    )
    answer_ids = token_strings_to_ids(tokenizer, answer_strs)

    if not answer_ids:
        return []

    sample_id_base = row.get("id", f"traj_{id(row)}")
    results = []
    for i in range(samples_per_traj):
        mask_bools, mask_ratio = random_mask(
            len(answer_ids), rng, min_mask_ratio, max_mask_ratio
        )
        results.append(
            {
                "id": f"{sample_id_base}#rnd{i}",
                "prompt_ids": prompt_ids,
                "answer_ids": answer_ids,
                "mask_bools": mask_bools,
                "mask_ratio": mask_ratio,
                "t_step": i,
            }
        )
    return results


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows = []
    with args.inp.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit and args.limit < len(rows):
        rows = rng.sample(rows, args.limit)
        print(f"Randomly sampled {len(rows)} trajectories from {args.inp}")
    else:
        print(f"Loaded {len(rows)} trajectories from {args.inp}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("w", encoding="utf-8") as fout:
        for row in rows:
            pairs = extract_pairs(
                row,
                tokenizer,
                samples_per_traj=args.samples_per_trajectory,
                min_mask_ratio=args.min_mask_ratio,
                max_mask_ratio=args.max_mask_ratio,
                rng=rng,
            )
            for pair in pairs:
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total += 1

    print(f"Wrote {total} pairs to {args.out}")


if __name__ == "__main__":
    main()
