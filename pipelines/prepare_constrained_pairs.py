#!/usr/bin/env python3
"""
Section 3.2: Extract (x_i, x_0) training pairs from filtered diffusion trajectories.

Input JSONL (from generate.py + filter steps):
  { "prompt": [tok_str, ...], "final": [tok_str, ...], "steps": [...], ... }

Output JSONL:
  {
    "id": "...",
    "prompt_ids": [int, ...],       # prompt token IDs
    "answer_ids": [int, ...],       # clean answer token IDs (x_0)
    "mask_bools": [bool, ...],      # which answer positions are still masked at step i (x_i)
    "mask_ratio": float,            # fraction of answer positions masked
    "t_step": int                   # which diffusion step index this came from
  }

Usage:
  python -m pipelines.prepare_constrained_pairs \
    --inp runs/gsm8k_head100/block32/trajectories/teacher_logprob.jsonl \
    --out data/constrained_pairs/gsm8k_head100.jsonl \
    --samples-per-trajectory 3
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer  # type: ignore

MASK_TOKEN_STR = "<|mdm_mask|>"
LLADA_MASK_ID = 126336


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract (x_i, x_0) pairs from diffusion trajectories")
    p.add_argument("--inp", type=Path, required=True, help="Input trajectory JSONL")
    p.add_argument("--out", type=Path, required=True, help="Output pairs JSONL")
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                   help="Tokenizer name (needed to convert token strings → IDs)")
    p.add_argument("--samples-per-trajectory", type=int, default=3,
                   help="How many (x_i, x_0) pairs to sample per trajectory")
    p.add_argument("--min-step", type=int, default=1,
                   help="Skip very early steps (nearly all masked)")
    p.add_argument("--min-mask-ratio", type=float, default=0.05,
                   help="Skip steps with fewer than this fraction of masks")
    p.add_argument("--max-mask-ratio", type=float, default=0.95,
                   help="Skip steps with more than this fraction of masks")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_trajectories(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No records in {path}")
    return rows


def token_strings_to_ids(tokenizer, token_strings: List[str]) -> List[int]:
    """Convert token strings (from tokenizer.convert_ids_to_tokens) back to IDs."""
    ids = []
    for tok in token_strings:
        if tok == MASK_TOKEN_STR:
            ids.append(LLADA_MASK_ID)
        else:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                ids.append(tid)
            # skip unknown tokens
    return ids


def extract_pairs(
    row: Dict,
    tokenizer,
    samples_per_traj: int,
    min_step: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
    rng: random.Random,
) -> List[Dict]:
    prompt_strs: List[str] = row.get("prompt", [])
    final_strs: List[str] = row.get("final", [])
    steps: List[Dict] = row.get("steps", [])

    if not prompt_strs or not final_strs or not steps:
        return []

    prompt_len = len(prompt_strs)

    # prompt_ids and answer_ids (x_0)
    prompt_ids = token_strings_to_ids(tokenizer, prompt_strs)
    answer_strs = final_strs[prompt_len:]
    answer_ids = token_strings_to_ids(tokenizer, answer_strs)

    if not answer_ids:
        return []

    # Collect candidate steps
    candidates: List[Tuple[int, List[bool], float]] = []
    for step_idx, step in enumerate(steps):
        if step_idx < min_step:
            continue
        mask_list: List[bool] = step.get("mask", [])
        if len(mask_list) <= prompt_len:
            continue
        # mask_bools for answer positions only
        ans_mask = mask_list[prompt_len:]
        # truncate to answer_ids length
        ans_mask = ans_mask[: len(answer_ids)]
        if len(ans_mask) < len(answer_ids):
            # pad with False if step tokens are shorter
            ans_mask = ans_mask + [False] * (len(answer_ids) - len(ans_mask))
        mask_ratio = float(sum(1 for m in ans_mask if m)) / max(len(ans_mask), 1)
        if mask_ratio < min_mask_ratio or mask_ratio > max_mask_ratio:
            continue
        candidates.append((step_idx, ans_mask, mask_ratio))

    if not candidates:
        return []

    # Sample steps
    n = min(samples_per_traj, len(candidates))
    chosen = rng.sample(candidates, k=n)

    sample_id_base = row.get("id", f"traj_{id(row)}")
    results = []
    for step_idx, mask_bools, mask_ratio in chosen:
        results.append({
            "id": f"{sample_id_base}#step{step_idx}",
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
            "mask_bools": mask_bools,
            "mask_ratio": mask_ratio,
            "t_step": step_idx,
        })
    return results


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows = load_trajectories(args.inp)
    print(f"Loaded {len(rows)} trajectories from {args.inp}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("w", encoding="utf-8") as fout:
        for row in rows:
            pairs = extract_pairs(
                row,
                tokenizer,
                samples_per_traj=args.samples_per_trajectory,
                min_step=args.min_step,
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
