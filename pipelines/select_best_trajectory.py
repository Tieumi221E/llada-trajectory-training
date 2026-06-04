#!/usr/bin/env python3
"""
Per-problem ELBO selection: group trajectories by problem_id,
select the one with highest logprob_mean (ELBO proxy) per problem.

Used in proper Section 3.2 implementation where N trajectories are
generated per problem using free-order diffusion (block_len=gen_len,
temperature>0), and the best-ordering one is selected for training.

Usage:
  # Select best ELBO per problem
  python -m pipelines.select_best_trajectory \
    --inp runs/gsm8k_proper/trajectories/correct_multi.jsonl \
    --out runs/gsm8k_proper/trajectories/best_elbo.jsonl

  # Random selection (control group)
  python -m pipelines.select_best_trajectory \
    --inp runs/gsm8k_proper/trajectories/correct_multi.jsonl \
    --out runs/gsm8k_proper/trajectories/random_order.jsonl \
    --random-select
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--random-select", action="store_true",
                   help="Randomly pick one trajectory per problem instead of best ELBO")
    p.add_argument("--group-by", type=str, default="prompt_text",
                   choices=["problem_id", "prompt_text"],
                   help="Field to group trajectories by (prompt_text is safer for multi-file runs)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def elbo_score(record: dict) -> float:
    summary = record.get("summary", {}) or {}
    # Prefer content_logprob_mean (excludes EOS padding, more accurate ordering signal)
    score = summary.get("content_logprob_mean", None)
    if score is not None and summary.get("content_commit_tokens", 0) > 0:
        return float(score)
    # Fallback: full logprob_mean (older trajectories without content_logprob_mean)
    score = summary.get("logprob_mean", None)
    if score is not None:
        return float(score)
    lp_sum = summary.get("logprob_sum", None)
    commit = summary.get("commit_tokens", None)
    if lp_sum is not None and commit and commit > 0:
        return float(lp_sum) / float(commit)
    return float("-inf")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    groups: dict[str, list[dict]] = defaultdict(list)
    with args.inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if args.group_by == "prompt_text":
                key = record.get("prompt_text", record.get("problem_id", record.get("id", "")))
            else:
                key = str(record.get("problem_id", record.get("id", "")))
            groups[key].append(record)

    print(f"Loaded {sum(len(v) for v in groups.values())} trajectories across {len(groups)} problems")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as fout:
        for pid, candidates in groups.items():
            if args.random_select:
                chosen = rng.choice(candidates)
            else:
                chosen = max(candidates, key=elbo_score)
            fout.write(json.dumps(chosen, ensure_ascii=False) + "\n")
            written += 1

    mode = "random" if args.random_select else "best ELBO"
    print(f"Wrote {written} trajectories ({mode}) to {args.out}")

    if not args.random_select:
        # Print ELBO stats
        scores_per_problem = [
            [elbo_score(c) for c in candidates]
            for candidates in groups.values()
            if len(candidates) > 1
        ]
        if scores_per_problem:
            variances = [max(s) - min(s) for s in scores_per_problem]
            avg_var = sum(variances) / len(variances)
            max_var = max(variances)
            print(f"ELBO spread (max-min per problem): avg={avg_var:.4f}, max={max_var:.4f}")
            print(f"  (larger spread = ELBO selection more meaningful)")


if __name__ == "__main__":
    main()
