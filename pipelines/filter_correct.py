#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only trajectories that yield the correct answer"
    )
    parser.add_argument("--inp", type=str, required=True, help="Input JSONL")
    parser.add_argument("--out", type=str, required=True, help="Destination JSONL")
    parser.add_argument(
        "--limit", type=int, default=0, help="Optional cap on processed rows (0 = all)"
    )
    parser.add_argument(
        "--print-stats", action="store_true", help="Print how many items were kept"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.inp)
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0

    with (
        src.open("r", encoding="utf-8") as fin,
        dst.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if args.limit and total >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            total += 1
            data = json.loads(line)
            summary = data.get("summary", {}) or {}
            if summary.get("correct"):
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept += 1

    if args.print_stats:
        print(f"Processed: {total} rows")
        print(f"Kept: {kept} rows")


if __name__ == "__main__":
    main()
