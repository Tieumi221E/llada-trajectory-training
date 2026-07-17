#!/usr/bin/env python3
"""
Summarize evaluation results from generate.py outputs.

Usage:
  python eval_results.py runs/eval/baseline_block32.jsonl runs/eval/constrained_block32.jsonl ...

Or compare a sweep:
  python eval_results.py --sweep runs/eval/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_results(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: List[Dict]) -> Dict:
    if not rows:
        return {}
    correct = sum(1 for r in rows if r.get("summary", {}).get("correct"))
    latencies = [r["latency"] for r in rows if "latency" in r]
    block_len = rows[0].get("block_len", "?")
    lora = rows[0].get("lora_path", "")
    model_tag = "baseline" if not lora else "+3.2"
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "avg_latency": sum(latencies) / len(latencies) if latencies else None,
        "block_len": block_len,
        "model": model_tag,
        "tokens_per_sec": rows[0].get("gen_len", 128)
        / (sum(latencies) / len(latencies))
        if latencies
        else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", type=Path, help="Result JSONL files")
    p.add_argument(
        "--sweep", type=Path, default=None, help="Directory to scan for all *.jsonl"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths: List[Path] = list(args.files)
    if args.sweep:
        paths += sorted(args.sweep.glob("*.jsonl"))

    if not paths:
        print("No files provided.")
        return

    results = []
    for p in paths:
        rows = load_results(p)
        if not rows:
            continue
        s = summarize(rows)
        s["file"] = p.name
        results.append(s)

    # Print table
    print(
        f"\n{'File':<45} {'Model':<10} {'Block':>6} {'N':>5} {'Acc':>7} {'Latency':>9} {'tok/s':>8}"
    )
    print("-" * 95)
    for r in results:
        acc = f"{r['accuracy']:.1%}"
        lat = f"{r['avg_latency']:.2f}s" if r["avg_latency"] else "  -"
        tps = f"{r['tokens_per_sec']:.1f}" if r["tokens_per_sec"] else "  -"
        print(
            f"{r['file']:<45} {r['model']:<10} {str(r['block_len']):>6} "
            f"{r['n']:>5} {acc:>7} {lat:>9} {tps:>8}"
        )

    # Block-size comparison if multiple block_lens present
    block_lens = sorted(
        set(r["block_len"] for r in results if isinstance(r["block_len"], int))
    )
    if len(block_lens) > 1:
        print("\n── Block-size quality degradation ──")
        models = sorted(set(r["model"] for r in results))
        by = {(r["model"], r["block_len"]): r for r in results}
        ref_block = min(block_lens)
        for model in models:
            if (model, ref_block) not in by:
                continue
            ref_acc = by[(model, ref_block)]["accuracy"]
            print(f"\n  {model}  (ref block{ref_block} = {ref_acc:.1%})")
            for bl in block_lens:
                if (model, bl) not in by:
                    continue
                acc = by[(model, bl)]["accuracy"]
                delta = acc - ref_acc
                sign = "+" if delta >= 0 else ""
                print(f"    block{bl:>4}:  {acc:.1%}  ({sign}{delta:.1%})")


if __name__ == "__main__":
    main()
