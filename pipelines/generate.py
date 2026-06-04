#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer  # type: ignore

from pipelines.official_trace import generate_with_trace
from pipelines.model_utils import load_diffusion_model


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate teacher trajectories using official LLaDA sampler (with trace)")
    p.add_argument("--jsonl", type=str, required=True, help="Input JSONL with fields: prompt, answer")
    p.add_argument("--out", type=str, required=True, help="Output JSONL of trajectories")
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--gen-len", type=int, default=128)
    p.add_argument("--block-len", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"])
    p.add_argument("--mask-id", type=int, default=126336)
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--keep-only-correct", action="store_true", help="Only write samples judged correct")
    p.add_argument("--min-pmax", type=float, default=0.0, help="Min average pmax across steps")
    p.add_argument("--min-auc", type=float, default=0.0, help="Min normalized AUC of commit progress [0,1]")
    p.add_argument("--max-idle", type=float, default=1.0, help="Max ratio of steps with zero commits")
    p.add_argument("--min-margin", type=float, default=0.0, help="Min mean top-1 minus top-2 prob margin")
    p.add_argument("--min-pmax-slope", type=float, default=-1.0, help="Min increase in pmax from first to last step")
    p.add_argument("--topk-save", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=1, help="Number of trajectories to sample per problem (for diverse ordering)")
    p.add_argument("--lora-path", type=str, default="", help="Optional LoRA adapter path")
    p.add_argument("--answer-format", type=str, default="gsm8k", choices=["gsm8k", "math"],
                   help="Answer format: gsm8k (#### number) or math (\\boxed{})")
    return p.parse_args()


def load_rows(path: str, limit: int = 0) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_gold_answer(ans: str) -> Optional[str]:
    m = re.search(r"####\s*([^\n#]+)", ans)
    if not m:
        return None
    return m.group(1).strip()


def extract_last_number(text: str) -> Optional[str]:
    matches = re.findall(r"\$?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1]


def normalize_money(s: str) -> str:
    s = s.strip().lstrip("$")
    if re.match(r"^\d+\.0+$", s):
        s = s.split(".")[0]
    return s


def extract_boxed(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{} in text, handling nested braces."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    depth, start = 0, idx + len(r"\boxed{")
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def check_math_correct(pred_text: str, gold_solution: str) -> bool:
    """Use math_verify to compare predicted answer with gold solution."""
    try:
        from math_verify import verify, parse  # type: ignore
        pred_boxed = extract_boxed(pred_text)
        gold_boxed = extract_boxed(gold_solution)
        if pred_boxed is None or gold_boxed is None:
            return False
        return bool(verify(parse(gold_boxed), parse(pred_boxed)))
    except Exception:
        return False


def decode_suffix_text(tokenizer, full_ids: List[int], prompt_len: int) -> str:
    ids = full_ids[prompt_len:]
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=True)


def main():
    args = build_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_diffusion_model(
        args.model,
        device=device,
        lora_path=args.lora_path,
        merge_lora=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    rows = load_rows(args.jsonl, limit=args.limit)

    kept = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for row_idx, row in enumerate(rows):
            prompt_raw = str(row.get("prompt", "")).strip()
            answer_raw = str(row.get("answer", ""))
            if not prompt_raw:
                continue

            if args.use_chat_template:
                messages = [{"role": "user", "content": prompt_raw}]
                prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                prompt_text = prompt_raw

            prompt_ids = tokenizer(prompt_text)["input_ids"]
            x = torch.tensor(prompt_ids, device=device).unsqueeze(0)

            # Use row's id field if available, else use input filename + row index for uniqueness
            _src_stem = Path(args.jsonl).stem
            problem_id = row.get("id", f"{_src_stem}_{row_idx}")

            for sample_idx in range(args.num_samples):
                if args.num_samples > 1:
                    torch.manual_seed(sample_idx * 1000 + row_idx)

                t0 = __import__("time").perf_counter()
                traj, summ = generate_with_trace(
                    model,
                    tokenizer,
                    x,
                    steps=args.steps,
                    gen_length=args.gen_len,
                    block_length=args.block_len,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                    topk_save=args.topk_save,
                )
                latency = __import__("time").perf_counter() - t0

                # Decode readable final text (suffix only)
                final_ids_raw = [tokenizer.convert_tokens_to_ids(t) for t in traj.final]
                final_ids = [int(i) for i in final_ids_raw if isinstance(i, int) and i >= 0]
                suffix_text = decode_suffix_text(tokenizer, final_ids, prompt_len=len(prompt_ids))

                # Correctness check
                if args.answer_format == "math":
                    correct = check_math_correct(suffix_text, answer_raw)
                else:
                    gold_val = parse_gold_answer(answer_raw) or ""
                    pred_last = extract_last_number(suffix_text) or ""
                    pred_norm = normalize_money(pred_last)
                    gold_norm = normalize_money(gold_val) if gold_val else ""
                    correct = bool(gold_norm) and (pred_norm == gold_norm)

                # Aggregate average confidence across steps if present
                pmax_vals: List[float] = []
                margin_vals: List[float] = []
                for st in traj.steps:
                    if st.meta:
                        if "pmax_mean" in st.meta:
                            try:
                                pmax_vals.append(float(st.meta["pmax_mean"]))
                            except Exception:
                                pass
                        if "margin_mean" in st.meta:
                            try:
                                margin_vals.append(float(st.meta["margin_mean"]))
                            except Exception:
                                pass
                avg_pmax = float(sum(pmax_vals) / max(1, len(pmax_vals))) if pmax_vals else 0.0
                avg_margin = float(sum(margin_vals) / max(1, len(margin_vals))) if margin_vals else 0.0

                # Build record
                d = traj.to_dict()
                d["prompt_text"] = prompt_text
                d["final_text_suffix"] = suffix_text
                d["summary"] = {**summ, "correct": bool(correct), "avg_pmax": avg_pmax, "avg_margin": avg_margin}
                d["latency"] = round(latency, 3)
                d["gen_len"] = args.gen_len
                d["block_len"] = args.block_len
                d["task"] = row.get("task")
                d["problem_id"] = problem_id
                d["sample_idx"] = sample_idx

                # Filtering
                if args.keep_only_correct and not correct:
                    continue
                if avg_pmax < args.min_pmax:
                    continue
                if summ.get("auc_progress", 0.0) < args.min_auc:
                    continue
                if summ.get("idle_ratio", 0.0) > args.max_idle:
                    continue
                if avg_margin < args.min_margin:
                    continue
                if summ.get("pmax_slope", 0.0) < args.min_pmax_slope:
                    continue

                if args.lora_path:
                    d["lora_path"] = args.lora_path
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                kept += 1

    print(f"wrote {kept} trajectories to {args.out}")


if __name__ == "__main__":
    main()
