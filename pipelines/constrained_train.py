#!/usr/bin/env python3
"""
Section 3.2: Constrained-Order Fine-tuning for LLaDA.

Implements Eq. 9 from Seed Diffusion (2508.02193):
  L_c(θ) = E_{τ~U(T), (x_i, x_0)∈τ} [ -λ(x_i) * log p_θ(x_0 | f(x_i)) ]

Where:
- x_i is an intermediate trajectory step (partial masked state)
- x_0 is the final clean output
- λ(x_i) = 1 - mask_ratio  (lower noise level → higher weight)
- f(x_i) is the edit augmentation function (Remark 3.1): random token substitutions
  on already-committed (non-masked) answer positions, forcing the model to re-evaluate
  all positions rather than blindly trusting the context.

Training pairs are produced by prepare_constrained_pairs.py.

Usage:
  python -m pipelines.constrained_train \
    --train-jsonl runs/gsm8k/pairs/proper_constrained.jsonl \
    --output-dir runs/checkpoints/gsm8k_proper_constrained \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --lora-rank 32 --lora-target-modules q_proj,k_proj,v_proj,o_proj \
    --epochs 5 --lr 2e-5 --batch-size 4 --grad-accum 8
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup  # type: ignore

LLADA_MASK_ID = 126336
LLADA_EOS_ID = 126081  # <|endoftext|> - first occurrence is the real terminator
# Random substitution token range: avoid special tokens at vocab boundaries
EDIT_TOKEN_MIN = 100
EDIT_TOKEN_MAX = 126000


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class TrainingPair:
    prompt_ids: List[int]
    answer_ids: List[int]
    mask_bools: List[bool]
    mask_ratio: float


class ConstrainedPairDataset(Dataset):
    def __init__(self, path: Path):
        self.samples: List[TrainingPair] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.samples.append(
                    TrainingPair(
                        prompt_ids=list(d["prompt_ids"]),
                        answer_ids=list(d["answer_ids"]),
                        mask_bools=[bool(b) for b in d["mask_bools"]],
                        mask_ratio=float(d.get("mask_ratio", 0.5)),
                    )
                )
        if not self.samples:
            raise RuntimeError(f"No samples in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TrainingPair:
        return self.samples[idx]


def collate_fn(
    batch: List[TrainingPair],
    pad_id: int,
    max_length: int,
    edit_frac: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """
    For each (prompt_ids, answer_ids, mask_bools):
      - Build f(x_i): trajectory masks + edit augmentation on committed positions
        * Trajectory-masked positions → LLADA_MASK_ID
        * Committed (non-masked) answer positions → with prob edit_frac, random substitution
      - Labels: x_0 tokens at ALL corrupted positions (trajectory-masked + edited)
      - Weights: 1 - mask_ratio  (λ in Eq. 9)

    The edit augmentation (f) prevents the model from blindly trusting the committed
    tokens in x_i, as noted in Remark 3.1 of the paper.
    """
    B = len(batch)
    max_len = min(max(len(s.prompt_ids) + len(s.answer_ids) for s in batch), max_length)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    weights = torch.zeros(B, dtype=torch.float)

    for i, s in enumerate(batch):
        prompt_len = len(s.prompt_ids)
        answer_len = len(s.answer_ids)

        # Build f(x_i): start with prompt (always clean), then answer
        x_i = list(s.prompt_ids)
        for j in range(min(answer_len, max_length - prompt_len)):
            pos = prompt_len + j
            if pos >= max_length:
                break
            if s.mask_bools[j]:
                # Trajectory-masked position → MASK token
                x_i.append(LLADA_MASK_ID)
                labels[i, pos] = s.answer_ids[j]
            elif edit_frac > 0.0 and random.random() < edit_frac:
                # Edit augmentation f: substitute committed token with random token
                rand_tok = random.randint(EDIT_TOKEN_MIN, EDIT_TOKEN_MAX)
                x_i.append(rand_tok)
                labels[i, pos] = s.answer_ids[j]  # still predict original x_0
            else:
                x_i.append(s.answer_ids[j])

        seq_len = min(len(x_i), max_length)
        input_ids[i, :seq_len] = torch.tensor(x_i[:seq_len], dtype=torch.long)
        attention_mask[i, :seq_len] = 1

        # Suppress supervision on EOS padding: only the first EOS position in the
        # answer is a meaningful learning target; all subsequent EOS tokens are
        # padding artifacts from fixed-length generation and would otherwise
        # dominate the loss signal (91 % of MATH answer tokens are EOS padding).
        first_eos = next(
            (j for j, t in enumerate(s.answer_ids) if t == LLADA_EOS_ID),
            len(s.answer_ids),
        )
        for j in range(first_eos + 1, min(len(s.answer_ids), max_length - prompt_len)):
            labels[i, prompt_len + j] = -100

        # λ(x_i) = 1 - mask_ratio
        weights[i] = 1.0 - s.mask_ratio

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "weights": weights,
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def masked_diffusion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy at masked positions only, weighted per sample by λ(x_i).

    logits:  (B, L, V)
    labels:  (B, L)  -100 = ignore
    weights: (B,)
    """
    B, L, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)

    valid = labels != -100  # (B, L)
    safe_labels = labels.clone()
    safe_labels[~valid] = 0

    token_nll = -torch.gather(log_probs, -1, safe_labels.unsqueeze(-1)).squeeze(
        -1
    )  # (B, L)
    token_nll = token_nll * valid.float()

    per_sample_count = valid.sum(dim=1).float().clamp(min=1)
    per_sample_loss = token_nll.sum(dim=1) / per_sample_count  # (B,)

    w = weights.to(logits.device)
    has_targets = (per_sample_count > 0).float()
    total_weight = (w * has_targets).sum().clamp(min=1e-8)
    return (per_sample_loss * w * has_targets).sum() / total_weight


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Constrained-order fine-tuning for LLaDA (Section 3.2)"
    )
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--eval-jsonl", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct")

    # LoRA (optional)
    p.add_argument(
        "--lora-rank", type=int, default=0, help="LoRA rank (0 = full fine-tuning)"
    )
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target module names",
    )

    # Training
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument(
        "--warmup-frac",
        type=float,
        default=0.1,
        help="Warmup as fraction of total optimizer steps",
    )
    p.add_argument("--max-length", type=int, default=640)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument(
        "--mixed-precision", choices=["bf16", "fp16", "none"], default="bf16"
    )
    p.add_argument("--gradient-checkpointing", action="store_true")

    # Edit augmentation f(x_i)
    p.add_argument(
        "--edit-frac",
        type=float,
        default=0.15,
        help="Fraction of committed answer tokens to randomly substitute (f augmentation)",
    )

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    dtype = (
        torch.bfloat16
        if args.mixed_precision == "bf16"
        else torch.float16
        if args.mixed_precision == "fp16"
        else torch.float32
    )

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if args.lora_rank > 0:
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore
        except ImportError:
            raise RuntimeError("LoRA requires `pip install peft`")
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        w = batch.pop("weights")
        lb = batch.pop("labels")
        with torch.autocast(
            device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None
        ):
            logits = model(**batch).logits
            loss = masked_diffusion_loss(logits, lb, w)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_ds = ConstrainedPairDataset(args.train_jsonl)
    eval_ds = ConstrainedPairDataset(args.eval_jsonl) if args.eval_jsonl else None
    print(
        f"Train: {len(train_ds)} pairs"
        + (f"  Eval: {len(eval_ds)} pairs" if eval_ds else "")
    )

    def _collate(batch):
        return collate_fn(batch, pad_id, args.max_length, args.edit_frac)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        pin_memory=True,
    )
    eval_loader = (
        DataLoader(
            eval_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate,
            pin_memory=True,
        )
        if eval_ds
        else None
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = (len(train_loader) * args.epochs) // max(args.grad_accum, 1)
    warmup_steps = max(10, int(total_steps * args.warmup_frac))
    print(
        f"Optimizer steps: total={total_steps}, warmup={warmup_steps} "
        f"(batches/epoch={len(train_loader)}, epochs={args.epochs}, grad_accum={args.grad_accum})"
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision == "fp16")
    autocast_dtype = (
        torch.bfloat16
        if args.mixed_precision == "bf16"
        else torch.float16
        if args.mixed_precision == "fp16"
        else None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            weights = batch.pop("weights")
            labels = batch.pop("labels")

            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                logits = model(**batch).logits
                loss = masked_diffusion_loss(logits, labels, weights) / args.grad_accum

            if args.mixed_precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                if args.mixed_precision == "fp16":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % 50 == 0:
                print(
                    f"[ep{epoch + 1} step{global_step}] loss={loss.item() * args.grad_accum:.4f}"
                )

            global_step += 1

            if args.save_steps and global_step % args.save_steps == 0:
                ckpt = args.output_dir / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                print(f"Saved checkpoint to {ckpt}")

            if eval_loader and args.eval_steps and global_step % args.eval_steps == 0:
                eval_loss = evaluate(model, eval_loader, device, autocast_dtype)
                print(f"[step {global_step}] eval_loss={eval_loss:.4f}")

        print(f"Epoch {epoch + 1}/{args.epochs} done")

    final = args.output_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"Saved final model to {final}")


if __name__ == "__main__":
    main()
