from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TopKEntry:
    token: str
    prob: float


@dataclass
class MaskedDist:
    index: int
    topk: List[TopKEntry]


@dataclass
class TrajectoryStep:
    t: int
    tokens: List[str]
    mask: List[bool]
    dists: List[MaskedDist]
    commit: List[bool]
    block: Optional[int] = None
    meta: Optional[Dict] = None


@dataclass
class TrajectorySample:
    prompt: List[str]
    steps: List[TrajectoryStep]
    final: List[str]

    def to_dict(self) -> Dict:
        return {
            "prompt": self.prompt,
            "final": self.final,
            "steps": [
                {
                    "t": st.t,
                    "tokens": st.tokens,
                    "mask": st.mask,
                    "commit": st.commit,
                    "block": st.block,
                    "meta": st.meta,
                    "dists": [
                        {
                            "index": md.index,
                            "topk": [
                                {"token": e.token, "prob": e.prob} for e in md.topk
                            ],
                        }
                        for md in st.dists
                    ],
                }
                for st in self.steps
            ],
        }

    @staticmethod
    def from_dict(d: Dict) -> "TrajectorySample":
        steps: List[TrajectoryStep] = []
        for sd in d.get("steps", []):
            mds: List[MaskedDist] = []
            for md in sd.get("dists", []):
                topk = [
                    TopKEntry(e["token"], float(e["prob"])) for e in md.get("topk", [])
                ]
                mds.append(MaskedDist(index=int(md["index"]), topk=topk))
            steps.append(
                TrajectoryStep(
                    t=int(sd["t"]),
                    tokens=list(sd.get("tokens", [])),
                    mask=[bool(x) for x in sd.get("mask", [])],
                    dists=mds,
                    commit=[bool(x) for x in sd.get("commit", [])],
                    block=sd.get("block"),
                    meta=sd.get("meta"),
                )
            )
        return TrajectorySample(
            prompt=list(d.get("prompt", [])),
            steps=steps,
            final=list(d.get("final", [])),
        )


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
    device = prompt_ids.device

    Lp = int(prompt_ids.shape[1])
    total_len = Lp + int(gen_length)
    x = torch.full((1, total_len), int(mask_id), dtype=torch.long, device=device)
    x[:, :Lp] = prompt_ids.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0 and gen_length > 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0 and steps > 0
    steps_per_block = steps // num_blocks

    steps_out: List[TrajectoryStep] = []
    per_step_pmax_mean: List[float] = []
    per_step_margin_mean: List[float] = []
    per_step_entropy_topk_mean: List[float] = []
    per_step_commit_count: List[int] = []

    logprob_sum = 0.0
    logprob_weighted_sum = 0.0
    commit_token_count = 0
    weighted_step_count = 0.0
    position_logprob: dict[int, float] = {}  # position → logprob at commit time

    def batch_logits(inp: torch.Tensor) -> torch.Tensor:
        if cfg_scale > 0.0:
            assert inp.shape[1] == x.shape[1]
            pidx = prompt_index[0]
            un_x = inp.clone()
            un_x[:, pidx] = mask_id
            x_ = torch.cat([inp, un_x], dim=0)
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            return logits[:, : inp.shape[1]]
        return model(inp).logits

    total_iters = 0
    for block_idx in range(num_blocks):
        gen_start = Lp + block_idx * block_length
        gen_end = Lp + (block_idx + 1) * block_length
        block_mask_index = x[:, gen_start:gen_end] == mask_id
        mask_num = block_mask_index.sum(dim=1, keepdim=True)

        base = mask_num // steps_per_block
        remainder = mask_num % steps_per_block
        num_transfer_tokens = (
            torch.zeros(
                mask_num.size(0), steps_per_block, device=device, dtype=torch.int64
            )
            + base
        )
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1

        for s in range(steps_per_block):
            total_iters += 1
            mask_index = x == mask_id
            logits = batch_logits(x)

            if temperature and temperature > 0:
                logits64 = logits.to(torch.float64)
                noise = torch.rand_like(logits64, dtype=torch.float64)
                gumbel_noise = (-torch.log(noise)) ** temperature
                logits_with_noise = logits64.exp() / gumbel_noise
                x0 = torch.argmax(logits_with_noise, dim=-1)
                p = F.softmax(logits64.to(torch.float32), dim=-1)
            else:
                x0 = torch.argmax(logits, dim=-1)
                p = F.softmax(logits, dim=-1)

            log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)

            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )

            masked_dists: List[MaskedDist] = []
            margins: List[float] = []
            ent_topk_list: List[float] = []
            for i in range(x.shape[1]):
                if mask_index[0, i]:
                    li = logits[0, i]
                    topk = torch.topk(li, k=min(topk_save, li.shape[-1]))
                    probs = F.softmax(topk.values, dim=-1).tolist()
                    toks = [
                        tokenizer.convert_ids_to_tokens(int(idx))
                        for idx in topk.indices.tolist()
                    ]
                    masked_dists.append(
                        MaskedDist(
                            index=i,
                            topk=[
                                TopKEntry(token=t, prob=float(pv))
                                for t, pv in zip(toks, probs)
                            ],
                        )
                    )
                    if len(probs) >= 2:
                        margins.append(float(probs[0] - probs[1]))
                    elif probs:
                        margins.append(float(probs[0]))
                    ent = 0.0
                    for pv in probs:
                        if pv > 0:
                            ent += -float(pv) * float(np.log(max(pv, 1e-12)))
                    ent_topk_list.append(float(ent))

            conf = x0_p.clone()
            conf[:, gen_end:] = -float("inf")

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for j in range(conf.shape[0]):
                conf_j = conf[j]
                conf_j = torch.where(
                    mask_index[j], conf_j, torch.tensor(-float("inf"), device=device)
                )
                k = int(num_transfer_tokens[j, s].item())
                if k > 0:
                    _, select_index = torch.topk(conf_j, k=k)
                    transfer_index[j, select_index] = True

            tokens_before = x.clone()
            mask_before = mask_index.clone()

            x0_eff = torch.where(mask_index, x0, x)
            x[transfer_index] = x0_eff[transfer_index]

            tok_ids = tokens_before
            tok_strs = [
                tokenizer.convert_ids_to_tokens(int(t)) for t in tok_ids[0].tolist()
            ]
            mask_list = [bool(m) for m in mask_before[0].tolist()]
            commit_vec = [bool(transfer_index[0, i].item()) for i in range(x.shape[1])]

            masked_positions = [i for i, m in enumerate(mask_list) if m]
            pm_list = [float(x0_p[0, i].item()) for i in masked_positions]
            pmax_mean = float(sum(pm_list) / max(1, len(pm_list)))
            margin_mean = float(sum(margins) / max(1, len(margins))) if margins else 0.0
            ent_topk_mean = (
                float(sum(ent_topk_list) / max(1, len(ent_topk_list)))
                if ent_topk_list
                else 0.0
            )
            commit_count = int(sum(1 for b in commit_vec if b))

            if commit_count > 0:
                step_index = float(total_iters - 1)
                weighted_step_count += float(commit_count) * step_index
                for i, did_commit in enumerate(commit_vec):
                    if not did_commit:
                        continue
                    token_id = int(x0[0, i].item())
                    lp = float(log_probs[0, i, token_id].item())
                    logprob_sum += lp
                    logprob_weighted_sum += lp * step_index
                    commit_token_count += 1
                    position_logprob[i] = lp

            steps_out.append(
                TrajectoryStep(
                    t=total_iters - 1,
                    tokens=tok_strs,
                    mask=mask_list,
                    dists=masked_dists,
                    commit=commit_vec,
                    block=int(block_length),
                    meta={
                        "pmax_mean": pmax_mean,
                        "masked_count": len(masked_positions),
                        "margin_mean": margin_mean,
                        "entropy_topk_mean": ent_topk_mean,
                        "commit_count": commit_count,
                    },
                )
            )

            per_step_pmax_mean.append(pmax_mean)
            per_step_margin_mean.append(margin_mean)
            per_step_entropy_topk_mean.append(ent_topk_mean)
            per_step_commit_count.append(commit_count)

    final_ids = x[0].tolist()
    final_tok_strs = [tokenizer.convert_ids_to_tokens(int(t)) for t in final_ids]

    # Content-only logprob: only positions before the first EOS token
    EOS_ID = 126081
    first_eos_pos = next(
        (i for i, t in enumerate(final_ids) if int(t) == EOS_ID), len(final_ids)
    )
    content_lp_sum = sum(
        lp for pos, lp in position_logprob.items() if pos < first_eos_pos
    )
    content_count = sum(1 for pos in position_logprob if pos < first_eos_pos)

    sample = TrajectorySample(
        prompt=[
            tokenizer.convert_ids_to_tokens(int(t)) for t in prompt_ids[0].tolist()
        ],
        steps=steps_out,
        final=final_tok_strs,
    )

    total_to_generate = int(gen_length)
    cum = 0
    cum_series: List[int] = []
    for c in per_step_commit_count:
        cum += c
        cum_series.append(min(cum, total_to_generate))
    if steps_out:
        auc_progress = float(sum(cum_series) / (total_to_generate * len(steps_out)))
    else:
        auc_progress = 0.0
    idle_steps = int(sum(1 for c in per_step_commit_count if c == 0))
    idle_ratio = float(idle_steps / max(1, len(per_step_commit_count)))
    pmax_slope = 0.0
    if per_step_pmax_mean:
        pmax_slope = float(per_step_pmax_mean[-1] - per_step_pmax_mean[0])
    margin_step_mean = float(
        sum(per_step_margin_mean) / max(1, len(per_step_margin_mean))
    )
    ent_step_mean = float(
        sum(per_step_entropy_topk_mean) / max(1, len(per_step_entropy_topk_mean))
    )

    summary = {
        "nfe": float(len(steps_out)),
        "auc_progress": auc_progress,
        "idle_ratio": idle_ratio,
        "pmax_slope": pmax_slope,
        "margin_mean": margin_step_mean,
        "entropy_topk_mean": ent_step_mean,
        "logprob_sum": float(logprob_sum),
        "logprob_mean": float(logprob_sum / max(commit_token_count, 1)),
        "commit_tokens": int(commit_token_count),
        "logprob_weighted": float(logprob_weighted_sum),
        "weighted_step_count": float(weighted_step_count),
        "content_logprob_mean": float(content_lp_sum / max(content_count, 1)),
        "content_commit_tokens": int(content_count),
    }
    return sample, summary
