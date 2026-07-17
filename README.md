# LLaDA × Seed Diffusion Section 3.2

Reproduces **Section 3.2 (Constrained-Order Training)** from [Seed Diffusion (arXiv 2508.02193)](https://arxiv.org/abs/2508.02193) and applies it to **LLaDA-8B-Instruct** on GSM8K.

---

## Background

LLaDA is a masked diffusion language model that generates tokens in arbitrary order via iterative unmasking. This flexibility degrades at large block sizes: when the model must commit many tokens simultaneously without seeing neighboring context, quality drops sharply.

Section 3.2 addresses this by fine-tuning the model on its own naturally-ordered generation trajectories, teaching it to prefer the orderings it finds most coherent.

---

## Method

**Pipeline:**
1. Generate N candidate trajectories per problem with `block_len = gen_len` (unconstrained order) and `temperature > 0`
2. Keep only trajectories that produce correct answers
3. Select the best trajectory per problem by maximizing `content_logprob_mean` (ELBO proxy, computed only over tokens before the first EOS)
4. Extract `(x_t, x_0)` training pairs from each trajectory
5. Fine-tune with LoRA using the trajectory's commit order as the mask sequence

**2×2 Ablation design:**

|  | Best-ELBO trajectory | Random trajectory |
|---|---|---|
| **Trajectory mask** (Section 3.2) | `proper_constrained` | `random_constrained` |
| **Random mask** (SFT control) | `sft_control` | `sft_random` |

- `proper_constrained` vs `sft_control`: isolates the trajectory mask (order signal)
- `proper_constrained` vs `random_constrained`: isolates ELBO selection
- `sft_random` vs baseline: verifies the negative control

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| gen_len | 256 | Covers p97 of GSM8K answer lengths |
| block_len (trajectory gen) | 256 | = gen_len; unconstrained order |
| temperature | 0.6 | Diversity for ELBO comparison |
| num_samples | 4 | P(≥2 correct) = 95.4% at 76% base accuracy |
| ELBO metric | content_logprob_mean | Tokens before first EOS only |
| LoRA rank | 64, q/k/v/o_proj | |
| epochs | 3 | |
| lr | 2e-5 | |
| batch size | 4, grad-accum 8 | effective batch 32 |

---

## Results (GSM8K, 500 test problems)

| Condition | block32 | block64 | block128 | block256 | drop (32→256) |
|---|---|---|---|---|---|
| baseline | 73.6% | 75.0% | 73.0% | 57.2% | -16.4% |
| **proper_constrained** | **75.2%** | **74.2%** | **72.0%** | **61.8%** | **-13.4%** |
| random_constrained | 73.8% | 72.4% | 70.8% | 61.0% | -12.8% |
| sft_control | 71.0% | 72.8% | 69.0% | 57.4% | -13.6% |
| sft_random | 73.0% | 71.6% | 68.2% | 56.6% | -16.4% |

---

## Analysis

### Trajectory mask is the primary driver

The trajectory mask contributes a consistent **+4.4%** at block256, regardless of which trajectory was selected:
- `proper_constrained`(61.8%) - `sft_control`(57.4%) = +4.4%
- `random_constrained`(61.0%) - `sft_random`(56.6%) = +4.4%

### Two fundamentally different mechanisms of slope reduction

The drop in slope is not equivalent across conditions:

| Condition | Mechanism | Assessment |
|---|---|---|
| proper/random_constrained | Lifts large-block accuracy; small-block unchanged | Genuine robustness improvement |
| sft_control | Suppresses small-block accuracy; large-block unchanged | Artifact, not robustness |

`proper_constrained` improves large-block performance (block256: 57.2% → 61.8%) without sacrificing small-block performance (block32: 73.6% → 75.2%). The SFT conditions, by contrast, reduce the slope by degrading the easier end of the curve.

This is confirmed by training loss: constrained conditions maintain high loss throughout (2.56 → 2.71), while SFT conditions converge to near-zero (1.14 → 0.04), indicating overfitting to the random-mask objective rather than learning anything transferable.

### ELBO selection has a smaller but real effect

`proper_constrained` outperforms `random_constrained` consistently across **all** block sizes by 0.8-1.8%, with a uniform direction. This is not noise-it is a systematic lift of the entire accuracy curve.

| | block32 | block64 | block128 | block256 |
|---|---|---|---|---|
| proper - random | +1.4% | +1.8% | +1.2% | +0.8% |

The effect is an order of magnitude smaller than the trajectory mask contribution, but the per-problem ELBO selection does add value.

### Unexpected finding: any correct trajectory suffices

The paper motivates ELBO selection as necessary for identifying the "most natural" generation order. Our results show that even a randomly-chosen correct trajectory achieves nearly the same robustness benefit. On GSM8K, the ordering differences between correct trajectories appear to carry similar training signal-the critical ingredient is using *a* trajectory mask, not finding the *best* one.

### Limitations

- Verified on GSM8K (arithmetic reasoning) only; generalization to other tasks is unknown
- LoRA fine-tuning may underestimate the ceiling effect
- 4 samples per problem limits ELBO comparison granularity
- Epoch-level overfitting was not systematically analyzed

---

## Directory Structure

```
pipelines/
  generate.py                   # Trajectory generation + greedy eval
  filter_correct.py             # Keep only correct-answer trajectories
  select_best_trajectory.py     # Per-problem ELBO selection (Section 3.2 core)
  prepare_constrained_pairs.py  # Extract (x_t, x_0) pairs with trajectory mask
  prepare_sft_control_pairs.py  # Same data, random mask (SFT ablation)
  constrained_train.py          # LoRA fine-tuning
  model_utils.py                # LLaDA model loading (AutoModel, not CausalLM)
  official_trace.py             # LLaDA sampler with trajectory recording

run_gsm8k.sh                    # End-to-end experiment script (Steps 0-6)
eval_results.py                 # Utility: summarize eval JSONL files
```

---

## Data

Data files are not included in this repository. Prepare them as follows:

```bash
mkdir -p data/prepared

# GSM8K training set (6000 problems)
# Download from https://huggingface.co/datasets/openai/gsm8k
# Expected format per line: {"problem": "...", "answer": "..."}
# Save to: data/prepared/gsm8k.jsonl

# Evaluation subset: 500 problems sampled from the GSM8K test split with random.seed(42)
# Save to: data/prepared/gsm8k_test_500.jsonl
```

---

## Usage

**Full experiment (Steps 0-6, requires 4 GPUs):**
```bash
nohup bash run_gsm8k.sh > run_gsm8k.log 2>&1 &
```

**Individual steps:**
```bash
# Step 1: Generate trajectories (single GPU)
CUDA_VISIBLE_DEVICES=0 python -m pipelines.generate \
    --jsonl data/prepared/gsm8k.jsonl \
    --out runs/trajectories.jsonl \
    --steps 128 --gen-len 256 --block-len 256 \
    --temperature 0.6 --num-samples 4 --use-chat-template

# Step 2: Filter + select
python -m pipelines.filter_correct \
    --inp runs/trajectories.jsonl --out runs/correct.jsonl --print-stats
python -m pipelines.select_best_trajectory \
    --inp runs/correct.jsonl --out runs/best_elbo.jsonl --group-by prompt_text

# Step 3: Prepare training pairs
python -m pipelines.prepare_constrained_pairs \
    --inp runs/best_elbo.jsonl --out runs/pairs.jsonl --samples-per-trajectory 10

# Step 4: Fine-tune
python -m pipelines.constrained_train \
    --train-jsonl runs/pairs.jsonl \
    --output-dir runs/checkpoints/my_model \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --lora-rank 64 --lora-target-modules q_proj,k_proj,v_proj,o_proj \
    --edit-frac 0.15 --epochs 3 --lr 2e-5 \
    --batch-size 4 --grad-accum 8 --mixed-precision bf16

# Step 5: Evaluate
python -m pipelines.generate \
    --jsonl data/prepared/gsm8k_test_500.jsonl \
    --out runs/eval_block256.jsonl \
    --steps 128 --gen-len 256 --block-len 256 --temperature 0.0 \
    --lora-path runs/checkpoints/my_model/final --use-chat-template

# Summarize
python eval_results.py runs/eval_block256.jsonl
```

---

## Key Implementation Notes

- **Use `AutoModel`, not `AutoModelForCausalLM`**: LLaDA is a bidirectional masked diffusion model; CausalLM breaks the loss computation.
- **Trajectory generation requires `block_len = gen_len`**: Any shorter block forces left-to-right order, making ELBO comparison meaningless.
- **ELBO selection is per-problem**: Compare N trajectories *for the same problem*; cross-problem ELBO comparison is meaningless.
- **EOS handling**: `content_logprob_mean` is computed only up to the first EOS token to avoid bias from EOS commit timing.
- **LoRA key prefix**: PEFT saves keys as `base_model.model.<path>`; strip the full prefix before loading or LoRA silently has no effect.

---

## Requirements

```bash
python -m pip install -r requirements.txt
```

Model: `GSAI-ML/LLaDA-8B-Instruct` (HuggingFace Hub)

## License

MIT. See [LICENSE](LICENSE).
