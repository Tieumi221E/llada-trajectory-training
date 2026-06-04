#!/bin/bash
# GSM8K experiment: Section 3.2 × 2x2 ablation on GPU 0-3
# See README.md for hyperparameter rationale.
set -e
cd "$(dirname "$0")"

mkdir -p runs/gsm8k/{trajectories,pairs,logs}
mkdir -p runs/checkpoints
mkdir -p runs/gsm8k_eval

log() { echo "=== $* $(date '+%H:%M:%S') ==="; }

# ── Step 0: Split training set (6000 problems → 4 × 1500) ────────────
log "Step 0: split data"
python3 -c "
lines = open('data/prepared/gsm8k.jsonl').readlines()
assert len(lines) == 6000, f'Expected 6000, got {len(lines)}'
per = 1500
for i in range(4):
    part = lines[i*per:(i+1)*per]
    open(f'data/prepared/gsm8k_part{i+1}.jsonl','w').writelines(part)
    print(f'  part{i+1}: {len(part)} problems')
"

# ── Step 1: Trajectory generation (4 GPUs in parallel) ───────────────
log "Step 1: generate trajectories"
GEN_FLAGS="--steps 128 --gen-len 256 --block-len 256 \
           --temperature 0.6 --topk-save 1 --num-samples 4 \
           --use-chat-template"

for i in 1 2 3 4; do
    gpu=$((i-1))  # GPUs 0,1,2,3
    CUDA_VISIBLE_DEVICES=$gpu python -m pipelines.generate \
        --jsonl data/prepared/gsm8k_part${i}.jsonl \
        --out   runs/gsm8k/trajectories/part${i}.jsonl \
        $GEN_FLAGS \
        > runs/gsm8k/logs/gen_part${i}.log 2>&1 &
    echo "  part${i} → GPU${gpu} (PID $!)"
done
wait
log "Step 1 done"
wc -l runs/gsm8k/trajectories/part*.jsonl

# ── Step 2: Merge, filter correct, select per-problem best/random ─────
log "Step 2: filter + select"

cat runs/gsm8k/trajectories/part*.jsonl \
    > runs/gsm8k/trajectories/full.jsonl
echo "  merged: $(wc -l < runs/gsm8k/trajectories/full.jsonl) trajectories"

python -m pipelines.filter_correct \
    --inp  runs/gsm8k/trajectories/full.jsonl \
    --out  runs/gsm8k/trajectories/correct.jsonl \
    --print-stats

# per-problem best ELBO (content_logprob_mean) → proper_constrained + sft_control
python -m pipelines.select_best_trajectory \
    --inp  runs/gsm8k/trajectories/correct.jsonl \
    --out  runs/gsm8k/trajectories/best_elbo.jsonl \
    --group-by prompt_text

# per-problem random → random_constrained + sft_random
python -m pipelines.select_best_trajectory \
    --inp  runs/gsm8k/trajectories/correct.jsonl \
    --out  runs/gsm8k/trajectories/random_order.jsonl \
    --group-by prompt_text --random-select

echo "  best_elbo:    $(wc -l < runs/gsm8k/trajectories/best_elbo.jsonl) problems"
echo "  random_order: $(wc -l < runs/gsm8k/trajectories/random_order.jsonl) problems"
log "Step 2 done"

# ── Step 3: Prepare training pairs (4 conditions) ─────────────────────
log "Step 3: prepare pairs"

# proper_constrained: best-ELBO trajectory + trajectory mask (Section 3.2)
python -m pipelines.prepare_constrained_pairs \
    --inp runs/gsm8k/trajectories/best_elbo.jsonl \
    --out runs/gsm8k/pairs/proper_constrained.jsonl \
    --samples-per-trajectory 10

# sft_control: best-ELBO trajectory + random mask (ablates order signal)
python -m pipelines.prepare_sft_control_pairs \
    --inp runs/gsm8k/trajectories/best_elbo.jsonl \
    --out runs/gsm8k/pairs/sft_control.jsonl \
    --samples-per-trajectory 10

# random_constrained: random trajectory + trajectory mask (ablates ELBO selection)
python -m pipelines.prepare_constrained_pairs \
    --inp runs/gsm8k/trajectories/random_order.jsonl \
    --out runs/gsm8k/pairs/random_constrained.jsonl \
    --samples-per-trajectory 10

# sft_random: random trajectory + random mask (negative control)
python -m pipelines.prepare_sft_control_pairs \
    --inp runs/gsm8k/trajectories/random_order.jsonl \
    --out runs/gsm8k/pairs/sft_random.jsonl \
    --samples-per-trajectory 10

for f in proper_constrained sft_control random_constrained sft_random; do
    echo "  ${f}: $(wc -l < runs/gsm8k/pairs/${f}.jsonl) pairs"
done
log "Step 3 done"

# ── Step 4: Training (4 models in parallel, one per GPU) ──────────────
log "Step 4: train"
TRAIN_FLAGS="--model GSAI-ML/LLaDA-8B-Instruct \
             --lora-rank 64 --lora-target-modules q_proj,k_proj,v_proj,o_proj \
             --edit-frac 0.15 --epochs 3 --lr 2e-5 \
             --batch-size 4 --grad-accum 8 --mixed-precision bf16"

CUDA_VISIBLE_DEVICES=0 python -m pipelines.constrained_train \
    --train-jsonl runs/gsm8k/pairs/proper_constrained.jsonl \
    --output-dir  runs/checkpoints/gsm8k_proper_constrained \
    $TRAIN_FLAGS > runs/checkpoints/gsm8k_proper_constrained.log 2>&1 &
echo "  proper_constrained   → GPU0 (PID $!)"

CUDA_VISIBLE_DEVICES=1 python -m pipelines.constrained_train \
    --train-jsonl runs/gsm8k/pairs/sft_control.jsonl \
    --output-dir  runs/checkpoints/gsm8k_sft_control \
    $TRAIN_FLAGS > runs/checkpoints/gsm8k_sft_control.log 2>&1 &
echo "  sft_control          → GPU1 (PID $!)"

CUDA_VISIBLE_DEVICES=2 python -m pipelines.constrained_train \
    --train-jsonl runs/gsm8k/pairs/random_constrained.jsonl \
    --output-dir  runs/checkpoints/gsm8k_random_constrained \
    $TRAIN_FLAGS > runs/checkpoints/gsm8k_random_constrained.log 2>&1 &
echo "  random_constrained   → GPU2 (PID $!)"

CUDA_VISIBLE_DEVICES=3 python -m pipelines.constrained_train \
    --train-jsonl runs/gsm8k/pairs/sft_random.jsonl \
    --output-dir  runs/checkpoints/gsm8k_sft_random \
    $TRAIN_FLAGS > runs/checkpoints/gsm8k_sft_random.log 2>&1 &
echo "  sft_random           → GPU3 (PID $!)"

wait
log "Step 4 done"

# ── Step 5: Evaluation (5 conditions × 4 block sizes = 20 jobs) ───────
log "Step 5: evaluate"
EVAL_FLAGS="--steps 128 --gen-len 256 --temperature 0.0 \
            --topk-save 1 --use-chat-template"
TEST="data/prepared/gsm8k_test_500.jsonl"
PC="runs/checkpoints/gsm8k_proper_constrained/final"
SC="runs/checkpoints/gsm8k_sft_control/final"
RC="runs/checkpoints/gsm8k_random_constrained/final"
SR="runs/checkpoints/gsm8k_sft_random/final"

eval_job() {
    local gpu=$1 name=$2 block=$3 lora=$4
    local out="runs/gsm8k_eval/${name}_block${block}.jsonl"
    local log_f="runs/gsm8k_eval/${name}_block${block}.log"
    local lora_flag=""
    [ -n "$lora" ] && lora_flag="--lora-path $lora"
    echo "  [eval] GPU${gpu} ${name} block${block}"
    CUDA_VISIBLE_DEVICES=$gpu python -m pipelines.generate \
        --jsonl "$TEST" --out "$out" $EVAL_FLAGS \
        --block-len "$block" $lora_flag \
        > "$log_f" 2>&1 &
}

# Batch 1: block256 × 4
eval_job 0 baseline             256 ""
eval_job 1 proper_constrained   256 "$PC"
eval_job 2 sft_control          256 "$SC"
eval_job 3 random_constrained   256 "$RC"
wait; log "Eval batch 1 done (block256×4)"

# Batch 2: block256×1 + block128×3
eval_job 0 sft_random           256 "$SR"
eval_job 1 baseline             128 ""
eval_job 2 proper_constrained   128 "$PC"
eval_job 3 sft_control          128 "$SC"
wait; log "Eval batch 2 done"

# Batch 3: block128×2 + block64×2
eval_job 0 random_constrained   128 "$RC"
eval_job 1 sft_random           128 "$SR"
eval_job 2 baseline              64 ""
eval_job 3 proper_constrained    64 "$PC"
wait; log "Eval batch 3 done"

# Batch 4: block64×3 + block32×1
eval_job 0 sft_control           64 "$SC"
eval_job 1 random_constrained    64 "$RC"
eval_job 2 sft_random            64 "$SR"
eval_job 3 baseline              32 ""
wait; log "Eval batch 4 done"

# Batch 5: block32×4
eval_job 0 proper_constrained   32 "$PC"
eval_job 1 sft_control          32 "$SC"
eval_job 2 random_constrained   32 "$RC"
eval_job 3 sft_random           32 "$SR"
wait; log "Eval batch 5 done"

log "Step 5 done"

# ── Step 6: Summarize results ─────────────────────────────────────────
log "Step 6: summarize"
python3 -c "
import json
from pathlib import Path

conditions = ['baseline', 'proper_constrained', 'sft_control', 'random_constrained', 'sft_random']
blocks = [32, 64, 128, 256]

print()
print(f'{'Condition':<25}' + ''.join(f'  block{b:>3}' for b in blocks) + '  drop(32→256)')
print('-'*82)
for cond in conditions:
    accs = []
    for b in blocks:
        p = Path(f'runs/gsm8k_eval/{cond}_block{b}.jsonl')
        correct = total = 0
        if p.exists():
            for line in p.open():
                d = json.loads(line)
                total += 1
                if d.get('summary', {}).get('correct'): correct += 1
        accs.append(correct/total*100 if total else float('nan'))
    drop = accs[0] - accs[-1]
    print(f'{cond:<25}' + ''.join(f'  {a:>6.1f}%' for a in accs) + f'  {drop:>+6.1f}%')
print()
print('2x2 ablation:')
print('  proper_constrained vs sft_control    -> trajectory mask contribution (same data)')
print('  proper_constrained vs random_constrained -> ELBO selection contribution (same mask)')
print('  sft_control vs sft_random            -> data quality contribution (same mask)')
"

log "Experiment complete"
