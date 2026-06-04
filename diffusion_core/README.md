# Diffusion Core Module

A standalone, general-purpose implementation of a Masked Diffusion Language Model (LLaDA/MDM style).

## Features
- **Generic Masking**: Support for global sequence masking (Pretrain) and selective masking (SFT/Denoising).
- **Bidirectional Transformer**: A self-contained Transformer implementation that supports full-context attention.
- **Iterative Denoising Inference**: Implements block-wise parallel decoding with remasking.
- **Data Agnostic**: Works with any tokenized text data.

## Directory Structure
- `model.py`: The `DiffusionTransformer` architecture.
- `masking.py`: Logic for applying Bernoulli noise based on time-step $t$.
- `loss.py`: Masked cross-entropy with $1/p$ importance sampling.
- `data.py`: `DiffusionCollator` for handling batching and masking in DataLoader.
- `inference.py`: `DiffusionSampler` for generating text through iterative denoising.

## Quick Start
See `example_usage.py` for a complete end-to-end example of training on raw strings and performing inference.

```python
from diffusion_core.model import DiffusionTransformer
from diffusion_core.inference import DiffusionSampler

# Initialize model
model = DiffusionTransformer(...)

# Generate
sampler = DiffusionSampler(model, tokenizer, mask_token_id, device)
output_ids = sampler.generate(prompt_ids, max_new_tokens=10)
```
