# Discrete Diffusion Language Model (Masked Diffusion LM) Technical Guide

This guide provides a comprehensive reference for the Discrete Diffusion Language Model (MDM) implementation. It covers the mathematical foundations, engineering details, and tuning strategies required to implement, optimize, and adapt this architecture for your own experiments.

---

## 1. Mathematical Framework: Diffusion in Discrete Space

Unlike Gaussian Diffusion in continuous domains (e.g., images), Language Models operate on discrete tokens. We utilize **Masking** as the source of noise.

### 1.1 Forward Process
The forward process $q(x_t | x_0)$ is defined over a continuous time-step $t \in (0, 1]$. At each step, each token in the clean sequence $x_0$ has a probability $p_t$ of being replaced by a special `[MASK]` token.

**Noise Schedule**:
We use a linear schedule:
$$p_t = (1 - \epsilon)t + \epsilon$$
Where $\epsilon$ (e.g., $1e-3$) is a small constant ensuring minimal noise even at $t \to 0$, improving robustness.

**Mathematical Property**:
This is an independent Bernoulli trial per token. For a sequence of length $L$, the number of masked tokens follows a Binomial distribution $B(L, p_t)$.

### 1.2 Training Objective
The model $f_\theta(x_t, t)$ aims to predict the original tokens at the masked positions based on the noisy observation $x_t$.

**Loss Function and Importance Sampling**:
In standard MLM (like BERT), the loss is the average cross-entropy over masked positions. In diffusion models, to simulate a continuous denoising process from high to low noise, we must weight the loss generated at different $t$.

We minimize the following expected loss:
$$\mathcal{L} = \mathbb{E}_{t \sim U(0,1), x_t \sim q(x_t|x_0)} \left[ \frac{1}{p_t} \sum_{i \in \text{masked}} \text{CrossEntropy}(f_\theta(x_t, i), x_{0,i}) \right]$$

**Why multiply by $1/p_t$?**
When $t$ is small, $p_t$ is small, and only a few tokens are masked. The prediction task is trivial. Without weighting, the model would overfit to easy tasks. The $1/p_t$ weight forces the model to gain sufficient gradient signal even when noise is extreme ($p_t \approx 1$), encouraging it to learn global structure.

---

## 2. Engineering Implementation

### 2.1 Model Architecture
*   **Bidirectional Attention**: The causal mask used in Autoregressive models (GPT) must be removed. Every token prediction must depend on the full sentence context.
*   **Positional Embeddings**: Since generation is no longer strictly left-to-right, Absolute Positional Embeddings or RoPE are crucial for the model to determine token order.

### 2.2 Training Nuances
*   **Dynamic Masking**: Masking is applied on-the-fly in the `DiffusionCollator` for every batch based on a randomly sampled $t$.
*   **Normalization**: When calculating batch loss, we sum the weighted losses per sample and divide by the total number of masked tokens in that batch.

### 2.3 Inference: Iterative Remasking
This is the "soul" of diffusion generation. Instead of a single-pass sampling, we refine the result through multiple forward passes.

**Algorithm (Block-wise Denoising)**:
1.  **Initialize**: Input Prompt + $N$ `[MASK]` tokens.
2.  **Predict**: Model outputs probability distributions for all $N$ positions.
3.  **Sample**: Pick $N$ tokens based on Temperature ($T$) and Top-k.
4.  **Confidence**: Record the logit probability for each sampled token.
5.  **Remask**:
    *   Calculate the retention ratio $r = 1 - \frac{\text{step}}{\text{total\_steps}}$.
    *   **Strategy**: Re-mask tokens with the lowest confidence scores (Strategy B).
6.  **Iterate**: Repeat from Step 2 until steps are exhausted.

---

## 3. Tuning Guide

### 3.1 Training Hyperparameters
*   **`eps` ($\epsilon$)**: Recommended $1e-3$. Too large prevents precise denoising; too small can lead to numerical instability ($1/p_t$ exploding).
*   **$t$ Distribution**: Default is Uniform. You can use a `beta` distribution skewed toward larger $t$ if you want the model to prioritize global structure recovery.

### 3.2 Inference Hyperparameters
*   **`block_size`**: Number of tokens generated at once. Large values (>64) may weaken bidirectional constraints; 8~16 is recommended.
*   **`steps_per_block`**: Iterations per block. 5~10 is sufficient for simple tasks; 20+ for complex logical reasoning.
*   **`remask_mode`**: `low_confidence` is highly recommended as it enables "self-correction".

---

## 4. Portability

1.  **Data Agnostic**: Simply tokenize your text and pass it to the provided `SimpleTextDataset`.
2.  **Model Swapping**: Replace `DiffusionTransformer` with any bidirectional architecture (BERT, RoBERTa, etc.), provided it accepts `input_ids` and `attention_mask`.
3.  **SFT Adaptation**: In SFT mode, modify `valid_mask` in the Collator to only calculate loss on the Response part, while keeping the Prompt visible but unmasked.

This implementation provides a clean baseline. If generation logic is inconsistent, first verify **`steps_per_block`** and the **`1/p` loss scaling** implementation.
