# Softmax

> This is the deep-dive note on the softmax layer. For its entry in the layer reference (what it does and where it goes), see [[List of Layers#Softmax]].

The softmax function converts a vector of real numbers ([[Logits|logits]]) into a probability distribution. Given a vector $z = [z_1, z_2, ..., z_K]$, the softmax is:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

## How it works

1. **Exponentiate**: Apply $e^x$ to each logit, mapping negatives to small positives and positives to large positives.
2. **Normalize**: Divide by the sum of all exponentiated values.

This guarantees:
- Every output is in $[0, 1]$
- All outputs sum to exactly 1
- The ordering is preserved (bigger logit → bigger probability)

## Why "soft" max?

The softmax is a **differentiable** approximation of the hard argmax. The hard argmax sets the max entry to 1 and all others to 0 — non-differentiable and useless for backpropagation. Softmax gives a "soft" version where:

- The largest logit gets the highest probability (but not 1)
- Smaller logits still get some probability mass
- The more extreme the gap between logits, the closer softmax gets to a one-hot vector ("sharper" distribution)

The "temperature" of the softmax controls this sharpness:

$$\text{softmax}\left(\frac{z_i}{T}\right) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}$$

- **Low $T$** (e.g., 0.1): sharper, more like argmax. Used when you want confident predictions.
- **High $T$** (e.g., 5): flatter, more uniform. Used when you want more variety/diversity in sampling.

## In PyTorch

```python
import torch.nn.functional as F

probs = F.softmax(logits, dim=1)
```

Or manually, which makes the two steps explicit:

```python
counts = logits.exp()           # exponentiate
probs = counts / counts.sum(dim=1, keepdim=True)  # normalize
```

## Numerical stability trick

Directly computing $e^{z_i}$ can overflow for large $z_i$. The standard fix is to subtract the max before exponentiating:

$$\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

This doesn't change the result (it's algebraically equivalent) but keeps values safe from overflow. PyTorch's `F.softmax` does this internally.