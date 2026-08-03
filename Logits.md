# Logits

Logits are the **raw, unnormalized output scores** of a neural network before any activation function like softmax is applied. They are called "logits" because they live in **log-odds space** — the inverse of the sigmoid/softmax.

## Key properties

- **Range**: Logits can be any real number, from $-\infty$ to $+\infty$.
- **Interpretation**: A higher logit means the model "prefers" that class more, but logits are not directly comparable across classes without normalization.
- **Relationship to probability**: For a binary classification, applying the sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$ converts a logit to a probability. For multi-class, softmax is used instead.

## Logits vs probabilities

| Logits | Probabilities |
|--------|---------------|
| Range: $(-\infty, \infty)$ | Range: $[0, 1]$ |
| Not interpretable as-is | Directly interpretable |
| Used internally by the network | Used for prediction and loss |
| Can be added/subtracted freely | Must stay in $[0, 1]$ and sum to 1 |

## Why networks output logits

1. **Numerical stability**: Working in log-space avoids underflow when probabilities get very small. Multiplying many probabilities (as in likelihood) becomes adding logits.
2. **Unconstrained optimization**: The network can freely output any real number. If it had to output probabilities directly, each output would need to be clamped to $[0, 1]$ and sum to 1 — an awkward constraint.
3. **Gradient flow**: The softmax + cross-entropy combination has clean gradients: $\frac{\partial L}{\partial z_i} = p_i - y_i$ (the gradient is simply `probs - targets`).