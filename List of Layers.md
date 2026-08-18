# List of Layers

The layer reference of this vault. Each entry covers **what the layer does**, **where it is usually placed** in a network, where it appears in `makemore_part3_bn.ipynb` (the [[Basic MLP Overview|Basic MLP Overview]] note), and an **example implementation** in the notebook's module style — a class with `__call__` and a `parameters()` method (assumes `torch` is imported).

## Embedding

![[Layer - Embedding.png]]

**What it does:** A lookup table that maps discrete symbols (characters, words, tokens) to dense, learned vectors. `C` is a matrix of shape `[vocab_size, n_embd]`; indexing it with a symbol's integer index *plucks out that symbol's row*: `emb = C[X]`. It is mathematically identical to multiplying a [[One-hot encoding|one-hot]] vector by a matrix (`one_hot(x) @ C == C[x]`), so it is a linear layer with no activation — written in index form for convenience. The vectors are trained by backprop, so similar symbols end up with similar vectors: a *distributed representation*.

**Where it is usually placed:** Always at the very **input** of the network — the first layer. It converts token indices into the vectors the rest of the network consumes. If the context holds several tokens, the plucked vectors are concatenated (flattened) before the next layer.

**Example implementation:**

```python
class Embedding:
    def __init__(self, vocab_size, n_embd):
        self.C = torch.randn((vocab_size, n_embd))   # the lookup table

    def __call__(self, x):
        self.out = self.C[x]   # x: [N, block_size] indices -> [N, block_size, n_embd]
        return self.out

    def parameters(self):
        return [self.C]
```

*With `block_size > 1` the output is 3D — flatten it (`.view(N, -1)`) before the next [[List of Layers#Linear|Linear]].*

## Flatten

**What it does:** Reshapes its input into 2D rows — `x.view(x.shape[0], -1)` collapses every dimension after the batch into one, gluing the per-token vectors end to end. No parameters, no math: it only rearranges memory so the next layer sees one long vector per example.

**Where it is usually placed:** Right after the [[List of Layers#Embedding|Embedding]] when the context holds several tokens: the plucked vectors go from `[N, block_size, n_embd]` to `[N, block_size * n_embd]` before the first [[List of Layers#Linear|Linear]]. It is *usually* and *only* used with [[List of Layers#Linear|Linear]]. 

**Example implementation:**

```python
class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)   # [N, ...] -> [N, product of the rest]
        return self.out

    def parameters(self):
        return []
```

## Linear

![[Layer - Linear.png]]

**What it does:** A fully connected (dense) layer: `y = x @ W + b`. Each neuron computes a weighted sum of *all* its inputs — a [[Dot product]] of one input row against one weight column — plus a bias. Every output neuron depends on every input, hence "fully connected". The weights are the layer's parameters, learned by backprop. It is the "workhorse" transformation of an MLP.

**Where it is usually placed:** Between every pair of feature vectors in a network — it is what fills *hidden layers* and the *output layer*. The standard modern motif is **Linear → normalization → nonlinearity**. When a Linear is followed by [[List of Layers#BatchNorm1d|BatchNorm]], its bias is dropped (`bias=False`): the normalization subtracts the bias right back out, so it would never learn (its gradient is exactly zero).

**Example implementation:**

```python
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
```

*When followed by [[List of Layers#BatchNorm1d|BatchNorm]], call it with `bias=False`. As in the notebook, the tensors are created without `requires_grad` — set it on all collected parameters before training: `for p in parameters: p.requires_grad = True`.*

## BatchNorm1d

![[Layer - BatchNorm1d.png]]

**What it does:** Normalizes a layer's pre-activations **per neuron, over the batch**: subtract the batch mean, divide by the batch standard deviation (plus a tiny `eps`), then scale and shift with two *learnable* parameters, `gamma` (gain) and `beta` (bias). During training it uses the current batch's statistics; at test time it uses **running statistics**, accumulated with an exponential moving average. The whole operation is differentiable, so it slots into backprop like any layer.

**Where it is usually placed:** Between the linear layer and the nonlinearity — `Linear → BatchNorm1d → Tanh`. That position is what keeps the nonlinearity out of its saturated/zero regions. It decouples "what the statistics are" (fixed by normalization) from "where the distribution sits" (learned by gamma/beta). Side effects: examples in a batch become coupled (acts as a mild regularizer), and the bias of the preceding Linear is useless.

**Example implementation:**

```python
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):  # notebook default; the hand-rolled version used 0.001
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # learned parameters (backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # running statistics (EMA), used when training=False
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True)   # per-neuron stats over the batch
            xvar = x.var(0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        if self.training:   # update the running stats
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
```

## Tanh

![[Layer - Tanh.png]]

**What it does:** A nonlinearity (activation function) that squashes any real number into the open interval `(−1, 1)`:

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Its local gradient is $1 - \tanh^2(x)$, which is **always ≤ 1** — tanh is *contractive*. For inputs near 0 the gradient is ~1 (healthy); for inputs far from 0 (`|x| > ~2`) it saturates at ±1, where the gradient ≈ 0. A neuron stuck there gets ~zero gradient forever: a **dead neuron**. Stacked through many layers, tanh also keeps *shrinking* the activation distribution unless later layers push back (hence the `5/3` gain).

**Where it is usually placed:** After the linear/normalization block, before the next layer. It provides the **nonlinearity** that lets stacked layers compose into something more expressive than a single linear map. Alternatives: [[List of Layers#Sigmoid|Sigmoid]], [[List of Layers#ReLU|ReLU]], etc.

**Example implementation:**

```python
class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []
```

## Softmax

![[Layer - Softmax.png]]

**What it does:** Turns a vector of [[Logits|logits]] into a probability distribution:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Exponentiate, then normalize so every entry is in `[0, 1]` and the vector sums to 1. It preserves ordering (bigger logit → bigger probability) and is a differentiable approximation of argmax. The deep-dive note is [[Softmax]].

**Where it is usually placed:** At the **very end** of a classification network, after the last [[List of Layers#Linear|Linear]] layer. In practice it is *fused into the loss* — `F.cross_entropy(logits, y)` computes log-softmax + negative log-likelihood in one numerically stable kernel — so the network outputs logits and the softmax only becomes explicit when sampling.

**Example implementation:**

```python
class Softmax:
    def __call__(self, x):
        # numerically stable: subtract the row max before exponentiating
        xmax = x.max(dim=1, keepdim=True).values
        exps = (x - xmax).exp()
        self.out = exps / exps.sum(dim=1, keepdim=True)
        return self.out

    def parameters(self):
        return []
```

## ReLU

![[Layer - ReLU.png]]

**What it does:** $\text{ReLU}(x) = \max(0, x)$. Trivially cheap: gradient is 1 for positive inputs, **exactly 0** below zero. A neuron whose pre-activations are always negative is permanently dead — Karpathy calls it "permanent brain damage". Variants that fix the flat region: **Leaky ReLU** (small slope below zero), **ELU** (smoothly curves below zero).

**Where it is usually placed:** After a linear/conv layer *instead of* [[List of Layers#Tanh|tanh]] — the modern default in deep networks, classically as `Conv → BatchNorm → ReLU`. Kaiming init uses gain $\sqrt{2}$ for it.

**Example implementation:**

```python
class ReLU:
    def __call__(self, x):
        self.out = torch.relu(x)   # max(0, x)
        return self.out

    def parameters(self):
        return []
```

## Sigmoid

![[Layer - Sigmoid.png]]

**What it does:** $\sigma(x) = \frac{1}{1 + e^{-x}}$ — squashes into `(0, 1)`, so it is interpretable as a probability for *binary* classification. Same saturation problem as [[List of Layers#Tanh|Tanh]]: far from 0 the gradient dies.

**Where it is usually placed:** As the final layer of a binary classifier (the lone probability). Historically used as a hidden activation; largely replaced by [[List of Layers#Tanh|tanh]]/[[List of Layers#ReLU|ReLU]] there because of gradient-killing tails.

**Example implementation:**

```python
class Sigmoid:
    def __call__(self, x):
        self.out = torch.sigmoid(x)   # 1 / (1 + exp(-x))
        return self.out

    def parameters(self):
        return []
```

## Dropout

![[Layer - Dropout.png]]

**What it does:** During training, randomly zeroes a fraction (e.g. 50%) of a layer's activations, forcing the network not to rely on any single neuron — a **regularizer** that fights overfitting. At test time no neurons are dropped; the layer's outputs are scaled to match the training-time expectation.

**Where it is usually placed:** After the nonlinearity, typically on the large hidden layers near the end of the network (e.g. before the final [[List of Layers#Linear|Linear]]). Not used in `makemore_part3_bn.ipynb`.

**Example implementation** (inverted dropout, the PyTorch convention):

```python
class Dropout:
    def __init__(self, p=0.5):
        self.p = p            # probability of dropping a neuron
        self.training = True

    def __call__(self, x):
        if self.training:
            keep = torch.bernoulli(torch.full_like(x, 1 - self.p))
            self.out = (x * keep) / (1 - self.p)   # scale up so the mean is unchanged
        else:
            self.out = x      # nothing dropped at test time
        return self.out

    def parameters(self):
        return []
```

## LayerNorm

![[Layer - LayerNorm.png]]

**What it does:** The same normalize-then-scale idea as [[List of Layers#BatchNorm1d|BatchNorm]], but the statistics are computed **per example, over the feature dimension** — not over the batch. No batch coupling, no running statistics needed: training and test behavior are identical.

**Where it is usually placed:** In the same slot as [[List of Layers#BatchNorm1d|BatchNorm]] (after the [[List of Layers#Linear|linear]] layer, before the nonlinearity). Preferred when batches are tiny or 1 — the standard choice in RNNs and Transformers. Not used in `makemore_part3_bn.ipynb`.

**Example implementation:**

```python
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)   # per example, over the feature dim
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
```

*No `training` flag and no running stats — train and test behavior are identical.*

## Recursive Attention

![[Layer - MultiHeadAttention.png]]

**What it does:** The standard name for this layer is **multi-head self-attention**. Every token produces a **query** (`Q`), a **key** (`K`), and a **value** (`V`) from the same input sequence. A query compares itself with every key using a dot product; the resulting scores decide how much of each value to mix into that token's output. The scores are scaled by the square root of the head width so their variance does not grow with the number of features:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

A single attention head performs this calculation in one feature subspace. **Multi-head** attention splits the embedding into `n_heads` smaller subspaces, runs attention in each one in parallel, concatenates the results, and projects them back to the original embedding width. Different heads can learn different relationships — for example, nearby context, syntax, or a long-range reference.

The input and output both have shape `[B, T, C]`: batch size, sequence length, and embedding width. Inside the layer, each head has width `d_k = C / n_heads`, so `Q`, `K`, and `V` are reshaped to `[B, n_heads, T, d_k]` before the score matrix `[B, n_heads, T, T]` is computed.

![[Layer - CausalMask.png]]

**Masking:** In autoregressive/decoder self-attention, a token must not use information from tokens that come later in the sequence. A **causal mask** therefore keeps the diagonal and lower triangle of the score matrix and replaces the upper triangle with `-∞` *before* softmax. Those entries become probability `0`, while every row still sums to `1`. The mask is not needed for the usual bidirectional self-attention in an encoder.

**Where it is usually placed:** Inside every Transformer block. The encoder uses unmasked multi-head self-attention; the decoder uses causal masked self-attention so it can generate one token at a time without seeing the target tokens that come later. Decoder cross-attention is a related operation, but it is outside this entry.

**Educational implementation:** This version follows the hand-written module style used throughout this note. `Linear` refers to the [[List of Layers#Linear|Linear]] layer defined above. The fixed triangular mask is not a learnable parameter.

```python
class MultiHeadAttention:
    def __init__(self, n_embd, n_heads, block_size):
        if n_embd % n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.block_size = block_size

        # Separate projections let every head learn its own Q/K/V features.
        self.q_proj = Linear(n_embd, n_embd, bias=False)
        self.k_proj = Linear(n_embd, n_embd, bias=False)
        self.v_proj = Linear(n_embd, n_embd, bias=False)
        self.out_proj = Linear(n_embd, n_embd, bias=False)

        # True means "this position may attend to that position".
        self.causal_mask = torch.tril(
            torch.ones((block_size, block_size), dtype=torch.bool)
        )

    def __call__(self, x):
        B, T, C = x.shape                         # [B, T, C]
        if C != self.n_embd:
            raise ValueError(f"expected embedding width {self.n_embd}, got {C}")
        if T > self.block_size:
            raise ValueError("sequence is longer than block_size")

        def split_heads(x):
            # [B, T, C] -> [B, n_heads, T, head_dim]
            return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        scores = q @ k.transpose(-2, -1) / self.head_dim**0.5
        allowed = self.causal_mask[:T, :T]
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)       # [B, h, T, T]

        y = weights @ v                               # [B, h, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        self.attention_weights = weights              # useful for inspection
        self.out = self.out_proj(y)
        return self.out

    def parameters(self):
        return (
            self.q_proj.parameters()
            + self.k_proj.parameters()
            + self.v_proj.parameters()
            + self.out_proj.parameters()
        )
```

**Production PyTorch implementation:** `nn.MultiheadAttention` performs the projections, head splitting, scaled dot products, and head merging internally. In a boolean `attn_mask`, `True` entries are blocked.

```python
import torch
from torch import nn


class ProductionMultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,              # x is [B, T, C]
        )

    def forward(self, x):
        T = x.size(1)
        # True above the diagonal blocks future keys/values.
        causal_mask = torch.triu(
            torch.ones((T, T), device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        out, _ = self.attention(
            x, x, x,                         # self-attention: Q = K = V = x
            attn_mask=causal_mask,
            need_weights=False,
        )
        return out
```

Some PyTorch versions also expose an `is_causal=True` fast path. The explicit boolean mask above makes the masking rule visible and works across versions that support `nn.MultiheadAttention`.
