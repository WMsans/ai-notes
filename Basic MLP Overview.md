# Basic MLP Overview

This note walks through `makemore_part3_bn.ipynb` — the third video in Karpathy's *makemore* series. It takes the character-level [[Building of MLP|MLP]] (predict the next character from 3 characters of context), fixes two initialization bugs, then stacks the whole thing **deeper** with [[List of Layers#BatchNorm1d|BatchNorm]] doing the heavy lifting. The layer reference for everything mentioned here is [[List of Layers]].

> [!NOTE] TL;DR
> The notebook is a story in four acts: (1) the plain MLP from before, (2) *why* it starts badly — logits too big and [[List of Layers#Tanh|tanh]] saturated, (3) the principled fix — **Kaiming init**, then **BatchNorm**, (4) a deeper 6-layer network that trains cleanly with almost no hand-tuning.

## The model at a glance

![[Basic MLP architecture.png]]

- **Input:** 3 characters (block_size = 3), each mapped to a 10-dim vector by the [[List of Layers#Embedding|Embedding]] table `C` (`27 × 10`), then concatenated into a 30-dim vector.
- **Body:** five hidden blocks of `[[List of Layers#Linear|Linear]] → [[List of Layers#BatchNorm1d|BatchNorm1d]] → [[List of Layers#Tanh|Tanh]]` (100 neurons each).
- **Head:** a final [[List of Layers#Linear|Linear]] to 27 logits, one [[List of Layers#BatchNorm1d|BatchNorm1d]], then [[List of Layers#Softmax|Softmax]] into probabilities.
- **Total:** 47,024 parameters. Loss: `F.cross_entropy(logits, y)`.

## Data and hyperparameters

- 32,033 names, split 80/10/10 into train/dev/test (the dev set is used for validation).
- `n_embd = 10`, `n_hidden = 200` for the small network (100 for the deep one), `block_size = 3`, `batch_size = 32`, `max_steps = 200,000`, learning rate `0.1` decaying to `0.01` at step 100,000.

## 1. The plain MLP, revisited

The network is the same as in [[Building of MLP]]:

```
Embedding → Linear(W1) → Tanh → Linear(W2) → Softmax
```

- `emb = C[X]` plucks a vector per context character; `emb.view(-1)` concatenates them. ([[List of Layers#Embedding]])
- `hpreact = embcat @ W1` — the hidden [[List of Layers#Linear|Linear]] layer.
- `h = torch.tanh(hpreact)` — the [[List of Layers#Tanh|Tanh]] nonlinearity.
- `logits = h @ W2 + b2` — the output [[List of Layers#Linear|Linear]] layer.
- `loss = F.cross_entropy(logits, y)` — log-softmax + negative log-likelihood fused ([[List of Layers#Softmax]], [[Softmax]]).

## 2. Problem 1: confidently wrong at initialization

With random init the loss starts at **27** and only then crashes down — a *hockey stick* curve. Expected: `ln(27) ≈ 3.29` (a uniform distribution over 27 characters, see [[Likelihood]]).

**Why:** the logits take extreme values, so [[List of Layers#Softmax|Softmax]] puts almost all probability on the *wrong* character.

**Fix:** shrink the last layer so the logits are ≈ 0 at init:

```python
W2 = torch.randn((n_hidden, vocab_size)) * 0.01
b2 = torch.randn(vocab_size) * 0
```

The loss now starts at ≈ 3.3. (This is the pre-[[List of Layers#BatchNorm1d|BatchNorm]] part of the notebook — the curve below is the full pipeline.)

## 3. Problem 2: tanh saturation and Kaiming init

The hidden pre-activations `hpreact` come out too broad (roughly −5 to 15), so [[List of Layers#Tanh|Tanh]] squashes almost everything into its flat tails. There the gradient is `1 − tanh² ≈ 0` — gradients die, and neurons stuck at ±1 never learn (**dead neurons**). Full story in [[Activations & Gradients, BatchNorm in MLP]].

**Fix:** analyze how `y = x @ W` spreads the distribution — std grows by `√fan_in` — and divide by it:

```python
W1 = torch.randn((n_embd * block_size, n_hidden)) * (5/3) / (n_embd * block_size)**0.5
```

That's **Kaiming/He initialization** (gain `5/3` for [[List of Layers#Tanh|tanh]]). Now `h` is healthy at init.

## 4. BatchNorm — the main event

Hand-tuning weight scales works for one layer, but errors *stack* in deep networks. The insight of *Ioffe & Szegedy 2015*: **why not just normalize the pre-activations?** Standardizing is differentiable, so it becomes a layer.

![[Basic MLP BatchNorm effect.png]]

The [[List of Layers#BatchNorm1d|BatchNorm1d]] layer, inserted between the [[List of Layers#Linear|Linear]] and the [[List of Layers#Tanh|Tanh]] (`Linear → BN → Tanh`):

1. **Batch statistics:** `mean` and `var` computed per neuron over the batch.
2. **Standardize:** `xhat = (x − mean) / sqrt(var + eps)` — forces the pre-activations to be unit Gaussian, *regardless of init*.
3. **Learnable scale & shift:** `gamma * xhat + beta` (initialized to identity), so the network can still decide how spread/offset each neuron is.

Key details from the notebook:

- **Running statistics for test time.** A batch needs many examples, but sampling uses one. So during training the notebook keeps an EMA of the batch stats (`0.999 * running + 0.001 * batch`), frozen at inference. Momentum `0.001` (small batch → noisy stats).
- **`bias=False` on the preceding [[List of Layers#Linear|Linear]].** Whatever the bias adds, the normalization subtracts right back out — its gradient is exactly zero, so it's dead weight. BatchNorm brings its own bias (`beta`).
- **Last layer:** `layers[-1].gamma *= 0.1` — keep the final logits unconfident (same spirit as the `W2 * 0.01` fix).

## 5. The deeper network

Karpathy wraps the layers into module classes (`Linear`, `BatchNorm1d`, `Tanh`) and stacks five hidden blocks plus the head (see the diagram at the top): 47,024 parameters, six [[List of Layers#Linear|Linear]] layers. With BatchNorm between every pair, the training is forgiving to init choices — no more magic numbers.

The notebook then inspects the network's health layer by layer:

1. **Forward histograms** — activations should stay spread (`std ≈ 0.65`, ~5% saturation), not collapse to 0 (gain too small) or saturate (gain too big).
2. **Backward histograms** — gradients should keep similar magnitude across layers.
3. **Update-to-data ratio** — `log10((lr * grad).std() / param.std())`, a good learning-rate gauge (~−3). The last [[List of Layers#Softmax|Softmax]] layer is the outlier (its gamma was shrunk).

## 6. Training and results

![[Basic MLP training loss.png]]

- 200,000 steps, batch 32, `lr 0.1 → 0.01`. The loss curve is noisy (log scale, per-batch) but trends down smoothly.
- Validation loss ends at **≈ 2.10** — better than the bigram baseline (`ln(27) ≈ 3.29`) and the plain MLP (~2.17–2.19).

## 7. Sampling

Same rolling-window loop as [[Building of MLP|the MLP sampler]], but the forward pass runs the whole `layers` list:

```python
x = emb.view(emb.shape[0], -1)
for layer in layers:
    x = layer(x)          # running stats are used now (training=False)
probs = F.softmax(x, dim=1)                 # explicit Softmax at sampling
ix = torch.multinomial(probs, num_samples=1).item()
```

[[List of Layers#Softmax|Softmax]] converts the logits to probabilities; `multinomial` samples *from* them (not argmax) so the generated names vary.

## Where this leads

- The deep-MLP failure modes and diagnostics are expanded in [[Activations & Gradients, BatchNorm in MLP]].
- The layer-by-layer reference: [[List of Layers]].
- Next in the series: [[Building of Wavenet]] — the same character model, rebuilt with convolutional layers.
