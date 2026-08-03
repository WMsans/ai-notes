In [[Building of MLP]], we used **activation function** to turn data into intepretable form (probabilities using cross entropy). We used **gradients** to tune the weights. We used **batches** to speed up the training process. 

These numbers are set casually. We are going to understand it in a complete sense.

# Setup: the same MLP, cleaned up

We keep the exact [[Building of MLP|MLP]] from before (11,000 parameters; 200,000 steps; batch size 32) but the code is refactored: the *magic numbers* (`10` embeddings, `200` hidden neurons, block size `3`) are pulled out into named variables, and we split the data into **train / dev / test** splits. We also use `@torch.no_grad()` when evaluating — this tells PyTorch that no backward pass will ever be called on this computation, so it skips building the graph and everything is much faster.

Trained as-is, the network reaches train & validation loss of about **2.16 – 2.17**. That's our baseline. Now let's scrutinize it.

# Problem 1: the network is confidently wrong at initialization

Look at the loss curve of the original network: it has a **hockey stick** shape — at iteration 0 the loss is a whopping **27**, and only then it rapidly falls. We should expect something much lower.

*Why?* At initialization the weights are random, so there's no reason to prefer any character over the others. We'd expect a **uniform distribution** over the 27 possible next characters, i.e. probability $\frac{1}{27}$ each, giving the expected [[Likelihood|negative log-likelihood]]:

$$-\ln\left(\tfrac{1}{27}\right) = \ln(27) \approx 3.29$$

But instead the [[Logits|logits]] coming out of the network take on **extreme values** — say `5` for a wrong character and something tiny for the right one. Softmax then puts nearly all probability on the wrong answer: the network is **confidently wrong**, so it records absurdly high loss.

>[!NOTE] Small example
> With 4 characters and logits all ≈ 0, softmax gives a uniform distribution and the loss is $\ln(4) \approx 1.38$ no matter the label. But if the logits are multiplied by 10, they take on extreme values, the model "guesses" the wrong bucket with confidence, and the loss explodes (possibly even to infinity).

## The fix: logits should be ≈ 0 at init

`logits = h @ W2 + b2`, so we make both contributions small:

```python
W2 = torch.randn((n_hidden, vocab_size)) * 0.01   # small, not zero!
b2 = torch.randn(vocab_size) * 0                  # exactly zero
```

- `b2 = 0` — we don't want a random offset.
- `W2` is scaled down but **not zero**: a small random matrix adds a little entropy and provides **symmetry breaking** (with exactly zero weights, every neuron would get identical gradients and the network could never differentiate them).
- The logits *don't strictly have to be zero* — softmax is invariant to adding a constant to all logits — but zero is the symmetric choice.

The initial loss is now ≈ 3.3 (what we expect), the hockey stick disappears, and — because training no longer wastes thousands of steps just squashing down the weights — the validation loss improves: **2.17 → 2.13**.

# Problem 2: tanh saturation — the "dead neuron" problem

The logits are fine now, but let's look at the *hidden layer*. Plotting the histogram of `h = tanh(hpreact)`:

![[Pasted image 20260719115605.png|476]]

Almost all values sit at **+1 and −1** — the tanh is *completely saturated*. The reason: the pre-activations `hpreact` are way too broad (roughly −5 to 15), so tanh squashes everything into its flat tails.

*Why is that bad?* Recall from [[Building of micrograd]] how the gradient flows through tanh — the local gradient is:

$$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x) = 1 - t^2$$

If $t \approx \pm 1$, this local gradient is ≈ 0, and by the chain rule the incoming gradient gets **multiplied by ~0**: the gradient is destroyed at that neuron. The further into the flat tail, the more the gradient shrinks. tanh can only ever *decrease* the gradient flowing through it.

We can visualize the damage directly: `(h.abs() > 0.99)` is a boolean tensor of shape `[32, 200]` (examples × neurons) — white where a neuron is in the flat tail. If an **entire column** (neuron) is white — i.e. no example ever lands in the active region of that tanh — the neuron's weights get zero gradient forever: it's a **dead neuron**, and it will never learn.

>[!WARNING] Dead neurons are not unique to tanh
> - **Sigmoid** — same squashing shape, same problem.
> - **ReLU** — has a completely flat region below zero, so a neuron whose pre-activations are always negative is *exactly* dead (gradient is exactly 0, not just tiny). Dead ReLUs can appear at initialization by chance, or during training when a too-large learning rate knocks a neuron off the data manifold — from then on nothing ever activates it again. Karpathy calls it *"permanent brain damage"*.
> - **Leaky ReLU / ELU** have no flat tails, so they're more forgiving.

## The fix: shrink the weights (but how much?)

`hpreact = embcat @ W1 + b1`, so we scale both down:

```python
W1 = torch.randn((n_embd * block_size, n_hidden)) * 0.2
b1 = torch.randn(n_hidden) * 0.01
```

Now the pre-activations are roughly within ±1.5, the histogram of `h` is nicely spread, no saturation. Validation loss: **2.13 → 2.10**.

*But this `0.2` is a magic number — I just guessed it from looking at a histogram.* For a 1-layer MLP the optimization is forgiving and the network learned anyway, but for a network with 50 layers, errors like this **stack up** until it barely trains at all. We need a principled way to set these scales.

# Kaiming initialization (He et al., 2015)

Let's analyze: if `x` has unit standard deviation and `W` has unit standard deviation, what happens to `y = x @ W`? Each output sums 10 (the *fan-in*) independent products, so:

$$\text{std}(y) = \sqrt{\text{fan\_in}} \approx \sqrt{10} \approx 3.16$$

The distribution **spreads** by $\sqrt{\text{fan\_in}}$. To keep it unit-Gaussian, divide the weights by $\sqrt{\text{fan\_in}}$:

```python
W = torch.randn((fan_in, fan_out)) / fan_in**0.5
```

That's the core of **Kaiming/He initialization** from the paper *"Delving Deep into Rectifiers"*. But there's a subtlety: the nonlinearity also squashes. Since tanh (and ReLU) are *contractive* — they squeeze the distribution — we multiply by a **gain** to push back:

| Nonlinearity | Gain |
|---|---|
| linear (no activation) | 1 |
| ReLU | $\sqrt{2}$ (clamps away half the distribution) |
| tanh | $\tfrac{5}{3}$ (fights the squashing) |

For our first layer, fan-in $= n_{embd} \times block\_size = 10 \times 3 = 30$:

```python
W1 = torch.randn((n_embd * block_size, n_hidden)) * (5/3) / (n_embd * block_size)**0.5
# std = (5/3) / sqrt(30) ≈ 0.30   (vs the guessed 0.2!)
```

PyTorch has this built in: `torch.nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='tanh')` — the most common init in practice.

With Kaiming init we land at the **same 2.10** validation loss — but without any hand-tuning. Modern practice has made even this less critical, thanks to a few *modern innovations*: **residual connections**, **normalization layers** (batch/layer/group), and better optimizers (**Adam**). Which brings us to the main event.

# Batch normalization

Paper: *"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"*, Ioffe & Szegedy, Google, 2015.

We keep saying *"we want the pre-activations to be roughly unit Gaussian"*... so **why not just normalize them?** That's the (initially crazy-sounding) insight of batchnorm: standardizing a tensor to zero mean and unit variance is a perfectly **differentiable** operation, so we can insert it into the network as a layer.

## The manual implementation

We insert it between the hidden linear layer and the tanh (its customary position — after the linear layer, before the nonlinearity):

```python
hpreact = embcat @ W1 + b1          # [32, 200]: 32 examples x 200 neurons

# --- batch normalization ---
bnmeani = hpreact.mean(0, keepdim=True)                 # [1, 200] mean over the batch
bndiff  = hpreact - bnmeani                             # center it
bndiff2 = bndiff**2
bnvar   = 1/(n-1) * bndiff2.sum(0, keepdim=True)        # variance (n-1 = unbiased)
bnvarinv = (bnvar + 1e-5)**-0.5                         # 1/std (eps prevents div-by-zero)
bnscale = bnvarinv * bndiff                             # now unit Gaussian per neuron
bnscale_bias = bnscale * bngain + bnbias                # scale & shift (learnable!)
hpreact = bnscale_bias                                  # back into the network

h = torch.tanh(hpreact)                                 # healthy activations!
```

*The mean and variance are computed over the **batch** dimension (dim 0) for each of the 200 neurons separately* — hence *batch* normalization.

## Why the extra gain and bias?

If we *only* standardized, every neuron's pre-activation would be forced to be exactly unit Gaussian **forever**, and the network could never make some neurons more "trigger happy" than others. So batchnorm adds two **trainable** parameters, initialized to the identity:

```python
bngain = torch.ones((1, n_hidden))    # gamma — scale
bnbias = torch.zeros((1, n_hidden))   # beta  — shift
```

At init, the output is exactly unit Gaussian *no matter what* `hpreact` looks like. During training, gradients flow into `bngain`/`bnbias` and let the network move the distribution around. So batchnorm decouples **"what the statistics are"** (fixed by the normalization) from **"where the distribution sits"** (learned).

>[!NOTE] The "terrible cost": examples get coupled
> Before, examples in a batch were processed **independently** — batching was just an efficiency trick. With batchnorm, an example's activation now depends on *which other examples happen to be in its batch* (they influence the mean/var). The activations "jitter" from batch to batch. This sounds like a bug, but it acts as a **regularizer** — it pads/jitters the inputs like a mild data augmentation, making it harder to overfit. This side effect is a big reason batchnorm is hard to remove even though nobody likes the coupling (it causes lots of bugs).

## The test-time problem: you need a batch, but you have one example

At test time we want to feed a **single** example. But batchnorm needs a batch to estimate mean/var! The paper's proposal comes in two flavors:

**Option 1 — stage 2 calibration:** after training, run the whole training set through the network once (`torch.no_grad()`, no gradient bookkeeping), estimate the mean/std once over all of it, and freeze those numbers for inference.

**Option 2 — running statistics (what everyone actually does):** estimate the stats *while training*, using an exponential moving average. No second stage needed:

```python
with torch.no_grad():    # this update is NOT part of gradient descent
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running  = 0.999 * bnstd_running  + 0.001 * bnstdi
```

- Initialized as `bnmean_running = 0`, `bnstd_running = 1` (consistent with proper init making `hpreact` roughly unit Gaussian).
- `0.001` is the **momentum** — how much the running estimate trusts each new batch. The PyTorch default is `0.1`, but with a batch as small as 32 the batch stats are noisy; momentum `0.1` makes the running stats *thrash* instead of converging, so Karpathy uses `0.001`. (Larger batches can afford larger momentum.)
- The running mean/std are **buffers**, not parameters: they are never trained by backprop, just updated with this "janky" running-mean update on the side.

The running estimates end up close to (but not identical to) the explicit stage-2 estimates, and the validation loss is essentially the same.

## Two more details

- **$\varepsilon = 10^{-5}$** in `bnvar + 1e-5`: prevents division by zero if a batch has exactly zero variance. Doesn't change results meaningfully; Karpathy skips it in the simple example.
- **The bias before batchnorm is useless.** Whatever `b1` adds gets subtracted right back out by the centering — so `b1.grad` is exactly zero, it never learns, it's just wasted parameters. When a layer is followed by batchnorm, drop its bias entirely (`bias=False`), because batchnorm has its own.

## BatchNorm1d in PyTorch

The manual code above is exactly `torch.nn.BatchNorm1d(num_features=200)`:

- **Parameters:** `gamma` (gain), `beta` (bias) — trained by backprop (`affine=True`).
- **Buffers:** `running_mean`, `running_var` — updated by EMA (`track_running_stats=True`), used at inference.
- **`eps`** (default `1e-5`), **`momentum`** (default `0.1`, use ~`0.001` for small batches).
- Like most `nn.Module`s it has a `.training` flag: batch stats + running-stat update during training; frozen running stats during eval.

The standard **motif** in modern networks is *weight layer → normalization → nonlinearity* — e.g. in a ResNet: `Conv → BatchNorm → ReLU` (with `bias=False` on the conv, exactly because of the point above).

# Bonus: torch-ifying the code and diagnostic plots

Karpathy wraps the layers into module classes (`Linear`, `BatchNorm1d`, `Tanh`) with the same API as PyTorch, then stacks them into a **6-layer** MLP (46,000 parameters): Linear → Tanh → Linear → Tanh → ... → Linear → Softmax. This lets us *systematically* inspect the network's health:

1. **Forward activation histograms** — at every tanh layer, plot the histogram of `h`, plus the mean, std, and **% saturation** ($|t| > 0.97$, i.e. in the gradient-killing tail). With gain $\tfrac{5}{3}$ the std stabilizes around **0.65** with ~5% saturation (the first layer starts more saturated, ~20%, then everything settles down). Gain too small (1) → activations slowly **shrink to zero** through the layers. Gain too big (3) → everything saturates.
2. **Backward gradient histograms** — gradients at every layer should have roughly similar magnitude; if they shrink or explode across layers, that's bad (and this is *exactly* why RNNs, which are just very deep unrolled networks, are hard to train — a teaser for the next part).
3. **Weight histograms** — mean/std of the parameters themselves.
4. **Update-to-data ratio** — the real diagnostic:

```python
update_to_data = log10( (lr * p.grad).std() / p.data.std() )
```

This compares the size of the *update we'll apply* to the size of the *values being updated*. A good heuristic: about **$10^{-3}$** (log scale ≈ −3). Much higher → learning rate too big; much lower (e.g. −4) → training too slow. The **last layer** is typically an outlier (its weights were artificially shrunk by ×0.01 to fix the softmax, so the ratio starts high until it grows into its weights). Forgetting fan-in normalization entirely shows up immediately: saturated activations, scrambled gradients, and ratios like −1 to −1.5.

## With batchnorm, everything becomes robust

Insert a `BatchNorm1d` before every tanh:

- The activation histograms *necessarily* look perfect (std ≈ 0.65, ~2% saturation) — every layer is normalized by construction.
- Changing the linear gains barely matters (e.g. gain 2 vs $\tfrac{5}{3}$ — activations identical). Even **no fan-in normalization at all** works, though you may need to retune the learning rate (Karpathy needed ~10× larger) because batchnorm rescales the incoming gradients.

*Why even keep the tanh layers, if they cause all this gain-tuning trouble?* Because a stack of pure linear layers **collapses to a single linear transformation** — no added representational power, no matter how deep. The nonlinearity is what turns the sandwich into a universal approximator.

# Results and takeaways

| Fix | Validation loss |
|---|---|
| Original MLP (Part 2) | 2.17 |
| + logits ≈ 0 at init (fix confident softmax) | 2.13 |
| + tanh not saturated (scaled W1, b1) | 2.10 |
| + Kaiming init (principled, no magic numbers) | 2.10 |
| + BatchNorm | ~2.10 |

*BatchNorm didn't actually improve the final loss here!* That's expected: with one hidden layer we could already compute the right weight scale by hand, so batchnorm has little to do. Its payoff comes in **deep** networks where hand-tuning every layer's scale is intractable. And our loss isn't bottlenecked by *optimization* anymore — it's bottlenecked by **context length** (3 characters is just not enough). Pushing further needs more powerful architectures: recurrent networks and transformers, which is exactly what the next parts of the tutorial build toward.

>[!NOTE] The moral of the story
> 1. At initialization, expect the loss you'd get from a **uniform** distribution — anything much worse means the network is confidently wrong.
> 2. Keep activations **roughly unit-Gaussian** throughout the net: not too small (tanh inactive, gradients shrink), not too big (tanh saturated, gradients killed).
> 3. Initialize weights with **Kaiming init** rather than magic numbers.
> 4. **BatchNorm** is the first normalization layer: center + scale with a learnable gain/bias, running stats for test time. It's powerful, but it couples the examples in a batch — which causes bugs, and why people increasingly prefer layer/group normalization.
