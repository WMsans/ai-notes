MLP, standing for multilayer perceptron, is neural network made of **fully connected** neurons arranged in an *input layer*, one or more *hidden layers*, and an *output layer*. The [[Building of micrograd|micrograd]] model is an example of MLP. 
![[Pasted image 20260719115605.png|476]]

In this note, we upgrade our [[Building of Bigram|bigram model]] into an MLP character predictor, following the paper **"A Neural Probabilistic Language Model"** by *Bengio, Ducharme, Vincent and Jauvin* (2003).

# The problem: more context explodes the table

Our bigram model only looks at **one** previous character. If we want better names, we want to look at *more* context — say, the last 3 characters. But the counting-table approach quickly explodes:

| Context length | Number of possible contexts |
|---|---|
| 1 character | 27 |
| 2 characters | $27 \times 27 = 729$ |
| 3 characters | $27^3 \approx 20{,}000$ |

*More context = exponentially more rows, and almost no counts per row.* This is the **curse of dimensionality**. 

# The solution: distributed representations

The paper's key idea: stop counting exact sequences, and instead give **every symbol a learned vector** (a *feature vector*). The model then predicts the next character from *vectors*, not from counts.

This would work because the probability function is a **smooth** function of the vectors: if two symbols have similar vectors, the model treats them as interchangeable. The paper's famous example:

- The model sees *"The **cat** is **walking** in the **bedroom**"* during training.
- At test time it is asked to predict *"A **dog** was **running** in a **room** —"* which it has **never seen**.
- Because *cat/dog*, *walking/running*, *bedroom/room*, *the/a* end up as **nearby vectors**, the new sentence gets high probability anyway.

*Knowledge transfers between similar symbols. That is generalization.*

>[!NOTE] The paper learns two things at once
>1. A **distributed representation** (embedding vector) for every word/character
>2. The **probability function** expressed in terms of those vectors
>
>Both are learned together by maximizing the log-likelihood of the training data ([[Likelihood|maximum likelihood estimation]]).

# The architecture (Figure 1 of the paper)

The model's architecture (we use characters, $|V| = 27$) is:

![[Pasted image 20260731225921.png|485]]

1. **Embedding lookup table $C$** — a matrix of size $|V| \times m$ (e.g. $27 \times 30$). Each symbol's index *plucks out a row*: its learned vector (which is *30 dimensions* in this case). Crucially, $C$ is **shared** across all positions in the context.
2. **Concatenation** — the vectors of the last $n-1$ symbols are glued together into one big input vector $x = (C(w_{t-1}), C(w_{t-2}), \dots, C(w_{t-n+1}))$.
3. **Hidden layer** — a fully connected layer followed by a $\tanh$ non-linearity.
4. **Output layer** — produces one raw score ([[Logits|logit]]) per possible next symbol.
5. **Softmax** — turns the logits into probabilities.

The the whole network can be summarized as one formula:

$$y = b + Wx + U \tanh(d + Hx)$$

where:
- $H$, $d$ — weights and bias of the hidden layer
- $U$, $b$ — weights and bias of the output layer
- $W$ — optional *direct* connections from input to output (set to 0 in our version)

Then $\hat{P}(w_t \mid \text{context}) = \frac{e^{y_{w_t}}}{\sum_i e^{y_i}}$, and the model is trained by maximizing the [[Likelihood|log-likelihood]] $\frac{1}{T}\sum_t \log \hat{P}(w_t \mid \text{context})$. 

# Building the model

## Step 1: the dataset

We choose a **block size** — how many characters of **context** to use (in this example, we use 3). Then we slide a rolling window over every name:

```python
block_size = 3
# for "emma" (padded with the special start token "."):
# [., ., .] -> 'e'     [., ., e] -> 'm'
# [., e, m] -> 'm'     [e, m, m] -> 'a'
# [m, m, a] -> '.'
# Note that each context 
```

`X` holds the contexts (one row of `block_size` integers per example), `Y` holds the label — the index of the actual next character.

## Step 2: the embedding lookup table C

```python
C = torch.randn((27, 2))   # 27 characters, embedded in 2D (so we can visualize later)
emb = C[X]                 # emb.shape -> [N, 3, 2], the vectors for every context. N = example number. 
```

*Indexing into C with a tensor of indices just works in PyTorch — no loop needed.*

>[!RECALL] Why is this the same as one-hot encoding?
>From [[One-hot encoding]]: a one-hot vector multiplied by a matrix **picks out one row** of the matrix (all other entries are multiplied by 0). So `one_hot(x) @ C == C[x]`. The embedding layer is just a **linear layer with no activation**. Wring in index form is just for convenience. 

## Step 3: the hidden layer

The embeddings of the 3 context characters are stacked in a 3rd dimension, so we first flatten them (concatenate the 3 vectors into one):

```python
emb_cat = emb.view(-1, 6)          # [N, 3, 2] -> [N, 6]  (concatenationk)
W1 = torch.randn((6, 100), requires_grad=True)   # the hidden layer can have any number of nuerons we like. For here, we have 100 hidden neurons
b1 = torch.randn(100, requires_grad=True)
h = torch.tanh(emb_cat @ W1 + b1)  # hidden layer activations, [N, 100]
```

*The size of the hidden layer (100) is a **hyper parameter** — a design choice we can tune.*

## Step 4: the output layer and probabilities

```python
W2 = torch.randn((100, 27), requires_grad=True)  # 27 possible next characters
b2 = torch.randn(27, requires_grad=True)
logits = h @ W2 + b2               # raw scores, [N, 27]
counts = logits.exp()              # "fake counts"
probs = counts / counts.sum(dim=1, keepdim=True)  # each row sums to 1
```

This is the [[Softmax|softmax]] we already know. `probs[i, j]` = the model's probability that character `j` comes after context `i`.

## Step 5: the loss

Same as in the bigram model — we look up the probability the model assigned to the *correct* next character and take the negative log:

```python
loss = -probs[torch.arange(N), Y].log().mean()
```

In practice we use PyTorch's built-in **cross entropy** function, which computes the **exact same number** but much better:

```python
loss = F.cross_entropy(logits, Y)
```

*Why prefer `F.cross_entropy`?* It fuses softmax + log + indexing into one efficient kernel (no giant intermediate tensors), has a cleaner backward pass, and is numerically stable (it subtracts the max logit internally, avoiding overflow — see [[Softmax]]).
## Model Overview
For a overview for the model, see [[MLP Computation Graph.html]]. 
![[Pasted image 20260803100737.png]]
# Training

## Sanity check: overfit a tiny batch

Before training on everything, we check the network *can* learn at all: we feed it only 5 words (32 examples) and train for 1,000 steps. The loss drops from ~17 to near zero — the network is **overfitting** those 32 examples (3,400 parameters vs 32 examples, so it easily can).

*Fun detail: it can't reach exactly 0, because the same context can have different labels — the context `...` predicts `e`, but also `o`, `a`, `s` in other names.*

## Mini-batches

The full dataset has **228,000 examples** — forward/backward on all of them per step is too slow. Instead we take a random **mini-batch** of 32 each step:

```python
ix = torch.randint(0, X.shape[0], (32,))
xb, yb = X[ix], Y[ix]      # only 32 random examples per step
```

The gradient is now a *noisy estimate* of the true gradient, but each step is ~7,000× cheaper — so we can take many more steps, and in practice it converges much faster. *Noisy but cheap beats exact but expensive.*

## The learning rate

`0.001` — loss barely moves (too small). `0.01` — decreases steadily. `0.1` — fast but wobbly (goes up and down). `1` and up — diverges completely. So the good range is somewhere in between; we can search it systematically by stepping **exponentially** over the powers of 10:

```python
lrs = 10 ** torch.linspace(-3, 0, 1000)   # 0.001 -> 0.01 -> ... -> 1.0
```

*Exponential spacing because learning rates matter on a log scale (0.01 and 0.02 are similar; 1 and 2 are not).*

## Results

Training a slightly bigger version (10-dim embeddings, 200 hidden neurons, ~11,000 parameters; learning rate 0.1 decayed to 0.01) gives:

| | Loss |
|---|---|
| Bigram baseline (random init) | $\ln(27) \approx 3.29$ |
| MLP training set | ~2.16 |
| MLP validation set | ~2.19 |

*Clearly beating the bigram baseline. (Karpathy's best validation loss in the video: 2.17.)*

# Sampling names

To generate names we roll the context window forward, same idea as [[Building of Bigram|the bigram sampling loop]]:

```python
for _ in range(20):                    # generate 20 names
    context = [0] * block_size         # start: all dots (index 0)
    out = []
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()   # sample, don't argmax
        context = context[1:] + [ix]   # shift the window
        out.append(ix)
        if ix == 0: break              # stop at the special token
    print(''.join(itos[i] for i in out))
```

*We sample with `torch.multinomial` (weighted random pick), not argmax — argmax would give the same boring name every time.*

The outputs — like **ham, joes, lela** — already sound much more name-like than the bigram's gibberish.

# What the embeddings learned

Because we embedded in 2D, we can plot every character's vector after training:
![[Pasted image 20260803102352.png|397]]

- **Similar characters cluster together** — e.g. vowels sit near each other.
- `q` is an exception — pushed far away, because it behaves specially (always followed by `u`).
- The special dot token sits all alone — it's a special character.

*The structure is not random — it emerged purely from training.* This is exactly the "similar symbols get similar vectors" claim of the paper, visible in 2D.