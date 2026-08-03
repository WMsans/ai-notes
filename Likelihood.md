# Likelihood
## What is Likelihood?

In the context of the bigram name generator, **likelihood** measures *how good the entire training dataset is under our model*.

For a single name like `"alex"`, the model generates it one letter at a time using bigram probabilities. The probability of the whole name is the product of each step:

$$\text{Likelihood}(\text{"alex"}) = P(\text{a} \mid \texttt{<start>}) \times P(\text{l} \mid \text{a}) \times P(\text{e} \mid \text{l}) \times P(\text{x} \mid \text{e}) \times P(\texttt{<end>} \mid \text{x})$$

For the **entire dataset** of 32,000 names, likelihood is the product of probabilities across *all* names:

$$\text{Likelihood} = \prod_{\text{all names}} * \prod_{\text{likelihood in each name}} *\ \ P(\text{next letter} \mid \text{current letter})$$

A higher likelihood means the model assigns higher probability to the names that actually appear in the dataset — i.e., the model "believes" the training data is likely. That's what we want.

> [!NOTE]
Likelihood is a good loss function because **it directly measures fit.** A model that assigns higher probability to the actual observed data is a better model. 
\
For instance, if a bigram appears in actual names that receives overly small probability, the likelihood would **reflect** it and thus punish it in the training process. 
## The Problem with Likelihood

Every probability $P$ is a number between $0$ and $1$. Multiplying thousands of them together produces a number so tiny that computers can't represent it (numerical **underflow**). For example:

$$0.01 \times 0.05 \times 0.1 \times \dots \text{ (10,000 times)} \approx 10^{-10000}$$

This is effectively zero in floating-point arithmetic.
## Log Likelihood

Instead of multiplying probabilities, we take the **natural log** of each and **sum** them:

$$\log\text{-Likelihood} = \sum_{\text{all names}} \sum_{\text{each bigram in name}} \log\big(P(\text{next letter} \mid \text{current letter})\big)$$

This works because of two properties of logarithms:

1. $\log(a \times b) = \log(a) + \log(b)$ — the product becomes a sum
2. $\log$ is **monotonically increasing** — if likelihood goes up, log-likelihood goes up too, so maximizing one is equivalent to maximizing the other

### Why This Fixes Underflow

Probabilities are tiny (e.g., $0.001$), but their logs are manageable negative numbers (e.g., $\log(0.001) \approx -6.9$). Summing thousands of these stays within a reasonable range.

| Value | log(Value) |
|-------|-----------|
| $1.0$ | $0$ |
| $0.1$ | $-2.30$ |
| $0.01$ | $-4.61$ |
| $0.001$ | $-6.91$ |
| $0.0001$ | $-9.21$ |

### From Log-Likelihood to Loss

Since probabilities are $\leq 1$, their logs are $\leq 0$, so log-likelihood is always **negative**. For a loss function, we prefer positive numbers where *lower is better*, so we flip the sign:

$$\text{Negative Log-Likelihood (NLL)} = -\sum \log\big(P(\text{each bigram})\big)$$

We often average it per bigram for a standardized measure:

$$\text{Loss} = \frac{1}{N} \cdot (-\text{Log-Likelihood})$$

Now:
- **High likelihood** → **high log-likelihood** (close to 0) → **low loss**
- **Low likelihood** → **low log-likelihood** (very negative) → **high loss**

## Maximum Likelihood Estimation (MLE)

The training objective: find model parameters that **maximize** the likelihood of the training data — or equivalently, **minimize** the negative log-likelihood loss.

For the bigram model, the optimal parameters are simply the empirical frequencies:

$$P(b \mid a) = \frac{\text{count}(a, b)}{\sum_{x} \text{count}(a, x)}$$

In other words: *count how many times each letter pair appears, then normalize by row.* This is exactly what the bigram frequency matrix in [[Building of Bigram]] does — it's the MLE solution.

