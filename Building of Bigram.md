# A letter predictor
We have a list of 32000 names. We are going to train an ai model to create a name for us based on the dataset of names.

An approach is to create a dictionary of frequencies all letter combinations (grams) in the dataset.
![[Pasted image 20260721183756.png|344]]
	*For instance, combination "al" would be in position [1, 12]*
	*Specially, start and end are also a part of combination and have their index. For instance, "e \<end\>"*

Then, run a loop of repeatedly selecting letter using weighted random number generator, with the weight being the frequency of each combination. 
	*For instance, we start with \<start\>. We randomly select the next letter by the weighted generator. Then the next one and the next one.*
We would get some results that look terrible.
![[Pasted image 20260722121621.png]]
The model's performance is not so great. We are going to improve it. 
# Training loss
We are going to use a single number (loss) to evaluate how good this model is, just like in [[Building of micrograd]]
We use maximum **likelihood** estimation. [[Likelihood]] is the product of all the gram probabilities in each name, multiplied. In this specific case, we use **log likelihood**. 
## Building of neural network

>[!RECALL]
>To recall how neural networks are built: 
![[Pasted image 20260719115605.png|516]]
We have layers of neurons and each layer connects the next with weighted multiplication. We summarize all multiplications using the **loss** function. Then, using the **loss** function and **back propagation**, we tune the weights of the connections to get ideal results. 

First, we **encode** the words for efficiency. The encoding algorithm we are using is called [[one-hot encoding]]. It would convert words into tensors with activation bits like this:
![[Pasted image 20260725103923.png]]
Out neural net would only accept **encoded data** through [[dot product]].
### Building a neuron
First lets define the weights of a neuron. Initially, it is random. 
```python
W = torch.randn((27, 1))
```
W would be a tensor with **27 rows** and **1 column**. Then, the weights are going to multiply the data.
```python
xenc @ W
```
It would give us a tensor of [5,1], representing *5 inputs* and *1 neuron*. 
```python
tensor([[0.6886],
[0.7705],
[-0.2257],
[-0.2257],
[0.0838]])
```
Since it is a [[dot product]], each row is 
```python
xenc[row, 0] * W[0] + xenc[row, 1] * W[1] + ... + xenc[row, 26] * W[26]
```
However, since `xenc` is [[One-hot encoding|one hot encoded]], most indexes in `xenc` are *zero* *except* the **activated indexes**, so the result of the dot product would only get us
```python
xenc[row, activated_idx] * W[activated_idx]
```
This is the **activation value** of the *row_th input* (`wx + b`). Through [[Dot product|dot multiplication]], we can process the 5 inputs (rows) in parallel. 
### Building a [[List of Layers#Linear|layer]]
We want 27 neurons instead of just 1 per [[List of Layers#Linear|layer]].
```python
W = torch.randn((27, 27))
xenc @ W
```
It would give us a tensor of [5, 27], representing *5 inputs* and *27 neurons*. 
![[Pasted image 20260725121953.png|418]]
Likewise, each value is the **activation value** of the *row-th input* to the *column-th neuron* . 
	For instance, `(xenc @ W)[3, 13]` is the activation value of the *13th neuron* to the *3rd input*, according to the formula `wx + b`:  $$w_{13} \cdot x_3 + b$$
For now, let's treat the [[List of Layers#Linear|layer]] (`W`) as the neural net, since it can already **take an input** (`xenc`) and **produce an output**. We need to **interpret** the output (`xenc @ W`). 
### Interpreting the output

The output `xenc @ W` — shape `[5, 27]` — gives us raw scores called **[[Logits|logits]]**. In other words, they are log counts. They can be **negative** and **non-integers** instead of like counts, and they don't **add up to one** like probabilities. 

To convert logits into a probability distribution, we use the **[[List of Layers#Softmax|softmax]]** function:
```python
# Step 1: exponentiate to make everything positive
counts = logits.exp()

# Step 2: normalize so each row sums to 1
probs = counts / counts.sum(dim=1, keepdim=True)
```
This works because:
- `exp(x)` maps any real number to a positive number
- Dividing by the sum ensures each row becomes a proper probability distribution (all values in `[0, 1]` and summing to 1)

Now `probs[row, column]` is the model's predicted probability that the **column-th** character comes after the **row-th** input character. For example, `probs[2, 12]` is the predicted probability of character `'l'` appearing right after the 3rd input character.

#### Computing the loss

With probabilities in hand, we can compute the **negative log-likelihood** loss:

```python
loss = -probs[torch.arange(5), Y].log().mean()
```

Where `Y` is the indices of the *actual* next characters from the training data set. For each input position, we look up the probability assigned to the correct next character, take its log, negate it, and average across all 5 positions.

At initialization (random weights), each of the 27 characters gets roughly equal probability of `1/27`, so the initial loss should be around:

$$-\ln\left(\frac{1}{27}\right) = \ln(27) \approx 3.2958$$

This is our baseline — any useful model should beat this. As training progresses and the probabilities concentrate on the correct characters, the loss will drop. 

Interestingly, the bigram count-based model described in *the beginning of this note* the theoretical best this single-layer network can achieve (since it directly memorizes all pair frequencies). A way to build a better model is to use [[Building of MLP|a MLP model]] with **multiple layers of neurons** instead of just one. 
