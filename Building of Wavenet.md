We are going to improve [[Building of MLP|MLP]] for a deeper, larger, and more comprehensive model. 
You may first need to review the [[List of Layers]]. 

# What is Wavenet

- **Dilated causal convolutions** — convolutions that only *look backwards* in time, with exponentially increasing dilation so a few layers can cover thousands of samples.
- **Gated activation units** — a *tanh gate* multiplied by a *sigmoid gate*, which helps capture complex audio dynamics.
- **Skip and residual connections** — letting gradients flow through the *deep* stack of layers.

For us, it's the next step after the [[Building of MLP|MLP]]: the same idea of **stacking layers**, but with a much **deeper** and more **specialized structure** built for sequence data.
# Prepare for more layers
We are going to prepare some layers so that we can maintain the `layers` array instead of hardcoding things in the training loop. 

1. [[List of Layers#Embedding|Embedding]]
To replace the hardcoded 
```python
x = C[Xb] # Tokenize the input vectors
```

2. [[List of Layers#Flatten|Flatten]]
To replace the hardcoded
```python
x=emb.view(emb.shape[o]，-1)# concatenate the vectors to a single demention
```

Now, our **neural network** looks like this:
![[Pasted image 20260812121200.png|504]]

And out **training loop** is merely iterating through the layers list: 
![[Pasted image 20260812121525.png|357]]

# Wavenet Architecture
Previously we have a simple MLP with only a single hidden layer: 
![[Pasted image 20260731225921.png|485]]

We squash all the information in a single step, which would not have a very good result. 
Wavenet decides to compress information **step by step**: 
![[Pasted image 20260814092705.png|505]]

> First step, we compress the context characters into **bigrams**. Then, we compress the bigrams into **quadgrams**. Eventually, we can get the **final compressed form of the single predicted character**. 

## Prepare the data
### Increase context size
First, instead of using merely 3, we use 8 characters to predict the 9th character. 
![[Pasted image 20260814093553.png|132]]

Interestingly, with merely increased context size, we got much better loss and much better results. 
This is the **scaling law**. 
### Rework the [[List of Layers#Flatten|flatten]] layer
Instead of flattening everything into a single layer, we need to flatten it into bigrams. 

Since the [[List of Layers#Linear|linear layer]] can take any dimension of input, since [[Dot product]] only cares about the matching of the **last dimension of first input** to the **first dimension of second input**. 

![[Pasted image 20260814095432.png|474]]
	We can have arbitrary number of dimensions like [4, 5, 6, 80], as long as the weights has [80, X]. 
