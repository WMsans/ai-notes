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
> We can have arbitrary number of dimensions like [4, 5, 6, 80], as long as the weights has [80, X]. 

Currently, the flatten layer convert the input size [4 examples, 8 context characters, 10 characteristic vector] to linear size [4 examples, 80 compressed context and characteristics]. Our goal is to flatten it to [4 examples, 4 context bigrams, 20 bi-characteristic vectors]. 

![[Pasted image 20260815111725.png|240]]
`n` is how much we want to compress. In our case, `n = 2`. 

## Building Wavenet
We are going to consecutively flatten the layer, each by a scale of 1/2. 

![[Pasted image 20260815113205.png|568]]

Now, if we iterate through the shape of each layer
```python
for layer in model.layers:
	print(layer.out.shape)
```
We get
![[Pasted image 20260815113407.png|220]]
> The first flatten is [4, 4, 20]
> The first linear brings it up to [4, 4, 20], since there's 200 hidden neurons in that layer
> 
> The second flatten is [4, 2, 400]
> The second linear bings it back to [4, 2, 200], since there's still 200 hidden neurons in that layer
> 
> The final flatten is [4, 400]. We squeeze out the second dimension since it is [1]. 

It is the exact 3-layer structure as the graph now: 
![[Pasted image 20260814092705.png|505]]

## Batchnorm in Wavenet
Now, the `torch.mean(dim=0, keepdim=true)` and `torch.var(dim=0, keepdim=true)` only take the mean or standard deviation for the first dimention. 

![[Pasted image 20260815151045.png|246]]
> The bigram dimension is not averaged properly, and thus would not be normalized. 

Luckily, the `dim` parameter can be a tuple, meaning taking multiple dimensions at once.
```python
e.mean((0, 1), keepdim=True)
e.var（（0,1）,keepdim=True)
```
So we can get the correct dimensions: 
![[Pasted image 20260815155716.png|172]]
