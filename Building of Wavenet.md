We are going to improve [[Building of MLP|MLP]] for a deeper, larger, and more comprehensive model. 
You may first need to review the [[List of Layers]]. 

# What is wavenet

>[!NOTE] Origin
>WaveNet is an autoregressive deep generative model for raw audio waveforms, introduced by DeepMind in 2016. Instead of predicting the next word or token, it models the probability of each audio sample given all the samples that came before it, one sample at a time. This lets it generate extremely realistic speech, music, and other sounds directly at the waveform level.

Its key architectural ideas are:

- **Dilated causal convolutions** — convolutions that only look backwards in time, with exponentially increasing dilation so a few layers can cover thousands of samples.
- **Gated activation units** — a tanh gate multiplied by a sigmoid gate, which helps capture complex audio dynamics.
- **Skip and residual connections** — letting gradients flow through the deep stack of layers.

Although WaveNet was originally used for text-to-speech, but the same architecture also powers models like SampleRNN's successors and is the conceptual ancestor of many modern generative audio models. 

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

And out **training loop** is merely iterating through our layers: 
![[Pasted image 20260812121525.png|357]]

