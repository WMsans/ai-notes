# Background
Before transformers, we use RNN to process sequential data, just like what we did in [[Building of Wavenet#What is Wavenet|Makemore]]. However, due to its limitation of limited memory and attention, we adapted it using a **recursive attentions** architecture --- **Transformer**. 

# Model Architecture
![[Pasted image 20260818093408.png|310]]

## Embedding
Just like [[Basic MLP Overview|RNN]], we start with an **input embedding** and **expected output embedding**, which converts each word into a 512 dimension vector. We call it **token**. 
This

However, in addition to merely embedding the word's information, we add a **positional encoding**. 

> [!NOTES] Position Encoding
> For instance, in "I love easy courses", to encode easy, we need to add the fact that it is at the third in the sentence. 
> 
> We use sine and cosine to encode position: 
> ![[Pasted image 20260818104911.png|321]]
> Where `pos` is the position, and `i` is the dimension. 
> Compared to *merely numbers*, a large vector allows the model to grasp the concept of position better, since the **position difference** of all tokens are encoded. 
> 
> We **add** the encoded position vector onto the token embedding (surprise, *not dot product*). 

## Encoder
The left hand side is the **encoder**, and the right hand side is the **decoder**. 
The encoder is composed of `N = 6` layers. Each layer is composed of a [[List of Layers#Recursive Attention|Attention]] layer and a [[List of Layers#Linear|Linear layer]] for feed forward. 

![[Pasted image 20260818110748.png|166]] 

### Recursive Attention
More detailed explanation are in [[List of Layers#Multi-head Self-attention]]. 
#### Single headed attention
The input produces a **query** vector (`Q`), a **key** vector (`K`), and a **value** vector (`V`), through three different networks (`Wq`, `Wk`, and `Wv`). Each network has 512 nuerons (so their shape is [512, 512]). 

Then we can get attention using this formula: 
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

>[!NOTE] Formula explained
> As a graph: 
> ![[Pasted image 20260818123909.png|194]]
> 
> Here, `Q` stands for the query target, `K` stands for the words in context. 
> 
> First, `Q` and `K` of words in context does a [[Dot product]]. This is evaluating the **similarity** of this input word to all other words in the context.
> 
> Then, we scale it by factor of $$\frac{1}{\sqrt{d_k}}$$
> `d_k` is dimension of the keys. This is to prevent the product to grow in higher dimensions, and thus stabilize the training. 
> 
> After that, we use the classic [[List of Layers#Softmax|softmax]] layer to activate it into probabilities. 
> This probability stands for the **importance of this token's meaning to this word**. Of course, we could expect that the **token of the word itself** would be the most important one in the probability distribution. 
> 
> Finally, we [[Dot product|dot multiply]] the `V`, which is to do a weighted average.
> 
> The final product is a vector that contains **all the context information**, and **how important each context token is to this word**. 

#### Multi-headed Self-attention

A single attention head performs this calculation in one feature subspace. 

![[Pasted image 20260818180608.png|222]]

**Multi-head** attention is basically: 
1. splits the embedding into `n_heads` smaller subspaces
	1. Each subspace is its own `W_Q`, `W_K`, and `W_V`. 
	2. The subspace is smaller. If a full `W` is [512, 512], divided into 8 subspaces, then the sub `W` would be [512, 512/8 = 64]. 
2. runs attention in each one in parallel
	1. Since `W` are only [512, 64], the result of the [[dot product]] to the input [512] would be only [64]. 
3. concatenates the results
	1. Since there are 8 subspaces of [64] result, we concatenates them into a [64 * 8 = 512] result.  

This way, each subspace learns different relationships — for example, nearby context, syntax, or a long-range reference. 

### Linear Feedforward
Using the [[List of Layers#Linear|linear layer]] fuses the **concatenated heads** together, we avoid the bad shape of shabby concatenation artifacts. 

### LayerNorm
See [[List of Layers#Layernorm]]. This is to stabilize the and lightly randomize the result. 

### Residual Connection
We add the input vector directly on the output of the multi head attention layer. 
![[Pasted image 20260818194740.png|259]]

This is to prevent the gradient to shift too far from the input. 

