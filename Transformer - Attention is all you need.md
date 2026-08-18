# Background
Before transformers, we use RNN to process sequential data, just like what we did in [[Building of Wavenet#What is Wavenet|Makemore]]. However, due to its limitation of limited memory and attention, we adapted it using a **recursive attentions** architecture --- **Transformer**. 

# Model Architecture
![[Pasted image 20260818093408.png|310]]

## Embedding
Just like [[Basic MLP Overview|RNN]], we start with an **input embedding** and **expected output embedding**, which converts each word into a 512 dimension vector. 
This

However, in addition to merely embedding the word's information, we add a **positional encoding**. 

> [!NOTES] Position Encoding
> For instance, in "I love easy courses", to encode easy, we need to add the fact that it is at the third in the sentence. 
> 
> We use sine and cosine to encode position: 
> ![[Pasted image 20260818104911.png|321]]
> Where `pos` is the position, and `i` is the dimension. 
> Compared to *merely numbers*, a large vector allows the model to grasp the concept of position better, since the **position difference** of all numbers are words in the this vector in some dimensions. 
> 
> We **add** the encoded position vector onto the word embedding (surprise, *not dot product*). 

## Encoder
The left hand side is the **encoder**, and the right hand side is the **decoder**. 
The encoder is composed of `N = 6` layers. Each layer is composed of a [[List of Layers#Recursive Attention|Attention]] layer and a [[List of Layers#Linear|Linear layer]] for feed forward. 

![[Pasted image 20260818110748.png|166]]

### Recursive Attention
More detailed explanation are in [[List of Layers#Multi-head Self-attention]]. 
The input produces a **query** (`Q`), a **key** (`K`), and a **value** (`V`), through three different networks (`Wq`, `Wk`, and `Wv`). 