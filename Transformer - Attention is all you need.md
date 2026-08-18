# Background
Before transformers, we use RNN to process sequential data, just like what we did in [[Building of Wavenet#What is Wavenet|Makemore]]. However, due to its limitation of limited memory and attention, we adapted it using a **recursive attentions** architecture --- **Transformer**. 

# Model Architecture
![[Pasted image 20260818093408.png|310]]

Just like [[Basic MLP Overview|RNN]], we start with an **input embedding** and **expected output embedding**. 
However, in addition to merely embedding the word's information, we add a **positional encoding**. 
> For instance, in "I love easy courses", to encode easy, we need to add information of the neighboring words, plus the fact that it is at the third in the sentence. 

The left hand side is the **encoder**, and the right hand side is the **decoder**. 
The only layer that's different from [[Building of Wavenet#Building Wavenet|RNN]] is the Attention layer. 

