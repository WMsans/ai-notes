## Encoding the words
Let's take the name *"Emma"* as an example. 
```python
w = "emma" # The name we are using
chs = ['<start>'] + list(w) + ['<end>'] # Extract it to chars
for ch1, ch2 in zip(chs,chs[1:]) # Iterate through the character pairs
	# Get the index of the characters
	ix1 = stoi[ch1]
	ix2 = stoi[ch2]
	print(ch1,ch2)
	# Encode the index of the characters
	xs.append(ix1)
	ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
```

The output would be 
```markdown
<start> e
e m
m m
m a
a <end>
```

`xs`, which indicates all the **starting character** in each pair, would look like
```python
tensor([0,5,13,13,1])
```
`ys`, which indicates all the **ending character** in each pair, would look like
```python
tensor([5,13,13,1,0])
```
Now, we can feed in `xs` and `ys` into the neural net.
## Convert data using one-hot
Conveniently, PyTorch provides us `torch.nn.functional.one_hot` function ([documentation]((https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.one_hot.html)).

>[!DOCUMENTATION]
>```python
>torch.nn.functional.one_hot(_tensor_, _num_classes=-1_) → LongTensor
>```
>
>What it does is generating a tensor where **everywhere is 0**, *except* for the location of the index that **included in the input tensor**.

```python
import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes = 27) 
# xs is [0,5,13,13,1]
# num_classes is all the possible outcome, here it is 26 letters and the starting/ending mark.
```
the `xenc` would be
![[Pasted image 20260725103839.png|474]]
Which is a [5, 27] tensor. For a better visualization: 
![[Pasted image 20260725103923.png]]
We see that for the first row, the index 0 (representing the **starting mark**) is activated while the rest of the position are 0. 
