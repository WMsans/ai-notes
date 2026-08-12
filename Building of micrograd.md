# Back propagation
Back propagation is a system that calculates every values' direvative relating to the final output value L. 
### Simple Example
	a * b + c = L

In this case, we would back propagate from **L to ab and c**, then from **ab to a and b**. 
$\frac{dc}{dL} = 1$, since for every 1 increase in c, L would increase 1.
$\frac{d(ab)}{dL} = 1$ as well.

Now back propagate to a and b. 
$\frac{da}{d(ab)} = b$
But, according to chain rule: $\frac{da}{dL} = \frac{da}{d(ab)} \cdot \frac{d(ab)}{dL}$
Therefore: $\frac{da}{dL} = b \cdot 1 = b$

Apply the same process to b: $\frac{db}{dL} = \frac{db}{d(ab)} \cdot \frac{d(ab)}{dL} = a \cdot 1 = a$

In production, back propagation is ready for us to use by using **pytorch** library. 
# Neural Network
![[Pasted image 20260719115028.png|473]]
A neural network is composed of neuron cells. 
Each cell does a set of simple calculation: 
1. takes output from neuron cells from last layer (`x`)
2. multiply each output by a weight (`w`)
3. sum all of them together ($x_1 w_1 + x_2 w_2 + \dots + x_n w_n$)
4. add a bias (`b`)
5. pass this value to an activation function. It would be non-linear and compress the result in (0, 1). Usually, it would be [[List of Layers#Tanh|`tanh`]].
6. output this value to the next layer of neuron cells

If we connect the layers of cells together, we get: 
![[Pasted image 20260719115605.png|476]]
Except the input layer (which outputs a tuple of constants), each layer are neurons that does the calculation of adding all weighted values together and passing them down. 
# Training the neural network
For instance, we have a list x: 
	x=[
	[2.0,3.0,-1.0],
	[3.0,-1.0,0.5],
	[0.5,1.0,1.0],
	[1.0,1.0,-1.0]
	]
Each item tuple in x is a input layer. Our goal of prediction, let's say would be:
	ys = [1.0,-1.0,-1.0,1.0] 
At first, the neural network (with randomized weights) would produce very random predictions. For instance: 
	Value(data=0.9160777125976044), *Should be 1.0*
	Value(data=0.8852695869413116), *Should be -1.0*
	Value(data=0.8829321387652947), *Should be -1.0*
	Value(data=0.8808920670754451) *Should be 1.0*

## Loss Function

To train the network, we need a way to measure *how wrong* our predictions are. This measurement is called the **loss** (or cost/error).

### Mean Squared Error (MSE)

A common loss function for regression is Mean Squared Error:

$$L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

Where we are adding all the **difference** from *prediction* ($\hat{y}_i$) to the *expecting value* ($y_i$). Squaring ensures that errors are positive and large errors get penalized more heavily. Finally, we divide the sum by N to average them. 

In code:
```python
# Forward pass: get predictions for all 4 examples
preds = [n(x) for x in xs]  # n is our MLP

# Calculate loss as sum of squared errors
loss = sum((p - y)**2 for p, y in zip(preds, ys))
```

This `loss` is itself a `Value` node in the computation graph, just like any other — it has a `data`, a `grad`, and pointers to its children via the `_prev` set. The difference is that the loss sits at the very **top** of the graph: it's the final scalar that captures the entire network's performance.

## Backpropagation Through the Whole Network

Once we have the loss value, we trigger backpropagation from it. This is the same chain-rule process described earlier, but now it flows backwards through **every** operation in the network:

```python
# Zero out all gradients from the previous step
for p in n.parameters():
    p.grad = 0.0

# Magic happens!
loss.backward()
```

The backward pass traces the computation graph in reverse topological order. Starting from the loss:

1. **Loss node** ($L$): gradient is $\frac{dL}{dL} = 1$
2. **Sum node** (summing per-sample errors): each term's local derivative distributes $1 \cdot \frac{\partial L}{\partial \text{term}_i}$
3. **Squared-error nodes**: each $(\hat{y}_i - y_i)^2$ passes gradient $2(\hat{y}_i - y_i)$ to the subtraction node
4. **Subtraction nodes**: gradient flows to $\hat{y}_i$ (unchanged) and $y_i$ (negated) — but $y_i$ is a constant, so its gradient is ignored
5. **Output neurons**: gradients flow through [[List of Layers#Tanh|`tanh`]], then through weighted sums, splitting across all incoming weights
6. **Hidden layers**: the same pattern repeats — [[List of Layers#Tanh|`tanh`]] gradient $\cdot$ incoming gradient, then branched out across weights and biases
7. **Weights and biases**: every `w` and `b` in the network now has a populated `.grad` — the partial derivative $\frac{\partial L}{\partial w}$ (or $\frac{\partial L}{\partial b}$)

The [[List of Layers#Tanh|`tanh`]] activation's local derivative is key: $\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$. This is why we store the [[List of Layers#Tanh|`tanh`]] output during the forward pass — the backward pass needs it to compute the local gradient.

## Gradient Descent — Updating the Weights

Now each parameter knows how much the loss would change if we nudge it slightly. We nudge each parameter in direction of its **negative** gradient to **reduce** the loss:

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$

Where $\eta$ (eta) is the **learning rate** — a small scalar like `0.01` or `0.001` that controls the step size:

```python
learning_rate = 0.01
for p in n.parameters():
    p.data += -learning_rate * p.grad
```

This is called **gradient descent**: we take a small step downhill on the loss surface. The gradient tells us the direction of steepest *ascent*, so we move against it.

### Why Small Steps?

- Too large a learning rate: we overshoot the minimum or diverge entirely.
- Too small: training takes forever.
- The right value depends on the problem and is typically found through experimentation.

## The Training Loop

Training is just repeating the above steps many times:

```python
for epoch in range(epochs):
    # 1. Forward pass: compute predictions
    preds = [n(x) for x in xs]

    # 2. Compute loss
    loss = sum((p - y)**2 for p, y in zip(preds, ys))

    # 3. Zero gradients
    for p in n.parameters():
        p.grad = 0.0

    # 4. Backward pass: compute gradients
    loss.backward()

    # 5. Update parameters (gradient descent)
    for p in n.parameters():
        p.data -= learning_rate * p.grad

    # Optionally: print loss to monitor progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.data:.4f}")
```

After enough epochs (iterations), the loss drops and predictions converge toward the target values:

| Epoch | Loss  | Predictions (vs targets)                   |
| ----- | ----- | ------------------------------------------ |
| 0     | ~4.84 | [0.92, 0.89, 0.88, 0.88] vs [1, -1, -1, 1] |
| 100   | ~0.45 | [0.72, -0.65, -0.68, 0.71]                 |
| 200   | ~0.08 | [0.94, -0.92, -0.91, 0.93]                 |
| 300   | ~0.01 | [0.99, -0.98, -0.98, 0.99]                 |

