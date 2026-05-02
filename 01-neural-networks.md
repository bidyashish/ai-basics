# 01 · Neural Networks from First Principles

> **TL;DR** A neural network is a stack of matrix multiplications and simple non-linear functions. Training it means tweaking the numbers inside those matrices so the network's output gets closer to what you want. Backpropagation is just the chain rule from calculus, applied to the whole stack at once.

## 1. The big picture

Imagine you want a machine to look at a picture and say "cat" or "dog." You can't write rules for every possible photo, so instead you give the machine millions of labelled examples and let it learn the rules itself.

A **neural network** is the function that does this. It takes input numbers (the pixels), passes them through layers of multiply-and-add operations, and spits out output numbers (a probability for "cat" and one for "dog").

Training is the process of nudging the multiplications until the answers come out right. Inference is using the trained network on new inputs.

That's it. The rest is details.

---

## 2. A single neuron

The smallest building block is a **neuron**. It does three things:

1. Takes a list of inputs `x = [x1, x2, ..., xn]`.
2. Multiplies each input by its **weight** `w = [w1, w2, ..., wn]` and sums them: `z = w1*x1 + w2*x2 + ... + wn*xn + b`. The extra `b` is a **bias**.
3. Passes that sum through a non-linear **activation function** `a = f(z)`.

In NumPy:

```python
import numpy as np

def neuron(x, w, b):
    z = np.dot(w, x) + b      # weighted sum
    return max(0.0, z)        # ReLU activation
```

Why the activation? Without it, stacking neurons gives you nothing more than a single big matrix multiply, which can only learn straight lines. The non-linearity is what lets the network bend, twist, and carve up data into complex shapes.

### Common activations

| Name | Formula | Looks like | When to use |
|------|---------|-----------|-------------|
| **ReLU** | `max(0, x)` | hockey stick | Default for hidden layers. Fast. |
| **Sigmoid** | `1 / (1 + e^-x)` | S-curve from 0 to 1 | Old-school, output probabilities of one class. |
| **Tanh** | `(e^x - e^-x) / (e^x + e^-x)` | S-curve from -1 to 1 | Sometimes used in RNNs. |
| **GELU** | `x * Φ(x)` (Gaussian CDF) | Smooth ReLU | Standard in transformers (BERT, GPT-2). |
| **SiLU / Swish** | `x * sigmoid(x)` | Smooth ReLU | Used inside SwiGLU in Llama, Qwen. |

A useful instinct: **modern LLMs barely use ReLU** in their feed-forward blocks anymore. SiLU and GELU dominate because their smooth gradients train more stably.

---

## 3. From neurons to layers

A **layer** is just many neurons sitting side by side, all looking at the same input. If layer `L` has `m` inputs and `n` neurons, you can write the whole layer as a single matrix multiplication:

```
y = activation(x @ W + b)
# x: (m,)   input vector
# W: (m, n) weight matrix
# b: (n,)   bias vector
# y: (n,)   output vector
```

When you process a **batch** of `B` inputs at once, `x` becomes `(B, m)` and `y` becomes `(B, n)`. The matmul handles all `B` examples in parallel — this is why GPUs are so fast at this stuff.

A **multi-layer perceptron (MLP)** is a stack of these layers:

```
x → Linear → ReLU → Linear → ReLU → Linear → output
```

Each linear layer has its own `W` and `b`. The whole network has a few thousand to a few billion of these parameters.

---

## 4. The forward pass in PyTorch

Here is a 2-layer MLP that takes 784 pixel values (a flattened 28×28 image) and outputs 10 logits (one per digit).

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):           # x: (B, 784)
        h = torch.relu(self.fc1(x)) # h: (B, 256)
        return self.fc2(h)          # logits: (B, 10)

model = MLP()
fake_image = torch.randn(1, 784)
print(model(fake_image).shape)      # torch.Size([1, 10])
```

`nn.Linear(in, out)` is exactly `y = x @ W + b` with `W` shaped `(in, out)` and `b` shaped `(out,)`.

---

## 5. Loss: how wrong are we?

The network's job is to be right more often. To train it, we need a single number that says "you were this wrong this time." That number is the **loss**.

For classification with `C` classes, we use **cross-entropy loss**:

```python
loss = -log(p_correct_class)
```

If the model assigns 0.9 probability to the correct class, `-log(0.9) ≈ 0.10` (small loss, good).
If it assigns 0.01 probability, `-log(0.01) ≈ 4.6` (big loss, bad).

PyTorch's `nn.CrossEntropyLoss` takes raw **logits** (any real numbers) and the integer target class. It internally applies softmax + log. Don't softmax yourself before passing to it — common bug.

```python
loss_fn = nn.CrossEntropyLoss()
logits = model(fake_image)              # (B, 10)
target = torch.tensor([3])              # the true digit
loss = loss_fn(logits, target)          # scalar
```

For regression, **mean squared error** (`nn.MSELoss`) is the default.

---

## 6. Gradient descent: nudging the weights

The loss is a function of the weights. Calculus tells us the **gradient** — the direction in weight-space where the loss increases fastest. To make the loss go down, step in the **opposite** direction:

```
w_new = w_old - learning_rate * gradient_of_loss_wrt_w
```

That is **gradient descent**, the heart of training.

The **learning rate** (LR) decides how big a step you take. Too big and you overshoot and diverge. Too small and you take forever. Typical values: `1e-3` for Adam, `1e-1` for SGD on small problems.

### Stochastic gradient descent (SGD)

Computing the gradient on the whole training set every step is expensive. Instead we use a **mini-batch** (say 64 examples), compute the gradient on it, take a step, and repeat. That's **mini-batch SGD**, the default flavor.

### Better optimizers

Plain SGD bounces around. Smarter optimizers track recent gradients and adapt the step size per-parameter:

- **Momentum** — like a heavy ball rolling downhill, smooths out the bounces.
- **Adam** — adapts LR per parameter using the first and second moments of the gradient.
- **AdamW** — Adam with **decoupled weight decay**. **This is what almost every modern LLM trains with.**

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
```

---

## 7. Backpropagation: the chain rule, automated

How do we compute that gradient? The answer is **backpropagation** — fancy name, simple idea.

The forward pass builds a chain of operations: `loss = f3(f2(f1(x; W1); W2); W3)`. The chain rule says:

```
d(loss)/d(W1) = d(loss)/d(f3) * d(f3)/d(f2) * d(f2)/d(f1) * d(f1)/d(W1)
```

We multiply local gradients backward through the graph. PyTorch builds the graph automatically as you do the forward pass, then computes all gradients with one call to `loss.backward()`.

The full training step:

```python
optimizer.zero_grad()      # clear last step's gradients
logits = model(x)          # forward
loss = loss_fn(logits, y)  # compute loss
loss.backward()            # backward — fills .grad on every parameter
optimizer.step()           # use .grad to update parameters
```

**Common bug:** forgetting `zero_grad()` causes gradients to accumulate across steps. Sometimes you *want* this (gradient accumulation for large effective batch sizes), but only deliberately.

---

## 8. The full training loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# fake data
X = torch.randn(1000, 784)
Y = torch.randint(0, 10, (1000,))
loader = DataLoader(TensorDataset(X, Y), batch_size=64, shuffle=True)

model = MLP()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    print(f"epoch {epoch}, loss {loss.item():.4f}")
```

This pattern — `zero_grad → forward → loss → backward → step` — is the same whether you're training a 100-parameter MLP or a 100-billion-parameter LLM. The scaffolding never changes; only the model and data do.

---

## 9. Initialization: where you start matters

If all weights start at zero, every neuron does the same thing forever. So we initialize them randomly. But not just any random — bad init either explodes activations (NaN loss) or kills them (zero gradients).

Two go-to schemes:

- **Xavier / Glorot** (`nn.init.xavier_uniform_`): good for tanh / sigmoid layers. Keeps variance roughly the same across layers.
- **Kaiming / He** (`nn.init.kaiming_normal_`): good for ReLU and friends. PyTorch's default for `nn.Linear`.

In modern transformers, you also see **scaled init**: `std = 0.02` for embeddings and `std = 0.02 / sqrt(2 * num_layers)` for residual projections (GPT-2 style). The point is to keep the variance of activations stable as the network goes deeper.

---

## 10. Regularization: don't just memorize

A network with millions of parameters can memorize the training set perfectly while doing nothing for new data. That's **overfitting**. Three knobs help:

- **Weight decay** (a.k.a. L2 regularization): a small term `λ * ||W||²` added to the loss. AdamW does this cleanly. Typical: `0.1` for LLMs, `0.01` for vision.
- **Dropout**: at training time, randomly zero-out fraction `p` of activations. Forces the network not to rely on any single neuron. Typical `p = 0.1`. *Note: most modern LLMs use dropout = 0; their main regularizer is the sheer scale of the data.*
- **Early stopping**: watch validation loss; stop when it stops improving.

---

## 11. Common bugs and how to spot them

- **Loss is NaN**: usually exploding gradients. Lower LR, add gradient clipping (`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`), check for `log(0)` or `0/0` in custom code.
- **Loss doesn't go down at all**: LR too low, bug in data pipeline, frozen parameters, or you forgot `optimizer.zero_grad()`.
- **Loss goes down then up**: LR too high. Reduce by 10×.
- **Train loss goes down, val loss doesn't**: overfitting. More data, more regularization, or smaller model.
- **All outputs the same**: dead network. Check init. Check activation function (ReLU dying problem at very high LR).

---

## 12. From this chapter to language models

A transformer is, at heart, a giant stack of MLPs interleaved with **attention** (chapter 8). The attention layers move information between sequence positions; the MLPs do the per-position thinking. Everything you learned here — linear layers, activations, loss, backprop, AdamW — applies. Language modeling is just an extra-fancy classification problem where the classes are the next token.

So when you read later chapters, remember: **it's all the same machinery, just in clever arrangements.**

---

## Going deeper

- *Neural Networks and Deep Learning* by Michael Nielsen (free online book) — the gentlest derivation of backprop.
- Andrej Karpathy's `micrograd` — 200 lines of Python that implement autograd from scratch. Reading it is the single fastest way to truly *get* backprop.
- The PyTorch docs on `autograd` mechanics — once you've built micrograd, this clicks.

Next: **[02-pytorch.md](./02-pytorch.md)** — let's master the framework so the rest of the book reads like English.
