# 00 · The Essential Math (Read Before Anything Else)

> **TL;DR** You need: vectors, matrices, matrix multiplication, derivatives, gradients, the chain rule, log/exp/softmax, expectation, and basic probability. That's it. This file shows each one with intuition, a picture, and PyTorch code so you can poke it.

You **do not** need: real analysis, measure theory, manifold theory, or anything from a graduate textbook. If a paper uses heavy notation, 95% of the time it is dressing up these basics.

---

## 1. Vectors and matrices

A **vector** is an ordered list of numbers. We write `x ∈ ℝⁿ` for a vector with `n` entries.

A **matrix** is a 2D grid of numbers. `A ∈ ℝᵐˣⁿ` has `m` rows and `n` columns.

A **tensor** generalizes this: 3D, 4D, ... arrays. In code, "tensor" is the umbrella term.

```python
import torch
v = torch.tensor([1.0, 2.0, 3.0])         # shape (3,)
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) # shape (2, 2)
T = torch.randn(2, 3, 4)                   # shape (2, 3, 4)
```

### Useful vector ops

```python
# dot product (scalar)
torch.dot(v, v)                   # 1*1 + 2*2 + 3*3 = 14

# norm (length)
v.norm()                          # sqrt(14) ≈ 3.74

# cosine similarity (angle between vectors)
def cos_sim(a, b):
    return torch.dot(a, b) / (a.norm() * b.norm())
```

Cosine similarity is everywhere in embeddings: 1 means same direction, 0 means perpendicular, -1 means opposite. It's how RAG systems judge "similar text."

---

## 2. Matrix multiplication (the most important operation in deep learning)

If `A` is `(m, k)` and `B` is `(k, n)`, then `C = A @ B` is `(m, n)`, with:

```
C[i, j] = sum over p of A[i, p] * B[p, j]
```

In words: each output entry is a **dot product** of one row of `A` with one column of `B`.

Visually, for `(2, 3) @ (3, 2)`:

```
A           B           C
[a b c]   [g h]       [ag+bj+cl  ah+bk+cm]
[d e f] @ [j k]   =   [dg+ej+fl  dh+ek+fm]
          [l m]
```

```python
A = torch.tensor([[1., 2., 3.], [4., 5., 6.]])    # (2, 3)
B = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])  # (3, 2)
C = A @ B
print(C)
# tensor([[4., 5.],
#         [10., 11.]])
```

### Why it matters

A `nn.Linear(in_features, out_features)` layer is **literally** `y = x @ W + b` where `W` is `(in, out)`. **Most of an LLM's time is matrix multiplications.** When you see a billion-parameter model, ~95% of those parameters live in matmul weight matrices.

### Shape rules to internalize

For matmul to work, the **last dim of A must equal the second-to-last dim of B**:

```python
torch.randn(B, T, D)  @  torch.randn(D, M)        # → (B, T, M)
torch.randn(B, H, T, d)  @  torch.randn(B, H, d, T)  # → (B, H, T, T)
```

In transformers, you'll see batched matmul like the second one constantly.

### Batched matmul

PyTorch broadcasts matmul over leading dims. `(B, T, D) @ (D, M) → (B, T, M)` runs `B` matmuls of shape `(T, D) @ (D, M)`. With Tensor Cores, this is super fast as long as `T`, `D`, `M` are big enough (~64+).

---

## 3. Broadcasting

When shapes don't match exactly, PyTorch tries to **broadcast** smaller dims by repeating along size-1 axes. The rule: align shapes from the right; sizes must match or be 1.

```python
a = torch.randn(3, 4)    # (3, 4)
b = torch.randn(4)       # (4,) → broadcast to (1, 4) → (3, 4)
a + b                    # works, shape (3, 4)

a = torch.randn(3, 1)
b = torch.randn(1, 4)
a + b                    # both stretched, shape (3, 4)
```

This makes positional encodings, masking, and per-head ops elegant. It also bites you when you didn't *want* broadcasting:

```python
logits = torch.randn(B, T, V)
target = torch.randn(B, T)         # forgot last dim
diff = logits - target              # quietly broadcasts to (B, T, V) — bug
```

Always print shapes when in doubt.

---

## 4. The basic functions everyone uses

### `exp` and `log`

`exp(x) = eˣ` (always positive). `log(x)` is its inverse, defined for `x > 0`.

Logs convert multiplication to addition — that's why we work in log-probabilities. `log(p₁ * p₂) = log(p₁) + log(p₂)`. Sums are stable; products of small probabilities underflow.

### Softmax

Turn a vector of any real numbers (logits) into a probability distribution:

```
softmax(x)[i] = exp(x[i]) / sum_j exp(x[j])
```

```python
def softmax(x):
    e = torch.exp(x - x.max())   # subtract max for numerical stability
    return e / e.sum()

x = torch.tensor([2.0, 1.0, 0.1])
print(softmax(x))                # tensor([0.6590, 0.2424, 0.0986])
```

Properties:
- All outputs in (0, 1), sum to 1 → it's a probability distribution.
- Big input → big output (winner-take-most).
- The "subtract max" trick prevents `exp` from overflowing on big logits — every framework does this.

### Cross-entropy

Given a model's predicted distribution `p` and the true class `y`:

```
loss = -log(p[y])
```

Equivalently, for one-hot `y`:

```
loss = -sum_i y[i] * log(p[i])
```

Cross-entropy in PyTorch eats raw logits and applies log-softmax internally:

```python
loss = torch.nn.functional.cross_entropy(logits, target)
```

This is the **only loss** used to train language models.

### Sigmoid and logit

```
sigmoid(x) = 1 / (1 + exp(-x))
logit(p)   = log(p / (1-p))
```

Sigmoid is softmax for two classes. Useful in binary classification, gating in MoE, and the SiLU activation (`x * sigmoid(x)`).

---

## 5. Derivatives in 5 minutes

The **derivative** of `f(x)` at point `x` is "how fast does `f` change when you nudge `x`?" Geometrically, the slope of the tangent line.

Five rules cover everything:

| Function | Derivative |
|----------|------------|
| `c` (constant) | `0` |
| `x` | `1` |
| `xⁿ` | `n * xⁿ⁻¹` |
| `eˣ` | `eˣ` |
| `log(x)` | `1/x` |

Plus three combination rules:

- **Sum**: `(f + g)' = f' + g'`
- **Product**: `(fg)' = f'g + fg'`
- **Chain**: `(f(g(x)))' = f'(g(x)) * g'(x)`  ← **this one is the whole game**

The chain rule says: to differentiate a function-of-a-function, multiply the outer derivative by the inner derivative. Backprop is *just the chain rule applied through a giant computational graph.* Once it clicks, every "magic" of training feels obvious.

### Worked example

```
f(x) = log(1 + exp(x))     (this is "softplus")
g(x) = exp(x)              inner part — derivative exp(x)
h(u) = log(1 + u)          outer part with u = g(x) — derivative 1/(1+u)
f'(x) = 1/(1 + exp(x)) * exp(x) = exp(x) / (1 + exp(x)) = sigmoid(x)
```

Cute — the derivative of softplus is sigmoid. PyTorch knows this without you writing the formula.

---

## 6. Gradients (multivariable derivatives)

For a function `f(x₁, ..., xₙ)` of many inputs, the **gradient** is the vector of partial derivatives:

```
∇f = [ ∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ ]
```

It points in the direction of fastest **increase** of `f`. To minimize `f`, walk in the direction `-∇f`. That's gradient descent:

```
x_new = x_old - η * ∇f(x_old)
```

`η` (eta) is the **learning rate** — how big a step you take.

### One-line intuition

Loss is a hilly landscape. The gradient is a compass that always points uphill. Subtract it (with a small step) and you walk downhill. Reach the bottom = trained.

### PyTorch does this for you

You don't compute gradients by hand. You build the forward pass, call `.backward()`, and PyTorch fills `.grad` on every `requires_grad=True` tensor.

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()       # y = 1 + 4 + 9 = 14
y.backward()
print(x.grad)            # tensor([2., 4., 6.])  i.e. 2x
```

---

## 7. The Jacobian (just so the word doesn't scare you)

If `f: ℝⁿ → ℝᵐ`, its **Jacobian** is the `m × n` matrix of all partial derivatives. For a single-output `f`, the Jacobian is just the gradient (a row vector). For matmul, the Jacobian is huge — but you never materialize it. Backprop multiplies Jacobians implicitly via vector-Jacobian products.

You can usually ignore this word. When papers say "vector-Jacobian product (VJP)," they mean "one step of backprop." That's it.

---

## 8. Probability essentials

### Expectation

`E[X]` is the average value of random variable `X`. For a discrete distribution, `E[X] = sum_i p_i * x_i`.

In ML, almost every loss is an expectation: "average loss over the data." We approximate it with a mini-batch — the **stochastic** in stochastic gradient descent.

### KL divergence

A measure of how different two distributions are:

```
KL(P || Q) = sum_x P(x) * log(P(x) / Q(x))
```

It's 0 when `P == Q` and grows as they diverge. Cross-entropy is `-E_P[log Q]` — minimizing it is the same as minimizing `KL(P || Q)` plus a constant. So **cross-entropy training is making the model's distribution match the data's.**

### Sampling vs argmax

Given a probability distribution over the next token, you can:
- **Argmax**: pick the most likely token (greedy decoding).
- **Sample**: roll a weighted die.
- **Top-k**: sample from the top `k` only.
- **Top-p / nucleus**: sample from the smallest set whose probability sums to `p`.
- **Temperature**: divide logits by `T` before softmax. Low `T` → sharp, deterministic. High `T` → flat, creative.

```python
def sample(logits, temperature=1.0, top_p=0.9):
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, idx = probs.sort(descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    keep = cumulative <= top_p
    keep[..., 0] = True  # always keep the top one
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum()
    sampled = torch.multinomial(sorted_probs, 1)
    return idx.gather(-1, sampled)
```

---

## 9. Tensor anatomy: the shapes you will see daily

In LLM code, you'll keep seeing dims with these conventional letters:

| Letter | Meaning | Typical size |
|--------|---------|--------------|
| `B` | batch | 1-1024 |
| `T` | time / sequence length | 1-128k |
| `D` (or `d_model`) | model dimension | 768-16k |
| `H` (or `n_heads`) | attention heads | 8-128 |
| `d` (or `head_dim`) | per-head dim | 64-128 |
| `V` | vocabulary size | 32k-256k |
| `F` (or `d_ff`) | feed-forward inner dim | 2-8 × D |

A typical attention input is `(B, T, D)`. After splitting into heads it's `(B, H, T, d)` with `D = H * d`. The shape gymnastics you'll do are reshape/permute calls between these forms.

```python
B, T, D, H = 2, 5, 128, 8
d = D // H

x = torch.randn(B, T, D)
x_heads = x.view(B, T, H, d).transpose(1, 2)   # (B, H, T, d)
```

---

## 10. Numerical stability tricks worth knowing

These are not "nice to haves" — without them, training silently goes to NaN.

- **Subtract max before softmax**: `softmax(x) = softmax(x - x.max())`. Same result, no overflow.
- **Use log-sum-exp**: `log(sum(exp(x))) = m + log(sum(exp(x - m)))` where `m = x.max()`.
- **Mixed precision** stores activations in bf16 but accumulates reductions in fp32 (PyTorch handles this for you).
- **Clip gradients**: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` keeps the rare bad batch from blowing up.
- **Avoid `log(0)`**: clamp with `torch.log(p.clamp(min=1e-10))` or work in log-space throughout.

---

## 11. A cheat sheet you can re-read in 60 seconds

- `Linear` = `y = x @ W + b`. Most of an LLM is this.
- `softmax` turns logits into a probability distribution.
- `cross_entropy` = `-log` of the probability you assigned to the right answer.
- `gradient` = vector of partial derivatives, points uphill, subtract to go down.
- `chain rule` is the soul of backprop.
- `bf16` for training, `int4/int8` for inference, `fp32` for sensitive bits (loss, optimizer state).
- Print shapes constantly when you're new.
- KL divergence and cross-entropy are basically the same training objective.

That's it. The rest of this guide builds on this. If a future chapter loses you, come back here.

Next: **[01-neural-networks.md](./01-neural-networks.md)**.
