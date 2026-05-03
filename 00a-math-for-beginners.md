# 00a · AI Math Explained Simply (for High Schoolers)

> **Read this first if `00-essential-math.md` felt scary.**
> Everything here uses only what you already know from school: numbers, lists, basic algebra, and a tiny bit of calculus (we'll teach what you need).

---

## Why bother with the math?

A neural network is **just math**: addition, multiplication, exponents, and logs, repeated billions of times. There is no "magic". If you understand the math, AI stops feeling mysterious.

Most people skip the math, copy-paste a tutorial, and then get stuck the moment something breaks. Don't be that person. Spend a few hours here. It pays back forever.

The good news: **you only need 7 ideas**.

1. Vectors, matrices, tensors (lists and grids of numbers)
2. Matrix multiplication
3. Broadcasting (how PyTorch handles shape mismatches)
4. The functions `exp`, `log`, `softmax`
5. Loss (how wrong is the model?)
6. Derivatives and gradients (how to make the model less wrong)
7. The chain rule (how to combine derivatives through many steps)

That's it. Master these and every AI paper becomes readable.

---

## Section 1: Numbers, lists, and grids

### A scalar is a single number

Like `5`, `-2.7`, or `3.14`. One value. One thought.

In AI we sometimes call these "scalars" to distinguish them from lists.

### A vector is a list of numbers

A **vector** is just an ordered list of numbers, like:

```
[3, 5, 2]
```

That's a vector of length 3.

**Real-world example.** Your grades for 4 subjects:

```
math = 92, english = 78, science = 85, art = 90
```

We can write this as a vector:

```
grades = [92, 78, 85, 90]
```

Length 4. Each position represents a subject.

**AI example.** When a model sees the word "cat", it converts it into a vector. A small model might use **768 numbers**; a frontier model like Claude or Llama uses **4,096 to 16,384**:

```
cat = [0.12, -0.45, 1.07, ..., 0.33]   # hundreds to thousands of numbers
```

Why so many? Because we need lots of dimensions to capture meaning ("animal", "furry", "small", "pet", "predator", and many more — all spread across these numbers). The model learns them during training; we don't pick them by hand.

### Things you can do with vectors

**Addition** — line them up and add by position:

```
[1, 2, 3] + [4, 5, 6] = [1+4, 2+5, 3+6] = [5, 7, 9]
```

**Multiplying by a number** (scaling):

```
3 * [2, 4, 6] = [6, 12, 18]
```

**Length** of a vector (the Pythagorean theorem!). For `[3, 4]`:

```
length = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
```

For longer vectors, same idea: square each number, add them up, take the square root.

**Dot product** — multiply position-by-position, then add:

```
[1, 2, 3] · [4, 5, 6] = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

The dot product tells you how "aligned" two vectors are.
- Big positive number = pointing the same way
- Zero = perpendicular (no relation)
- Big negative number = pointing opposite ways

**Cosine similarity** — dot product divided by `(length of A × length of B)`. The result is always between `-1` and `+1`:

- `+1` = same direction (very similar)
- `0` = perpendicular (unrelated)
- `-1` = opposite directions

This is how chatbots find "similar texts" or "similar images" — they convert each one to a vector, then compare with cosine similarity.

### A matrix is a grid of numbers

A **matrix** is a 2D arrangement of numbers — rows and columns, like a spreadsheet.

**Example.** 3 students × 4 subjects:

```
              Math  English  Science  Art
Alice    [    92      78       85      90  ]
Bob      [    65      82       71      88  ]
Carol    [    88      90       75      80  ]
```

This is a **3 × 4** matrix. We say its "shape" is `(3, 4)`. 3 rows, 4 columns.

In AI, all the *learned knowledge* of a model lives in matrices called **weights**. When people say "Llama 3 has 405 billion parameters", they mean the weights of all its matrices contain 405 billion numbers total.

### A tensor: just stacks (don't be scared by the word)

A **tensor** is the umbrella word. It just means "multi-dimensional array of numbers."

| Shape | What it is | Example |
|-------|-----------|---------|
| `()` | scalar | `5` |
| `(5,)` | vector | `[1, 2, 3, 4, 5]` |
| `(3, 4)` | matrix | a 3×4 grid |
| `(2, 3, 4)` | 3-D tensor | 2 stacks of 3×4 grids |
| `(32, 3, 224, 224)` | 4-D tensor | 32 photos, 3 colors, 224×224 pixels |

A color photo is a 3-D tensor. In PyTorch the convention is **channels first**: `3 colors × height × width`, so a single 224×224 image is shape `(3, 224, 224)`. A batch of 32 such photos is `(32, 3, 224, 224)`.

So when someone says "tensor of shape (8, 16, 768)", they just mean a 3D box of numbers with those dimensions. Nothing more.

### In code (PyTorch)

```python
import torch

# scalar
s = torch.tensor(5.0)                     # shape ()

# vector
v = torch.tensor([1.0, 2.0, 3.0])         # shape (3,)

# matrix
M = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])            # shape (2, 2)

# 3D tensor (random numbers)
T = torch.randn(2, 3, 4)                  # shape (2, 3, 4)

print(T.shape)
```

When confused about dimensions, **always print the shape**. It's the #1 debugging trick.

---

## Section 2: Matrix multiplication (the most important operation in AI)

About **95% of the time** an LLM spends running is doing matrix multiplications. So we have to understand it.

### The shape rule

If matrix `A` has shape `(m, k)` and matrix `B` has shape `(k, n)`, then `A × B` (we write `A @ B` in Python) has shape `(m, n)`.

The **inner numbers must match**:

```
(2, 3) @ (3, 4) → (2, 4)   ✓  (3 matches 3, drop them, keep outer 2 and 4)
(2, 3) @ (4, 5) → ERROR    ✗  (3 ≠ 4)
```

### How each entry is computed

Each entry of the result is a **dot product** of one row of A with one column of B:

```
C[i, j] = (i-th row of A)  ·  (j-th column of B)
```

That's the whole story. Matrix multiplication = a grid of dot products.

### Worked example (do this by hand once)

```
A = [1  2  3]       shape (2, 3)
    [4  5  6]

B = [1  0]
    [0  1]          shape (3, 2)
    [1  1]
```

Result shape: `(2, 2)`. Let's compute each entry.

```
C[0,0] = row 0 of A · col 0 of B = 1·1 + 2·0 + 3·1 = 1 + 0 + 3 = 4
C[0,1] = row 0 of A · col 1 of B = 1·0 + 2·1 + 3·1 = 0 + 2 + 3 = 5
C[1,0] = row 1 of A · col 0 of B = 4·1 + 5·0 + 6·1 = 4 + 0 + 6 = 10
C[1,1] = row 1 of A · col 1 of B = 4·0 + 5·1 + 6·1 = 0 + 5 + 6 = 11
```

So:

```
C = [4    5]
    [10  11]
```

In code:

```python
A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
B = torch.tensor([[1., 0.],
                  [0., 1.],
                  [1., 1.]])
print(A @ B)
# tensor([[ 4.,  5.],
#         [10., 11.]])
```

### Why does this matter?

A neural network "layer" is **literally** this:

```
output = input @ W + b
```

- `input` is a vector (or batch of vectors)
- `W` is a learned weight matrix
- `b` is a learned bias vector
- `output` is the layer's response

Stack many such layers and you have a deep neural network. Stack a *very* specific arrangement of them and you have a transformer (the architecture behind GPT, Claude, Gemini, etc.).

When we say "training a model", we mean: **adjust the numbers in `W` and `b` until the outputs match what we want.**

---

## Section 3: Broadcasting (when shapes don't quite match)

Sometimes you want to add a vector to every row of a matrix. PyTorch handles this automatically by "broadcasting" — pretending the smaller shape is repeated.

**Example:**

```python
A = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])      # shape (2, 3)
b = torch.tensor([10, 20, 30])     # shape (3,)
print(A + b)
# [[11, 22, 33],
#  [14, 25, 36]]
```

PyTorch added `[10, 20, 30]` to each row of `A`.

**Rules of broadcasting:**

1. Line up the shapes from the **right**.
2. For each pair of dimensions, the sizes must match OR one must be `1`.

This is great for clean code but can also cause **silent bugs** — PyTorch won't complain, but the result is wrong:

```python
loss = torch.randn(4)       # shape (4,)   — one loss per sample
weight = torch.randn(4, 1)  # shape (4, 1) — extra dim by mistake
result = loss * weight       # shape (4, 4) — not (4,)! Silent outer product.
```

PyTorch treated `(4,)` as `(1, 4)` and broadcast with `(4, 1)` to get `(4, 4)`. No error, just wrong.

**Habit: print shapes constantly when you're new.**

---

## Section 4: `exp`, `log`, and `softmax`

### `exp(x) = eˣ`

`e` is a special number ≈ `2.71828`. So:

- `exp(0) = 1`
- `exp(1) ≈ 2.718`
- `exp(2) ≈ 7.39`
- `exp(-2) ≈ 0.135`

**Properties:** always positive, grows fast, never zero.

### `log(x)`

`log` is the **inverse** of `exp`. So if `exp(2) = 7.39`, then `log(7.39) = 2`.

- `log(1) = 0`
- `log(e) = 1`
- `log(10) ≈ 2.30`

Only defined for `x > 0` (you can't log a zero or a negative).

**The magic property:** `log` turns multiplication into addition.

```
log(a × b) = log(a) + log(b)
```

Why we love this in AI: probabilities are tiny numbers, and multiplying many tiny numbers makes them so small the computer thinks they're zero (called "underflow"). If we work in log-space and **add** instead of multiply, no underflow.

### Softmax: turn raw scores into probabilities

A neural network's last layer outputs raw scores called **logits**. Logits can be any real number — positive, negative, big, small.

But we usually want **probabilities**: numbers between 0 and 1 that add up to 1.

Softmax does exactly that:

```
softmax(x)[i] = exp(x[i]) / sum_of_exp_of_all_x
```

**Worked example.** Suppose the model outputs logits `[2, 1, 0.1]` for three options.

Step 1: Take `exp` of each.
- `exp(2)  = 7.39`
- `exp(1)  = 2.72`
- `exp(0.1) = 1.11`

Step 2: Sum them up.
- `total = 7.39 + 2.72 + 1.11 = 11.22`

Step 3: Divide each by the sum.
- `7.39 / 11.22 ≈ 0.659`
- `2.72 / 11.22 ≈ 0.243`
- `1.11 / 11.22 ≈ 0.099`

Result: `[0.659, 0.243, 0.099]`. They add up to 1. We can read this as: **"the model is 65.9% sure of option A, 24.3% sure of option B, 9.9% sure of option C."**

```python
def softmax(x):
    e = torch.exp(x - x.max())   # subtract max for numerical safety
    return e / e.sum()

x = torch.tensor([2.0, 1.0, 0.1])
print(softmax(x))   # tensor([0.6590, 0.2424, 0.0986])
```

**Why subtract the max?** Because `exp(1000)` is an astronomical number that overflows your computer. By subtracting the max first, all numbers become 0 or negative, and `exp(negative)` is small and safe. The result is mathematically identical:

```
exp(x) / Σ exp(x)  =  exp(x - m) / Σ exp(x - m)
```

(Both `exp(m)` factors cancel.) Every framework does this trick.

### Sigmoid (a special case of softmax)

For just two choices, we use **sigmoid**:

```
sigmoid(x) = 1 / (1 + exp(-x))
```

It squishes any real number into the range `(0, 1)`:

- `sigmoid(0) = 0.5`
- `sigmoid(2) ≈ 0.88`
- `sigmoid(-2) ≈ 0.12`
- `sigmoid(10) ≈ 1.0`
- `sigmoid(-10) ≈ 0.0`

It looks like an S-shaped curve. Used for binary "yes/no" predictions, and for "gates" inside modern architectures that decide how much information passes through.

---

## Section 5: Loss — how wrong is the model?

We need a number that says "how bad is this prediction?" Lower = better. We call this number the **loss**.

For language models (and almost all classification), we use **cross-entropy loss**.

### The simple version

If the model said the right answer has probability `p`, then:

```
loss = -log(p)
```

That's it. One formula.

Examples:

| Model's probability of the correct answer | Loss |
|---|---|
| `1.00` (perfect) | `-log(1.00) = 0`        — no penalty |
| `0.90` | `-log(0.90) ≈ 0.105`              — small penalty |
| `0.50` (50/50 guess) | `-log(0.50) ≈ 0.693` — moderate penalty |
| `0.10` | `-log(0.10) ≈ 2.30`               — big penalty |
| `0.01` | `-log(0.01) ≈ 4.61`               — huge penalty |
| `0.0001` | `-log(0.0001) ≈ 9.21`           — massive penalty |

So the loss is **small when the model is confident and right**, and **huge when it's confident and wrong**. That's exactly what we want.

### In code

```python
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]])    # shape (1, 3) — 3 classes
target = torch.tensor([0])                   # the correct class is index 0

loss = F.cross_entropy(logits, target)
print(loss.item())   # ≈ 0.417
```

`cross_entropy` takes raw **logits** (not probabilities) and applies softmax + log internally. It's faster and more numerically stable that way.

This is the **core loss function** for LLM pretraining. Fine-tuning stages (RLHF, DPO) layer additional objectives on top, but cross-entropy is the foundation.

---

## Section 6: Derivatives — the rate of change

A **derivative** answers this question:
> *"If I nudge `x` by a tiny amount, how much does `f(x)` change?"*

Geometrically, the derivative is the **slope of the curve** at that point.

### The 5 rules you need

| Function | Derivative | In words |
|----------|-----------|----------|
| `f(x) = c` (any constant) | `0` | constants don't change |
| `f(x) = x` | `1` | x changes 1-for-1 |
| `f(x) = xⁿ` | `n × xⁿ⁻¹` | the "power rule" |
| `f(x) = eˣ` | `eˣ` | exp's derivative is itself! |
| `f(x) = log(x)` | `1/x` | |

Plus 3 rules for combining functions:

- **Sum rule:** derivative of `(f + g)` is `f' + g'`
- **Product rule:** derivative of `(f × g)` is `f' × g + f × g'`
- **Chain rule:** derivative of `f(g(x))` is `f'(g(x)) × g'(x)` ← **the most important rule in deep learning**

### Quick worked example

```
f(x) = 3x² + 5x + 7
```

Apply the rules:
- Derivative of `3x²` = `3 × 2x = 6x` (power rule × constant multiple)
- Derivative of `5x` = `5`
- Derivative of `7` = `0` (constant)

Add them up: `f'(x) = 6x + 5`.

If `x = 2`, then `f'(2) = 12 + 5 = 17`. So at `x = 2`, the curve has slope 17. If we nudge `x` by `0.001`, then `f(x)` goes up by approximately `17 × 0.001 = 0.017`.

### The chain rule (the soul of backpropagation)

Suppose you have a nested function:

```
f(x) = sin(x²)
```

The **outer** function is `sin`, and the **inner** function is `x²`.

Chain rule:

```
f'(x) = (derivative of outer)  ×  (derivative of inner)
      = cos(x²)                ×  2x
```

You differentiate from the outside in, multiplying as you go.

A neural network is a deep nesting of functions:

```
loss = cross_entropy(softmax(linear(linear(linear(input)))))
```

To find how the loss changes when we nudge any weight, we apply the chain rule layer by layer, from the loss back toward the input. **That's all backpropagation is** — chain rule applied through a giant graph.

PyTorch does this automatically. You never compute it by hand.

---

## Section 7: Gradients — derivatives for many inputs

Most functions in AI take **lots** of inputs. A neural network might take 768 numbers in.

The **gradient** is just the list of partial derivatives — one per input.

**Example:**

```
f(x₁, x₂, x₃) = x₁² + 3x₂ + x₃
```

Compute the partial derivative with respect to each variable (treat the others as constants):

- `∂f/∂x₁ = 2x₁`
- `∂f/∂x₂ = 3`
- `∂f/∂x₃ = 1`

So the gradient is `[2x₁, 3, 1]`. It's just a vector of derivatives.

### The gradient points uphill

Geometrically, the gradient at a point `x` points in the direction where `f` grows the **fastest**.

Imagine you're standing on a hilly landscape (the loss surface). The gradient is a compass that always points uphill. If we want the loss to **decrease**, we walk in the **opposite** direction — downhill.

### Gradient descent — how AI learns

Update rule:

```
new_x = old_x  -  learning_rate × gradient
```

The **learning rate** (often called `η` or `lr`) is a small number like `0.001`. Take many small steps downhill, and you reach a low point. That's training.

### PyTorch does this for you

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()       # y = 1 + 4 + 9 = 14
y.backward()             # PyTorch fills in the gradients
print(x.grad)            # tensor([2., 4., 6.])
```

`requires_grad=True` says "track gradients for this tensor". Then `.backward()` computes them, and you can read them off `x.grad`.

What just happened? `y = x₁² + x₂² + x₃²`, so the gradient is `[2x₁, 2x₂, 2x₃] = [2, 4, 6]` at `x = [1, 2, 3]`. Math checks out.

---

## Section 8: Probability essentials

### What's a probability?

A number between 0 and 1.
- `0` = impossible
- `1` = certain
- `0.5` = 50/50

A **distribution** is a list of probabilities for different outcomes; they must add up to 1.

Examples:
- Fair coin: `[P(heads)=0.5, P(tails)=0.5]`
- Weighted die that lands on 6 a lot: `[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]`

### Expectation = the average outcome

If a random variable `X` takes values `v_i` with probabilities `p_i`, then the expected value is:

```
E[X] = Σ p_i × v_i
```

For a fair die: `E[X] = (1/6)(1+2+3+4+5+6) = 21/6 = 3.5`. On average, you roll 3.5.

In AI, the loss we minimize is the **expected** loss over all training data. We approximate it by averaging over a small batch (say, 32 examples). That's why it's called **stochastic** gradient descent — each batch is a random sample.

### Sampling from a distribution

Given probabilities, we can pick (sample). For `[0.5, 0.3, 0.2]`:
- 50% of the time we pick option 0
- 30% of the time we pick option 1
- 20% of the time we pick option 2

In language models, after softmax we get a distribution over the next word. Strategies for picking:

- **Argmax (greedy)** — pick the highest probability every time. Boring, repetitive.
- **Sample** — pick at random according to probabilities. More creative.
- **Top-k** — only consider the `k` most likely options, sample from those.
- **Top-p (nucleus)** — keep only the smallest set of options whose probabilities sum to `p`, sample from those.
- **Min-p** — keep only tokens whose probability is at least `p × probability_of_top_token`. Adapts automatically to the model's confidence — standard in inference engines since 2025.
- **Temperature** — divide all logits by `T` before softmax.
  - `T < 1` sharpens (more deterministic)
  - `T > 1` flattens (more random)

```python
def sample(logits, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1)
```

### KL divergence (just so you've seen the word)

KL divergence measures **how different** two distributions are:

```
KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
```

It's `0` when `P == Q`, and grows as they diverge.

**Cool fact:** minimizing cross-entropy loss is *the same* as minimizing the KL divergence between the model's distribution and the data's distribution (up to a constant). So **training is making the model match the data's distribution**.

---

## Section 9: The shapes you'll see in real LLM code

These letters appear **everywhere** in transformer code:

| Letter | Means | Typical size |
|--------|-------|--------------|
| `B` | batch (how many examples at once) | 1 to 1024 |
| `T` | time = sequence length (number of tokens) | 1 to 1M+ |
| `D` (or `d_model`) | model dimension | 768 to 16k |
| `H` (or `n_heads`) | number of attention heads | 8 to 128 |
| `d` (or `head_dim`) | dimension per head | 64 to 128 |
| `V` | vocabulary size | 32k to 256k |
| `F` (or `d_ff`) | feed-forward inner dim | 2-8 × D |

Note: `D = H × d`. The model dimension splits across heads.

A typical attention input has shape `(B, T, D)`. After splitting into heads it becomes `(B, H, T, d)`. Most "shape gymnastics" is just reshaping between these forms.

```python
B, T, D, H = 2, 5, 128, 8
d = D // H

x = torch.randn(B, T, D)                       # (2, 5, 128)
x_heads = x.view(B, T, H, d).transpose(1, 2)   # (2, 8, 5, 16)
```

Don't memorize this — keep printing shapes until it sinks in.

---

## Section 10: Numerical tricks (so training doesn't blow up)

These small details prevent **NaN** (not-a-number) errors during training. Without them, training silently breaks.

1. **Subtract max before softmax.** Same answer, no overflow.
2. **Use log-sum-exp** instead of summing exp:
   ```
   log(Σ exp(x))  =  m + log(Σ exp(x − m))
   ```
   where `m = max(x)`. Numerically stable.
3. **Mixed precision.** Store activations in `bf16` (16-bit floats), but accumulate sums in `fp32` (32-bit). PyTorch handles this for you with `autocast`.
4. **Clip gradients.** Limit how big a gradient can be, so one bad batch doesn't blow training up:
   ```python
   torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
   ```
5. **Avoid log(0).** Add a tiny epsilon: `log(p + 1e-10)`, or better, work in log-space throughout.

You can usually ignore these until something breaks. When something breaks, this list is where you start looking.

---

## Section 11: One-page cheat sheet

- A **vector** is a list of numbers.
- A **matrix** is a 2D grid of numbers.
- A **tensor** is just a multi-D array of numbers.
- The most common operation is `output = input @ W + b`. Most of an LLM is exactly this, repeated many times.
- **Softmax** turns raw scores (logits) into probabilities that sum to 1.
- **Cross-entropy** = `-log(p_correct)` = the loss. Small when right, huge when confidently wrong.
- A **derivative** is the slope: how does output change when input nudges?
- A **gradient** is one derivative per input dimension. It points uphill — we move downhill.
- The **chain rule** is how to differentiate nested functions. Backprop is just the chain rule applied through a whole network.
- **KL divergence** and **cross-entropy** are basically twins. Training matches the model's distribution to the data's.

That's the entire math foundation of deep learning. Everything else is plumbing on top.

---

## Section 12: Run all of this on Google Colab (free, no install)

You don't need to install Python or PyTorch on your computer. **Google Colab** gives you a free Python notebook in the cloud, with PyTorch pre-installed and even a free GPU.

### Step-by-step setup

1. Open your browser and go to **https://colab.research.google.com**.
2. Sign in with any Google account.
3. Click **"New notebook"**. You'll see an empty cell with a blinking cursor.
4. Type or paste code into the cell. For example:
   ```python
   import torch
   v = torch.tensor([1.0, 2.0, 3.0])
   print(v)
   print(v.shape)
   ```
5. Run the cell — click the **▶** button to its left, or press `Shift + Enter`.
6. Add a new cell with the **+ Code** button at the top. Try matrix multiplication:
   ```python
   A = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
   B = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
   print(A @ B)
   ```

### Optional: turn on a free GPU

You don't need a GPU for the small examples in this guide, but if you want to train anything bigger:

1. Menu: **Runtime → Change runtime type**.
2. Hardware accelerator: choose **T4 GPU** (free).
3. Click **Save**.

Verify it works:

```python
import torch
print(torch.cuda.is_available())          # should print True
print(torch.cuda.get_device_name(0))      # e.g. 'Tesla T4'
```

### Try this exercise (uses everything from this guide)

Paste this whole block into a Colab cell and run it. Read each line — every concept is something we covered.

```python
import torch

# A simple "neural network" with 1 layer:
# input is 3 numbers, output is 2 numbers.
W = torch.randn(3, 2, requires_grad=True)   # weight matrix
b = torch.randn(2, requires_grad=True)      # bias vector
x = torch.tensor([1.0, 2.0, 3.0])           # input vector

# Forward pass: y = x @ W + b   (Section 2)
y = x @ W + b

# Compute loss: how far is y from the target?
target = torch.tensor([1.0, 0.0])
loss = ((y - target) ** 2).sum()            # squared error

# Backward pass: PyTorch computes gradients (Section 7)
loss.backward()

print("y     =", y)
print("loss  =", loss.item())
print("dL/dW =", W.grad)        # gradient w.r.t. W
print("dL/db =", b.grad)        # gradient w.r.t. b
```

What's happening:
- We made up a 3-input, 2-output "layer".
- We did one forward pass to get `y`.
- We compared `y` to a target and computed a loss.
- `loss.backward()` ran the chain rule and filled `.grad` on `W` and `b`.

Now you've literally executed every concept from this guide in 10 lines of code.

### Tips for using Colab

- The free GPU has limits (a few hours per day). Plenty for learning.
- Notebooks auto-save to your Google Drive.
- Reset everything if things get weird: **Runtime → Restart runtime**.
- Use **+ Text** buttons to add notes between code cells. Great for taking notes alongside the code.

---

## Where to go next

Now that this clicks, the compact version in **[00-essential-math.md](./00-essential-math.md)** will read fast — it's the same content in a tighter form.

Then move on to **[01-neural-networks.md](./01-neural-networks.md)** to actually build one.

If a future chapter ever loses you, **come back here**. The math is the ground floor. Don't skip past it twice.
