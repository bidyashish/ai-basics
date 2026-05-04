# 00 · ज़रूरी Math (कुछ भी पढ़ने से पहले ये पढ़ो)

> **TL;DR** आपको चाहिए: vectors, matrices, matrix multiplication, derivatives, gradients, chain rule, log/exp/softmax, expectation, और basic probability. बस इतना। ये file हर एक को intuition, picture, और PyTorch code के साथ दिखाती है ताकि आप experiment कर सको।

आपको **नहीं चाहिए**: real analysis, measure theory, manifold theory, या किसी graduate textbook का कुछ भी। अगर कोई paper heavy notation use करता है, तो 95% chance है कि वो basics को fancy तरीके से बता रहा है।

---

## 1. Vectors और Matrices

**Vector** एक ordered list है numbers की। हम लिखते हैं `x ∈ ℝⁿ` एक vector के लिए जिसमें `n` entries हैं।

**Matrix** numbers का 2D grid है। `A ∈ ℝᵐˣⁿ` में `m` rows और `n` columns हैं।

**Tensor** इसको generalize करता है: 3D, 4D, ... arrays. कोड में, "tensor" umbrella term है।

```python
import torch
v = torch.tensor([1.0, 2.0, 3.0])         # shape (3,)
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) # shape (2, 2)
T = torch.randn(2, 3, 4)                   # shape (2, 3, 4)
```

### Useful vector operations

```python
# dot product (scalar निकलता है)
torch.dot(v, v)                   # 1*1 + 2*2 + 3*3 = 14

# norm (length / लंबाई)
torch.linalg.norm(v)              # sqrt(14) ≈ 3.74

# cosine similarity (दो vectors के बीच का angle)
def cos_sim(a, b):
    return torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))
```

Cosine similarity embeddings में हर जगह है: 1 का मतलब same direction, 0 का मतलब perpendicular, -1 का मतलब opposite। RAG systems इसी से judge करते हैं कि "similar text" क्या है।

---

## 2. Matrix Multiplication (Deep Learning का सबसे important operation)

अगर `A` shape `(m, k)` का है और `B` shape `(k, n)` का है, तो `C = A @ B` shape `(m, n)` का होगा, जहां:

```
C[i, j] = sum over p of A[i, p] * B[p, j]
```

आसान शब्दों में: हर output entry `A` की एक row और `B` के एक column का **dot product** है।

Visually, `(2, 3) @ (3, 2)` के लिए:

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

### ये क्यों matter करता है

`nn.Linear(in_features, out_features)` layer **literally** `y = x @ W + b` है, जहां `W` shape `(in, out)` का है। **LLM का ज़्यादातर time matrix multiplications में जाता है।** जब आप billion-parameter model देखते हैं, तो ~95% parameters matmul weight matrices में रहते हैं।

### Shape rules जो आपको internalize करने चाहिए

Matmul के लिए, **A का last dim, B के second-to-last dim के equal होना चाहिए**:

```python
torch.randn(B, T, D)  @  torch.randn(D, M)        # → (B, T, M)
torch.randn(B, H, T, d)  @  torch.randn(B, H, d, T)  # → (B, H, T, T)
```

Transformers में, दूसरा वाला batched matmul हर जगह दिखेगा।

### Batched matmul

PyTorch matmul को leading dims पर broadcast करता है। `(B, T, D) @ (D, M) → (B, T, M)` `B` matmuls run करता है shape `(T, D) @ (D, M)` के। Tensor Cores के साथ, ये super fast है जब तक `T`, `D`, `M` काफी बड़े हों (~64+)।

### Transpose

**Transpose** rows और columns को flip करता है: shape `(m, n)` बन जाती है `(n, m)`।

```python
A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])   # (2, 3)
print(A.T)                         # (3, 2)
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])
```

Transformer code में `.T`, `.transpose(1, 2)`, और `.permute(...)` हर जगह दिखेंगे। Attention की key line `scores = Q @ K.T` एक transposed matrix के साथ matmul है — ये हर query और हर key के बीच dot products compute करती है।

### `*` vs `@` — common trap

`A * B` **element-wise** multiplication है (same position × same position, shapes match या broadcast होनी चाहिए)। `A @ B` **matrix multiplication** है (inner dims match होने चाहिए, dot products निकलते हैं)।

```python
A = torch.tensor([[1., 2.], [3., 4.]])
B = torch.tensor([[5., 6.], [7., 8.]])
print(A * B)   # element-wise: [[5, 12], [21, 32]]
print(A @ B)   # matmul:       [[19, 22], [43, 50]]
```

Transformer code में जब `*` दिखे, तो usually scaling है (जैसे `attn_weights * mask`)। जब `@` दिखे, तो core linear algebra।

### `einsum` — matmul को compact notation में लिखना

कई transformer implementations `torch.einsum` use करती हैं multi-dimensional matmuls को एक line में express करने के लिए:

```python
C = A @ B                                      # standard
C = torch.einsum('ik,kj->ij', A, B)            # same thing, explicit indices

# einsum shines for batched/multi-dim ops
attn = torch.einsum('bhqd,bhkd->bhqk', Q, K)   # attention scores
```

String `'bhqd,bhkd->bhqk'` ऐसे पढ़ो: "हर batch `b` और head `h` के लिए, query position `q` और key position `k` के बीच `d` dimension पर dot-product करो।" अगर subscript string पढ़ना आ गया, तो कोई भी attention implementation पढ़ लोगे।

---

## 3. Broadcasting

जब shapes exactly match नहीं करते, PyTorch try करता है **broadcast** करने के लिए — छोटे dims को size-1 axes के along repeat करके। Rule: shapes को right से align करो; sizes match करनी चाहिए या 1 होनी चाहिए।

```python
a = torch.randn(3, 4)    # (3, 4)
b = torch.randn(4)       # (4,) → broadcast to (1, 4) → (3, 4)
a + b                    # works, shape (3, 4)

a = torch.randn(3, 1)
b = torch.randn(1, 4)
a + b                    # both stretched, shape (3, 4)
```

ये positional encodings, masking, और per-head ops को elegant बनाता है। ये तब भी bite करता है जब आप broadcasting नहीं चाहते थे:

```python
loss = torch.randn(4)       # shape (4,)   — हर sample का एक loss
weight = torch.randn(4, 1)  # shape (4, 1) — ग़लती से extra dim आ गई
result = loss * weight       # shape (4, 4) — (4,) नहीं! Silent outer product.
```

PyTorch ने `(4,)` को `(1, 4)` मानकर `(4, 1)` के साथ broadcast किया → `(4, 4)` बना दिया। कोई error नहीं, बस ग़लत answer। Doubt में हमेशा shapes print करो।

---

## 4. वो basic functions जो हर कोई use करता है

### `exp` और `log`

`exp(x) = eˣ` (हमेशा positive)। `log(x)` इसका inverse है, defined for `x > 0`।

Logs multiplication को addition में convert करते हैं — इसलिए हम log-probabilities में काम करते हैं। `log(p₁ * p₂) = log(p₁) + log(p₂)`। Sums stable हैं; small probabilities के products underflow कर जाते हैं।

### Softmax

किसी भी real numbers (logits) के vector को probability distribution में बदलो:

```
softmax(x)[i] = exp(x[i]) / sum_j exp(x[j])
```

```python
def softmax(x):
    e = torch.exp(x - x.max())   # numerical stability के लिए max subtract करो
    return e / e.sum()

x = torch.tensor([2.0, 1.0, 0.1])
print(softmax(x))                # tensor([0.6590, 0.2424, 0.0986])
```

Properties:
- सारे outputs (0, 1) में, sum 1 → ये probability distribution है।
- Big input → big output (winner-take-most)।
- "subtract max" trick `exp` को big logits पर overflow होने से बचाता है — हर framework ये करता है।

### Cross-entropy

जब model का predicted distribution `p` है और true class `y` है:

```
loss = -log(p[y])
```

Equivalently, one-hot `y` के लिए:

```
loss = -sum_i y[i] * log(p[i])
```

PyTorch में cross-entropy raw logits लेता है और internally log-softmax apply करता है:

```python
loss = torch.nn.functional.cross_entropy(logits, target)
```

ये language models की **core pretraining loss** है। Fine-tuning stages (RLHF, DPO) इसके ऊपर और objectives add करते हैं, लेकिन cross-entropy foundation है।

### Sigmoid और Logit

```
sigmoid(x) = 1 / (1 + exp(-x))
logit(p)   = log(p / (1-p))
```

Sigmoid दो classes के लिए softmax है। Binary classification, MoE में gating, और SiLU activation (`x * sigmoid(x)`) में useful है।

---

## 5. Derivatives 5 minutes में

`f(x)` का **derivative** point `x` पर ये बताता है: "जब आप `x` को थोड़ा nudge करते हैं तो `f` कितनी जल्दी change होता है?" Geometrically, tangent line का slope।

पांच rules सब कुछ cover करते हैं:

| Function | Derivative |
|----------|------------|
| `c` (constant) | `0` |
| `x` | `1` |
| `xⁿ` | `n * xⁿ⁻¹` |
| `eˣ` | `eˣ` |
| `log(x)` | `1/x` |

Plus तीन combination rules:

- **Sum**: `(f + g)' = f' + g'`
- **Product**: `(fg)' = f'g + fg'`
- **Chain**: `(f(g(x)))' = f'(g(x)) * g'(x)`  ← **यही पूरा game है**

Chain rule कहता है: function-of-a-function को differentiate करने के लिए, outer derivative को inner derivative से multiply करो। Backprop *बस chain rule है जो giant computational graph से होकर apply होता है।* एक बार click हो जाए, तो training का हर "magic" obvious लगने लगता है।

### Worked example

```
f(x) = log(1 + exp(x))     (ये "softplus" है)
g(x) = exp(x)              inner part — derivative exp(x)
h(u) = log(1 + u)          outer part with u = g(x) — derivative 1/(1+u)
f'(x) = 1/(1 + exp(x)) * exp(x) = exp(x) / (1 + exp(x)) = sigmoid(x)
```

Cute — softplus का derivative sigmoid है। PyTorch ये जानता है बिना आपके formula लिखे।

---

## 6. Gradients (Multivariable Derivatives)

`f(x₁, ..., xₙ)` जो कई inputs का function है, उसके लिए **gradient** partial derivatives का vector है:

```
∇f = [ ∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ ]
```

ये `f` के fastest **increase** की direction को point करता है। `f` को minimize करने के लिए, `-∇f` direction में चलो। यही gradient descent है:

```
x_new = x_old - η * ∇f(x_old)
```

`η` (eta) **learning rate** है — आप कितना बड़ा step लेते हो।

### One-line intuition

Loss एक hilly landscape है। Gradient एक compass है जो हमेशा uphill point करता है। उसको subtract करो (small step के साथ) और आप downhill चलते हो। Bottom पर पहुंचना = trained।

### PyTorch आपके लिए ये करता है

आपको gradients hand से calculate नहीं करने। आप forward pass build करते हो, `.backward()` call करते हो, और PyTorch हर `requires_grad=True` tensor पर `.grad` fill कर देता है।

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()       # y = 1 + 4 + 9 = 14
y.backward()
print(x.grad)            # tensor([2., 4., 6.])  i.e. 2x
```

---

## 7. Jacobian (बस ताकि word आपको scare न करे)

अगर `f: ℝⁿ → ℝᵐ`, तो उसका **Jacobian** सारे partial derivatives का `m × n` matrix है। Single-output `f` के लिए, Jacobian बस gradient है (एक row vector)। Matmul के लिए, Jacobian huge है — लेकिन आप उसे कभी materialize नहीं करते। Backprop Jacobians को implicitly multiply करता है vector-Jacobian products के through।

आप usually इस word को ignore कर सकते हो। जब papers कहते हैं "vector-Jacobian product (VJP)," तो उनका मतलब है "backprop का one step." बस।

---

## 8. Probability essentials

### Expectation

`E[X]` random variable `X` की average value है। Discrete distribution के लिए, `E[X] = sum_i p_i * x_i`।

ML में, लगभग हर loss एक expectation है: "data के ऊपर average loss." हम mini-batch से इसको approximate करते हैं — stochastic gradient descent में **stochastic** यही है।

### KL Divergence

ये measure करता है कि दो distributions कितने different हैं:

```
KL(P || Q) = sum_x P(x) * log(P(x) / Q(x))
```

जब `P == Q` हो तो ये 0 है, और जैसे-जैसे वो diverge करते हैं ये grow करता है। Cross-entropy `-E_P[log Q]` है — उसको minimize करना `KL(P || Q)` plus एक constant को minimize करने जैसा है। तो **cross-entropy training model के distribution को data के distribution से match कराता है।**

### Sampling vs Argmax

जब next token के ऊपर एक probability distribution है, आप कर सकते हैं:
- **Argmax**: सबसे likely token pick करो (greedy decoding)।
- **Sample**: weighted die roll करो।
- **Top-k**: सिर्फ़ top `k` से sample करो।
- **Top-p / nucleus**: smallest set से sample करो जिसकी probability `p` तक sum होती है।
- **Min-p**: सिर्फ़ वो tokens रखो जिनकी probability ≥ `p × top_token_prob` हो। Model confidence के हिसाब से adapt होता है — 2025 से standard।
- **Temperature**: softmax से पहले logits को `T` से divide करो। Low `T` → sharp, deterministic. High `T` → flat, creative.

```python
def sample(logits, temperature=1.0, top_p=0.9):
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, idx = probs.sort(descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    keep = cumulative <= top_p
    keep[..., 0] = True  # top one को हमेशा keep करो
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum()
    sampled = torch.multinomial(sorted_probs, 1)
    return idx.gather(-1, sampled)
```

---

## 9. Tensor Anatomy: shapes जो रोज़ दिखेंगे

LLM कोड में, आप dims को इन conventional letters से देखते रहोगे:

| Letter | Meaning | Typical size |
|--------|---------|--------------|
| `B` | batch | 1-1024 |
| `T` | time / sequence length | 1-1M+ |
| `D` (या `d_model`) | model dimension | 768-16k |
| `H` (या `n_heads`) | attention heads | 8-128 |
| `d` (या `head_dim`) | per-head dim | 64-128 |
| `V` | vocabulary size | 32k-256k |
| `F` (या `d_ff`) | feed-forward inner dim | 2-8 × D |

Typical attention input `(B, T, D)` है। Heads में split करने के बाद ये `(B, H, T, d)` हो जाता है, जहां `D = H * d`। आप जो shape gymnastics करोगे वो इन forms के बीच reshape/permute calls हैं।

```python
B, T, D, H = 2, 5, 128, 8
d = D // H

x = torch.randn(B, T, D)
x_heads = x.view(B, T, H, d).transpose(1, 2)   # (B, H, T, d)
```

---

## 10. Numerical Stability Tricks जो जानने worth हैं

ये "nice to have" नहीं हैं — इनके बिना training silently NaN चली जाती है।

- **Softmax से पहले max subtract करो**: `softmax(x) = softmax(x - x.max())`। Same result, no overflow।
- **Log-sum-exp use करो**: `log(sum(exp(x))) = m + log(sum(exp(x - m)))` जहां `m = x.max()`।
- **Mixed precision** activations को bf16 में store करता है लेकिन reductions को fp32 में accumulate करता है (PyTorch आपके लिए handle करता है)।
- **Gradients clip करो**: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` rare bad batch को blow up होने से रोकता है।
- **`log(0)` से बचो**: `torch.log(p.clamp(min=1e-10))` से clamp करो या throughout log-space में काम करो।

---

## 11. एक cheat sheet जो आप 60 seconds में दोबारा पढ़ सकते हो

- `Linear` = `y = x @ W + b`। LLM का ज़्यादातर हिस्सा यही है।
- `softmax` logits को probability distribution में बदलता है।
- `cross_entropy` = सही answer को आपने जो probability assigned की उसका `-log`।
- `gradient` = partial derivatives का vector, uphill point करता है, downhill जाने के लिए subtract करो।
- `chain rule` backprop की soul है।
- Training के लिए `bf16`, inference के लिए `int4/int8`, sensitive bits (loss, optimizer state) के लिए `fp32`।
- जब नए हो तो shapes constantly print करो।
- KL divergence और cross-entropy basically same training objective हैं।

बस यही। बाकी ये guide इस पर build करती है। अगर future में कोई chapter में confuse हो जाओ, यहां वापस आओ।

Next: **[01-neural-networks.md](./01-neural-networks.md)**।
