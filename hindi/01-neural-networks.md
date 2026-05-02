# 01 · Neural Networks शुरू से समझो

> **TL;DR** Neural network matrix multiplications और simple non-linear functions का stack है। उसको train करने का मतलब है उन matrices के अंदर के numbers को tweak करना ताकि network का output उसके close आए जो आप चाहते हो। Backpropagation बस calculus का chain rule है, पूरे stack पर एक साथ apply होकर।

## 1. बड़ी picture

मान लो आप चाहते हो कि एक machine एक picture देखे और बताए "cat" है या "dog"। आप हर possible photo के लिए rules लिख नहीं सकते, इसलिए instead आप उसको millions of labelled examples देते हो और उसे rules खुद सीखने देते हो।

**Neural network** वो function है जो ये करता है। ये input numbers (pixels) लेता है, उन्हें multiply-and-add operations की layers से pass करता है, और output numbers spit करता है ("cat" के लिए एक probability और "dog" के लिए एक)।

Training वो process है जिसमें आप multiplications को nudge करते हो जब तक answers सही न आने लगें। Inference trained network को नए inputs पर use करना है।

बस यही। बाकी details हैं।

---

## 2. एक single neuron

सबसे छोटा building block **neuron** है। ये तीन काम करता है:

1. Inputs की list लेता है `x = [x1, x2, ..., xn]`।
2. हर input को उसके **weight** `w = [w1, w2, ..., wn]` से multiply करता है और sum करता है: `z = w1*x1 + w2*x2 + ... + wn*xn + b`। Extra `b` एक **bias** है।
3. वो sum एक non-linear **activation function** से pass करता है `a = f(z)`।

NumPy में:

```python
import numpy as np

def neuron(x, w, b):
    z = np.dot(w, x) + b      # weighted sum
    return max(0.0, z)        # ReLU activation
```

Activation क्यों? बिना उसके, neurons को stack करना सिर्फ़ एक single big matrix multiply जैसा होता है, जो सिर्फ़ straight lines सीख सकता है। Non-linearity वो चीज़ है जो network को bend, twist, और data को complex shapes में carve करने देती है।

### Common activations

| Name | Formula | कैसा दिखता है | कब use करें |
|------|---------|-----------|-------------|
| **ReLU** | `max(0, x)` | hockey stick | Hidden layers के लिए default. तेज़। |
| **Sigmoid** | `1 / (1 + e^-x)` | 0 से 1 तक S-curve | Old-school, एक class की probability output करना। |
| **Tanh** | `(e^x - e^-x) / (e^x + e^-x)` | -1 से 1 तक S-curve | कभी-कभी RNNs में use होता है। |
| **GELU** | `x * Φ(x)` (Gaussian CDF) | Smooth ReLU | Transformers (BERT, GPT-2) में standard। |
| **SiLU / Swish** | `x * sigmoid(x)` | Smooth ReLU | Llama, Qwen में SwiGLU के अंदर use होता है। |

Useful instinct: **modern LLMs अब अपने feed-forward blocks में ReLU shayad ही use करते हैं**। SiLU और GELU dominate करते हैं क्योंकि उनके smooth gradients ज़्यादा stably train करते हैं।

---

## 3. Neurons से Layers तक

**Layer** बस कई neurons हैं जो साथ-साथ बैठे हैं, सब same input को देख रहे हैं। अगर layer `L` में `m` inputs और `n` neurons हैं, आप पूरी layer को एक single matrix multiplication की तरह लिख सकते हो:

```
y = activation(x @ W + b)
# x: (m,)   input vector
# W: (m, n) weight matrix
# b: (n,)   bias vector
# y: (n,)   output vector
```

जब आप `B` inputs का **batch** एक साथ process करते हो, `x` बन जाता है `(B, m)` और `y` बन जाता है `(B, n)`। Matmul सारे `B` examples को parallel में handle करता है — इसलिए GPUs इसमें इतने तेज़ हैं।

**Multi-layer perceptron (MLP)** इन layers का stack है:

```
x → Linear → ReLU → Linear → ReLU → Linear → output
```

हर linear layer का अपना `W` और `b` होता है। पूरे network में कुछ हज़ार से कुछ billion तक ये parameters होते हैं।

---

## 4. PyTorch में Forward Pass

ये एक 2-layer MLP है जो 784 pixel values (एक flattened 28×28 image) लेता है और 10 logits (per digit एक) output करता है।

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

`nn.Linear(in, out)` exactly `y = x @ W + b` है, जिसमें `W` shape `(in, out)` और `b` shape `(out,)` का है।

---

## 5. Loss: हम कितने ग़लत हैं?

Network का काम है ज़्यादा बार सही होना। Train करने के लिए हमें एक single number चाहिए जो कहे "तुम इस बार इतने ग़लत थे।" वो number **loss** है।

`C` classes के साथ classification के लिए, हम **cross-entropy loss** use करते हैं:

```python
loss = -log(p_correct_class)
```

अगर model correct class को 0.9 probability assign करता है, `-log(0.9) ≈ 0.10` (small loss, अच्छा)।
अगर ये 0.01 probability assign करता है, `-log(0.01) ≈ 4.6` (big loss, बुरा)।

PyTorch का `nn.CrossEntropyLoss` raw **logits** (कोई भी real numbers) और integer target class लेता है। ये internally softmax + log apply करता है। उसको pass करने से पहले खुद softmax मत करो — common bug।

```python
loss_fn = nn.CrossEntropyLoss()
logits = model(fake_image)              # (B, 10)
target = torch.tensor([3])              # actual digit
loss = loss_fn(logits, target)          # scalar
```

Regression के लिए, **mean squared error** (`nn.MSELoss`) default है।

---

## 6. Gradient Descent: Weights को Nudge करना

Loss weights का function है। Calculus हमें **gradient** बताता है — weight-space में वो direction जहां loss सबसे fastest increase करता है। Loss को नीचे ले जाने के लिए, **opposite** direction में step लो:

```
w_new = w_old - learning_rate * gradient_of_loss_wrt_w
```

ये है **gradient descent**, training का heart।

**Learning rate** (LR) decide करता है आप कितना बड़ा step लेते हो। बहुत बड़ा और आप overshoot करते हो और diverge हो जाते हो। बहुत छोटा और आप forever लेते हो। Typical values: Adam के लिए `1e-3`, small problems पर SGD के लिए `1e-1`।

### Stochastic Gradient Descent (SGD)

हर step पर पूरे training set पर gradient compute करना expensive है। Instead हम **mini-batch** (मान लो 64 examples) use करते हैं, उस पर gradient compute करते हैं, step लेते हैं, और repeat करते हैं। ये **mini-batch SGD** है, default flavor।

### Better Optimizers

Plain SGD bounce करता है। Smarter optimizers recent gradients को track करते हैं और step size को per-parameter adapt करते हैं:

- **Momentum** — एक heavy ball की तरह जो downhill roll करती है, bounces को smooth करता है।
- **Adam** — gradient के first और second moments use करके LR को per parameter adapt करता है।
- **AdamW** — Adam **decoupled weight decay** के साथ। **हर modern LLM लगभग इसी से train होता है।**

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
```

---

## 7. Backpropagation: Chain Rule, Automated

हम वो gradient कैसे compute करते हैं? Answer है **backpropagation** — fancy name, simple idea।

Forward pass operations की एक chain build करता है: `loss = f3(f2(f1(x; W1); W2); W3)`। Chain rule कहता है:

```
d(loss)/d(W1) = d(loss)/d(f3) * d(f3)/d(f2) * d(f2)/d(f1) * d(f1)/d(W1)
```

हम local gradients को graph से backward multiply करते हैं। PyTorch graph automatically build करता है जैसे आप forward pass करते हो, फिर एक `loss.backward()` call से सारे gradients compute करता है।

पूरा training step:

```python
optimizer.zero_grad()      # last step के gradients clear करो
logits = model(x)          # forward
loss = loss_fn(logits, y)  # loss compute करो
loss.backward()            # backward — हर parameter पर .grad fill करता है
optimizer.step()           # parameters update करने के लिए .grad use करता है
```

**Common bug:** `zero_grad()` भूलने से gradients steps के बीच accumulate हो जाते हैं। कभी-कभी आप ये *चाहते* हो (large effective batch sizes के लिए gradient accumulation), लेकिन सिर्फ़ deliberately।

---

## 8. पूरा Training Loop

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

ये pattern — `zero_grad → forward → loss → backward → step` — same है चाहे आप 100-parameter MLP train कर रहे हो या 100-billion-parameter LLM। Scaffolding कभी नहीं बदलती; सिर्फ़ model और data बदलते हैं।

---

## 9. Initialization: कहां से शुरू करना matter करता है

अगर सारे weights zero पर start हों, तो हर neuron forever same काम करता है। इसलिए हम उन्हें randomly initialize करते हैं। लेकिन कोई भी random नहीं — bad init या तो activations explode करता है (NaN loss) या उन्हें kill करता है (zero gradients)।

दो go-to schemes:

- **Xavier / Glorot** (`nn.init.xavier_uniform_`): tanh / sigmoid layers के लिए अच्छा। Variance को layers के across roughly same रखता है।
- **Kaiming / He** (`nn.init.kaiming_normal_`): ReLU और friends के लिए अच्छा। `nn.Linear` के लिए PyTorch का default।

Modern transformers में, आप **scaled init** भी देखते हो: embeddings के लिए `std = 0.02` और residual projections के लिए `std = 0.02 / sqrt(2 * num_layers)` (GPT-2 style)। Point ये है कि network के deeper होने पर activations की variance को stable रखना है।

---

## 10. Regularization: सिर्फ़ memorize मत करो

Millions of parameters वाला network training set को perfectly memorize कर सकता है जबकि नए data के लिए कुछ भी नहीं करता। ये है **overfitting**। तीन knobs help करते हैं:

- **Weight decay** (a.k.a. L2 regularization): एक small term `λ * ||W||²` जो loss में add होता है। AdamW इसे cleanly करता है। Typical: LLMs के लिए `0.1`, vision के लिए `0.01`।
- **Dropout**: training time पर, fraction `p` activations को randomly zero-out करो। Network को force करता है किसी एक neuron पर depend न होने के लिए। Typical `p = 0.1`। *Note: ज़्यादातर modern LLMs dropout = 0 use करते हैं; उनका main regularizer data का sheer scale है।*
- **Early stopping**: validation loss को watch करो; जब improvement बंद हो जाए तो stop कर दो।

---

## 11. Common Bugs और उन्हें कैसे spot करें

- **Loss NaN है**: usually exploding gradients. LR कम करो, gradient clipping add करो (`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`), custom code में `log(0)` या `0/0` check करो।
- **Loss बिल्कुल नीचे नहीं जाता**: LR बहुत low, data pipeline में bug, frozen parameters, या आप `optimizer.zero_grad()` भूल गए।
- **Loss नीचे जाता है फिर ऊपर**: LR बहुत high. 10× से कम करो।
- **Train loss नीचे जाता है, val loss नहीं**: overfitting। ज़्यादा data, ज़्यादा regularization, या smaller model।
- **सारे outputs same**: dead network। Init check करो। Activation function check करो (ReLU dying problem at very high LR)।

---

## 12. इस chapter से Language Models तक

Transformer, heart में, **attention** (chapter 8) से interleaved MLPs का giant stack है। Attention layers sequence positions के बीच information move करती हैं; MLPs per-position thinking करते हैं। आपने जो कुछ यहां सीखा — linear layers, activations, loss, backprop, AdamW — सब apply होता है। Language modeling बस एक extra-fancy classification problem है जहां classes next token हैं।

तो जब आप बाद के chapters पढ़ो, याद रखो: **ये सब वही machinery है, बस clever arrangements में।**

---

## और गहराई से

- *Neural Networks and Deep Learning* by Michael Nielsen (free online book) — backprop का सबसे gentlest derivation।
- Andrej Karpathy का `micrograd` — 200 lines of Python जो scratch से autograd implement करता है। उसे पढ़ना backprop को truly *समझने* का सबसे fastest तरीका है।
- PyTorch के autograd mechanics के docs — micrograd build करने के बाद ये click करता है।

Next: **[02-pytorch.md](./02-pytorch.md)** — चलो framework master करें ताकि बाकी book English की तरह पढ़ी जाए।
