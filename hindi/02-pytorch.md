# 02 · PyTorch — Framework Master करो

> **TL;DR** PyTorch आपको चार superpowers देता है: tensors जो GPUs पर चलते हैं, automatic differentiation, models organize करने के लिए module system, और data loaders जो GPU को feed करते रहते हैं। ये चार सीख लो और बाकी ज़्यादातर function names look up करना है।

## 1. PyTorch क्यों

PyTorch deep learning research और ज़्यादातर production के लिए de facto standard है। ये NumPy की तरह feel होता है लेकिन GPU और automatic gradients के साथ। API इतना small है कि head में रख सकते हो और इतना flexible है कि किसी भी open-source LLM के पूरे codebase को read और modify कर सकते हो।

दो reasons जिनसे ये जीता:

1. **Eager execution** — operations तुरंत run होते हैं, तो आप mid-network में `print(tensor.shape)` कर सकते हो। Debugging normal Python debugging है।
2. **`nn.Module`** — models को define, save, और ship करने का clean तरीका।

Install:

```bash
pip install torch  # CPU build सीखने के लिए ठीक है
# CUDA के लिए, https://pytorch.org/get-started/locally/ follow करो
```

---

## 2. Tensors

**Tensor** एक multi-dimensional array है। NumPy का `ndarray` सोचो, लेकिन extra abilities के साथ: GPU placement और gradient tracking।

### Creation

```python
import torch

a = torch.tensor([1, 2, 3])              # list से
b = torch.zeros(2, 3)                    # सारे zeros
c = torch.ones(2, 3)                     # सारे ones
d = torch.randn(2, 3)                    # standard normal
e = torch.arange(0, 10, 2)               # [0, 2, 4, 6, 8]
f = torch.linspace(0, 1, 5)              # 0 से 1 तक 5 values
g = torch.eye(3)                         # 3x3 identity
h = torch.empty(2, 3)                    # uninitialized — fast, garbage values

# NumPy से
import numpy as np
i = torch.from_numpy(np.array([1, 2]))
```

### Dtypes

```python
torch.float32  # floats के लिए default — alias `torch.float`
torch.float16  # half precision (fp16) — memory बचाता है, underflow कर सकता है
torch.bfloat16 # brain float — fp32 जैसी range, कम precision (no underflow)
torch.int64    # ints के लिए default — alias `torch.long`
torch.int32, torch.int8, torch.uint8, torch.bool
```

`bfloat16` modern LLM training का workhorse है। इसमें same exponent range है `float32` जैसी तो आपको NaN-from-overflow नहीं मिलेगा, लेकिन सिर्फ़ 8 bits of mantissa।

### Devices

```python
x = torch.randn(2, 3)
x = x.to('cuda')              # GPU 0 पर move करो
x = x.to('cuda:1')            # GPU 1 पर move करो
x = x.cpu()                   # वापस CPU पर
x = x.to('mps')               # Apple Silicon

# सीधे GPU पर बनाओ (बना के move करने से faster)
y = torch.randn(2, 3, device='cuda', dtype=torch.bfloat16)
```

GPU पर एक tensor और CPU पर एक tensor **interact नहीं कर सकते**। आपको दिखेगा `RuntimeError: Expected all tensors to be on the same device`।

### Shapes और Reshaping

```python
x = torch.randn(2, 3, 4)
x.shape                       # torch.Size([2, 3, 4])
x.numel()                     # 24
x.dim()                       # 3

x.view(6, 4)                  # reshape — contiguous memory चाहिए
x.reshape(6, 4)               # reshape — non-contiguous handle करता है
x.permute(2, 0, 1).shape      # (4, 2, 3) — dims reorder
x.transpose(0, 1).shape       # (3, 2, 4) — दो dims swap
x.unsqueeze(0).shape          # (1, 2, 3, 4) — एक dim add करो
x.squeeze().shape             # सारे size-1 dims remove
x.flatten(1).shape            # (2, 12) — dim 1 से flatten
```

`view` free है (no copy)। `reshape` copy कर सकता है। `contiguous()` memory-contiguous layout में copy force करता है — कभी-कभी `view` से पहले चाहिए।

### Indexing

```python
x = torch.arange(20).reshape(4, 5)
x[0]                          # पहली row → (5,)
x[0, 2]                       # scalar
x[:, 1]                       # दूसरा column → (4,)
x[1:3]                        # rows 1 और 2 → (2, 5)
x[x > 10]                     # boolean mask, flat tensor return करता है
x[[0, 2]]                     # rows 0 और 2 → (2, 5)
```

### Operations

```python
a + b                         # elementwise add (broadcasting के साथ)
a * b                         # elementwise multiply (matmul NHI)
a @ b                         # matrix multiply, same as a.matmul(b)
torch.matmul(a, b)            # ditto
a.sum(); a.mean(); a.std()    # reductions
a.sum(dim=0)                  # dim 0 के along reduce
a.argmax(dim=-1)              # last dim के along max का index
torch.cat([a, b], dim=0)      # concatenate
torch.stack([a, b], dim=0)    # stack — नया dim add करता है
```

### Broadcasting

जब shapes match नहीं करते, PyTorch dims को right से align करने की कोशिश करता है। Size 1 का shape stretch हो जाता है।

```python
a = torch.randn(3, 1)
b = torch.randn(1, 4)
(a + b).shape                 # (3, 4) — दोनों stretched
```

ये *huge* productivity win है। ये subtle bugs का frequent source भी है — `(B, 1) + (B,)` वैसे behave नहीं करता जैसे आप expect करते हो।

---

## 3. Autograd: Free में Gradients

Tensor पर `requires_grad=True` set करो और PyTorch उस पर हर operation को track करेगा। फिर एक scalar पर `.backward()` call करो हर input tensor पर `.grad` fill करने के लिए।

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1        # y = (x+1)^2
y.backward()
print(x.grad)                 # 2x + 2 = 8
```

`nn.Module` के अंदर, हर `nn.Parameter` का by default `requires_grad=True` होता है।

### `torch.no_grad()` और `inference_mode()`

जब आपको gradients नहीं चाहिए (evaluation, inference), तो memory और time बचाने के लिए कोड wrap करो।

```python
with torch.no_grad():
    preds = model(x)

# Faster, stricter version (PyTorch 1.9+)
with torch.inference_mode():
    preds = model(x)
```

### Detaching

`tensor.detach()` एक नया tensor return करता है जो storage share करता है लेकिन autograd graph से cut है। उन values को log करने के लिए useful है जिनके through आप gradients flow नहीं करना चाहते।

---

## 4. `nn.Module`: Models को Organize करना

`nn.Module` एक class है जिसे आप model define करने के लिए subclass करते हो। ये parameters, sub-modules track करता है, और nice helpers देता है (`.to()`, `.train()`, `.eval()`, saving/loading)।

```python
import torch.nn as nn

class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):                     # x: (B, T, D)
        a, _ = self.attn(x, x, x)             # self-attention
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x

block = TinyTransformerBlock(d_model=256, n_heads=8)
print(sum(p.numel() for p in block.parameters()))   # ~600k
```

Key methods:

| Method | क्या करता है |
|--------|--------------|
| `model.parameters()` | सारे `nn.Parameter` पर iterator |
| `model.named_parameters()` | same, names के साथ जैसे `'attn.in_proj_weight'` |
| `model.state_dict()` | saving के लिए `{name: tensor}` का dict |
| `model.load_state_dict(d)` | ऐसे dict से restore |
| `model.train()` / `model.eval()` | dropout / batchnorm behavior toggle |
| `model.to('cuda')` | सारे parameters को device पर move करो |

### Common Layers

```python
nn.Linear(in, out, bias=True)
nn.Embedding(num_embeddings, embedding_dim)
nn.LayerNorm(d)
nn.RMSNorm(d)                 # PyTorch 2.4+
nn.Dropout(p=0.1)
nn.GELU(); nn.SiLU(); nn.ReLU()
nn.MultiheadAttention(d, h, batch_first=True)
nn.Conv2d(in_c, out_c, kernel_size)
nn.LSTM(in, hidden, batch_first=True)
```

---

## 5. Optimizers और Schedulers

```python
opt = torch.optim.AdamW(model.parameters(),
                        lr=3e-4,
                        betas=(0.9, 0.95),
                        weight_decay=0.1)
```

`betas` first और second moments के लिए EMA decays हैं। `(0.9, 0.95)` LLM standard है; default `(0.9, 0.999)` भी ठीक है।

**Learning-rate schedule** training के through LR change करता है। LLMs के लिए most common: warmup फिर cosine decay।

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

warmup = LinearLR(opt, start_factor=1e-4, total_iters=2000)
cosine = CosineAnnealingLR(opt, T_max=100_000, eta_min=3e-5)
scheduler = SequentialLR(opt, [warmup, cosine], milestones=[2000])

for step in range(102_000):
    # ... forward, backward, opt.step() ...
    scheduler.step()
```

---

## 6. Data: `Dataset` और `DataLoader`

**`Dataset`** बताता है कि एक example कैसे लाना है। **`DataLoader`** उन्हें batch करता है, shuffle करता है, और worker processes से prefetch करता है।

```python
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, text, block_size=128):
        self.text = text
        self.block_size = block_size
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.data = torch.tensor([self.stoi[c] for c in text])

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, i):
        x = self.data[i : i + self.block_size]
        y = self.data[i + 1 : i + 1 + self.block_size]
        return x, y

ds = CharDataset(open('shakespeare.txt').read())
loader = DataLoader(ds, batch_size=64, shuffle=True,
                    num_workers=4, pin_memory=True)
```

- `num_workers=4` 4 parallel processes चलाता है data loading करते हुए।
- `pin_memory=True` data को pinned RAM में डालता है तो GPU transfers faster होती हैं (`x.to('cuda', non_blocking=True)`)।
- Huge datasets के लिए, सब कुछ load करने के बजाय streaming `IterableDataset` लिखो।

---

## 7. Training Loop, सही तरीके से

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyModel().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = torch.amp.GradScaler('cuda')

for step, (x, y) in enumerate(loader):
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, V), y.view(-1))

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
```

कुछ tricks जो जानने worth हैं:

- `zero_grad(set_to_none=True)` zero-filling से faster है।
- `clip_grad_norm_` rare exploding step को training kill करने से रोकता है।
- `bfloat16` के लिए आपको actually `GradScaler` नहीं चाहिए — सिर्फ़ `fp16` underflow करता है। Modern code typically सीधे `bfloat16` use करता है बिना scaler के।

---

## 8. Mixed-precision Training

16-bit में math करने से memory roughly half cut हो जाती है और Tensor Cores ~2× faster run होते हैं। दो flavors:

- **fp16** — fp32 से narrower range, underflow कर सकता है। Loss को safe range में scale करने के लिए `GradScaler` चाहिए।
- **bf16** — fp32 जैसी same range, कम mantissa precision. **Ampere+ (A100, RTX 30xx और newer) पर बस works।**

```python
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    logits = model(x)
    loss = loss_fn(logits, y)
loss.backward()
opt.step()
```

Autocast context के अंदर, low precision से benefit होने वाले ops (matmul, conv) bf16 में run होते हैं; जिनको full precision चाहिए (reductions, softmax) वो fp32 में रहते हैं। Magic।

---

## 9. Saving और Loading

```python
# save
torch.save(model.state_dict(), 'model.pt')
torch.save({
    'model': model.state_dict(),
    'opt': opt.state_dict(),
    'step': step,
    'scaler': scaler.state_dict(),
}, 'checkpoint.pt')

# load
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
```

**Never** सीधे `torch.save(model)` — ये class को pickle करता है, जो refactor करते समय break हो जाता है। हमेशा `state_dict()` save करो।

Machines के बीच models share करने के लिए, `safetensors` भी use करो (`pip install safetensors`)। ये safe है (no pickle) और load करने में faster।

---

## 10. `torch.compile` — Easy Speedup

PyTorch 2.0+ में JIT compiler है। एक line, often 1.3-2× faster:

```python
model = torch.compile(model)
```

Caveats:
- First call slow है (compiling kernels)। Warmup के बाद, आप जीतते हो।
- `forward` के अंदर dynamic shapes और Python control flow इसे break कर सकते हैं। `mode='reduce-overhead'` या `fullgraph=True` useful flags हैं।
- Inference के लिए, `torch.compile(model, mode='max-autotune')` सबसे ज़्यादा speed देता है।

---

## 11. Distributed Training (Preview)

Real LLM training के लिए आप multiple GPUs चाहते हो। PyTorch offer करता है:

- **DDP** (`DistributedDataParallel`) — हर GPU पर model replicate करो, उन सब के across gradients sum करो। Easy। Single-GPU memory से limited।
- **FSDP** (`FullyShardedDataParallel`) — parameters/gradients/optimizer state को GPUs के across shard करो। आपको ऐसे models train करने देता है जो एक GPU पर fit नहीं होते।
- **Tensor / pipeline parallelism** — truly huge models के लिए। Megatron-LM, DeepSpeed, और `torch.distributed.tensor` जैसे frameworks ये provide करते हैं।

हम इसे **[14-training-small-language-models.md](./14-training-small-language-models.md)** में properly cover करते हैं।

---

## 12. Debugging Cheat Sheet

- Shapes print करो: हर layer के बाद `print(x.shape)` जब तक आपकी expectation match न करें।
- Tiny dataset (10 examples) use करो और उस पर overfit करो। अगर loss ~0 तक नहीं जाता, तो model में bug है, data problem नहीं।
- `assert torch.isfinite(loss).all(), 'NaN!'` source पर NaNs catch करता है।
- `torch.autograd.set_detect_anomaly(True)` — slow लेकिन उस op को pinpoint करता है जिसने NaN produce किया।
- `nvidia-smi` दूसरे terminal में — GPU memory और utilization watch करो।
- Shape mismatches के लिए, हर tensor के पास expected shape comment लिखो। Future-you आपका thanks करेगा।

---

## 13. PyTorch का Quiet Ecosystem

आप अक्सर इनके लिए reach करोगे:

- **`torch.func`** — `vmap`, `grad`, JAX-style functional transforms।
- **`torch.nn.functional` (F)** — `nn.*` के stateless versions। Custom forward methods में useful।
- **`torchvision`, `torchaudio`** — datasets और pre-built models।
- **`einops`** (`pip install einops`) — readable reshapes: `rearrange(x, 'b t (h d) -> b h t d', h=8)` `view` और `permute` के साथ fiddle करने से बेहतर है।

---

## और गहराई से

- Official [PyTorch tutorials](https://pytorch.org/tutorials/) — "60 minute blitz" से शुरू करो।
- Andrej Karpathy का `nanoGPT` — single best ~600-line PyTorch codebase जो पढ़ने worth है।
- `nn.Linear` और `nn.MultiheadAttention` के लिए PyTorch source — surprisingly readable।

Next: **[03-gpu-computing.md](./03-gpu-computing.md)** — actually क्या हो रहा है जब आप `.to('cuda')` call करते हो।
