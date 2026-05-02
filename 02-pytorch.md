# 02 · PyTorch — Master the Framework

> **TL;DR** PyTorch gives you four superpowers: tensors that run on GPUs, automatic differentiation, a module system for organizing models, and data loaders that keep the GPU fed. Learn these four and the rest is mostly looking up function names.

## 1. Why PyTorch

PyTorch is the de facto standard for deep learning research and most production. It feels like NumPy but with a GPU and automatic gradients. The API is small enough to keep in your head and flexible enough that you can read and modify the entire codebase of any open-source LLM.

Two reasons it won:

1. **Eager execution** — operations run immediately, so you can `print(tensor.shape)` mid-network. Debugging is normal Python debugging.
2. **`nn.Module`** — a clean way to define, save, and ship models.

Install:

```bash
pip install torch  # CPU build is fine for learning
# For CUDA, follow https://pytorch.org/get-started/locally/
```

---

## 2. Tensors

A **tensor** is a multi-dimensional array. Think NumPy's `ndarray`, but with extra abilities: GPU placement and gradient tracking.

### Creation

```python
import torch

a = torch.tensor([1, 2, 3])              # from a list
b = torch.zeros(2, 3)                    # all zeros
c = torch.ones(2, 3)                     # all ones
d = torch.randn(2, 3)                    # standard normal
e = torch.arange(0, 10, 2)               # [0, 2, 4, 6, 8]
f = torch.linspace(0, 1, 5)              # 5 values from 0 to 1
g = torch.eye(3)                         # 3x3 identity
h = torch.empty(2, 3)                    # uninitialized — fast, garbage values

# from NumPy
import numpy as np
i = torch.from_numpy(np.array([1, 2]))
```

### Dtypes

```python
torch.float32  # default for floats — alias `torch.float`
torch.float16  # half precision (fp16) — saves memory, can underflow
torch.bfloat16 # brain float — same range as fp32, less precision (no underflow)
torch.int64    # default for ints — alias `torch.long`
torch.int32, torch.int8, torch.uint8, torch.bool
```

`bfloat16` is the workhorse for modern LLM training. It has the same exponent range as `float32` so you don't get NaN-from-overflow, but only 8 bits of mantissa.

### Devices

```python
x = torch.randn(2, 3)
x = x.to('cuda')              # move to GPU 0
x = x.to('cuda:1')            # move to GPU 1
x = x.cpu()                   # back to CPU
x = x.to('mps')               # Apple Silicon

# create directly on GPU (faster than create-then-move)
y = torch.randn(2, 3, device='cuda', dtype=torch.bfloat16)
```

A tensor on GPU and a tensor on CPU **cannot** interact. You'll see `RuntimeError: Expected all tensors to be on the same device`.

### Shapes and reshaping

```python
x = torch.randn(2, 3, 4)
x.shape                       # torch.Size([2, 3, 4])
x.numel()                     # 24
x.dim()                       # 3

x.view(6, 4)                  # reshape — needs contiguous memory
x.reshape(6, 4)               # reshape — handles non-contiguous
x.permute(2, 0, 1).shape      # (4, 2, 3) — reorder dims
x.transpose(0, 1).shape       # (3, 2, 4) — swap two dims
x.unsqueeze(0).shape          # (1, 2, 3, 4) — add a dim
x.squeeze().shape             # remove all size-1 dims
x.flatten(1).shape            # (2, 12) — flatten from dim 1 onward
```

`view` is free (no copy). `reshape` may copy. `contiguous()` forces a copy to memory-contiguous layout — sometimes needed before `view`.

### Indexing

```python
x = torch.arange(20).reshape(4, 5)
x[0]                          # first row → (5,)
x[0, 2]                       # scalar
x[:, 1]                       # second column → (4,)
x[1:3]                        # rows 1 and 2 → (2, 5)
x[x > 10]                     # boolean mask, returns flat tensor
x[[0, 2]]                     # rows 0 and 2 → (2, 5)
```

### Operations

```python
a + b                         # elementwise add (with broadcasting)
a * b                         # elementwise multiply (NOT matmul)
a @ b                         # matrix multiply, same as a.matmul(b)
torch.matmul(a, b)            # ditto
a.sum(); a.mean(); a.std()    # reductions
a.sum(dim=0)                  # reduce along dim 0
a.argmax(dim=-1)              # index of max along last dim
torch.cat([a, b], dim=0)      # concatenate
torch.stack([a, b], dim=0)    # stack — adds a new dim
```

### Broadcasting

When shapes don't match, PyTorch tries to align dims from the right. A shape of size 1 is stretched.

```python
a = torch.randn(3, 1)
b = torch.randn(1, 4)
(a + b).shape                 # (3, 4) — both stretched
```

This is a *huge* productivity win. It's also a frequent source of subtle bugs — `(B, 1) + (B,)` does not behave the way you expect.

---

## 3. Autograd: gradients for free

Set `requires_grad=True` on a tensor and PyTorch will track every operation on it. Then call `.backward()` on a scalar to fill `.grad` on every input tensor.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1        # y = (x+1)^2
y.backward()
print(x.grad)                 # 2x + 2 = 8
```

Inside `nn.Module`, every `nn.Parameter` has `requires_grad=True` by default.

### `torch.no_grad()` and `inference_mode()`

When you don't need gradients (evaluation, inference), wrap the code to save memory and time.

```python
with torch.no_grad():
    preds = model(x)

# Faster, stricter version (PyTorch 1.9+)
with torch.inference_mode():
    preds = model(x)
```

### Detaching

`tensor.detach()` returns a new tensor that shares storage but is cut from the autograd graph. Useful for logging values you don't want to flow gradients through.

---

## 4. `nn.Module`: organizing models

`nn.Module` is a class you subclass to define a model. It tracks parameters, sub-modules, and provides nice helpers (`.to()`, `.train()`, `.eval()`, saving/loading).

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

| Method | What it does |
|--------|--------------|
| `model.parameters()` | iterator over all `nn.Parameter` |
| `model.named_parameters()` | same, with names like `'attn.in_proj_weight'` |
| `model.state_dict()` | dict of `{name: tensor}` for saving |
| `model.load_state_dict(d)` | restore from such a dict |
| `model.train()` / `model.eval()` | toggle dropout / batchnorm behavior |
| `model.to('cuda')` | move all parameters to a device |

### Common layers

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

## 5. Optimizers and schedulers

```python
opt = torch.optim.AdamW(model.parameters(),
                        lr=3e-4,
                        betas=(0.9, 0.95),
                        weight_decay=0.1)
```

`betas` are the EMA decays for the first and second moments. `(0.9, 0.95)` is the LLM standard; the default `(0.9, 0.999)` is also fine.

A **learning-rate schedule** changes LR over training. The most common for LLMs: warmup then cosine decay.

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

## 6. Data: `Dataset` and `DataLoader`

A **`Dataset`** says how to get one example. A **`DataLoader`** batches them, shuffles, and prefetches with worker processes.

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

- `num_workers=4` runs 4 parallel processes loading data.
- `pin_memory=True` puts data in pinned RAM so GPU transfers are faster (`x.to('cuda', non_blocking=True)`).
- For huge datasets, write a streaming `IterableDataset` instead of loading everything at once.

---

## 7. The training loop, properly

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

A few tricks worth knowing:

- `zero_grad(set_to_none=True)` is faster than zero-filling.
- `clip_grad_norm_` prevents the rare exploding step from killing training.
- For `bfloat16` you don't actually need `GradScaler` — only `fp16` underflows. Modern code typically just uses `bfloat16` directly without a scaler.

---

## 8. Mixed-precision training

Doing math in 16-bit cuts memory use roughly in half and lets Tensor Cores run ~2× faster. Two flavors:

- **fp16** — narrower range than fp32, can underflow. Needs `GradScaler` to scale loss into a safe range.
- **bf16** — same range as fp32, less mantissa precision. **Just works on Ampere+ (A100, RTX 30xx and newer).**

```python
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    logits = model(x)
    loss = loss_fn(logits, y)
loss.backward()
opt.step()
```

Inside the autocast context, ops that benefit from low precision (matmul, conv) run in bf16; ops that need full precision (reductions, softmax) stay in fp32. Magic.

---

## 9. Saving and loading

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

**Never** `torch.save(model)` directly — it pickles the class, breaking when you refactor. Always save `state_dict()`.

For sharing models between machines, also use `safetensors` (`pip install safetensors`). It's safe (no pickle) and faster to load.

---

## 10. `torch.compile` — the easy speedup

PyTorch 2.0+ has a JIT compiler. One line, often 1.3-2× faster:

```python
model = torch.compile(model)
```

Caveats:
- First call is slow (compiling kernels). After warmup, you win.
- Dynamic shapes and Python control flow inside `forward` can break it. `mode='reduce-overhead'` or `fullgraph=True` are useful flags.
- For inference, `torch.compile(model, mode='max-autotune')` gives the most speed.

---

## 11. Distributed training (preview)

For real LLM training you want multiple GPUs. PyTorch offers:

- **DDP** (`DistributedDataParallel`) — replicate the model on every GPU, sum gradients across them. Easy. Limited by single-GPU memory.
- **FSDP** (`FullyShardedDataParallel`) — shard parameters/gradients/optimizer state across GPUs. Lets you train models that don't fit on one GPU.
- **Tensor / pipeline parallelism** — for the truly huge models. Frameworks like Megatron-LM, DeepSpeed, and `torch.distributed.tensor` provide this.

We cover this properly in **[14-training-small-language-models.md](./14-training-small-language-models.md)**.

---

## 12. Debugging cheat sheet

- Print shapes: `print(x.shape)` after every layer until they match what you expect.
- Use a tiny dataset (10 examples) and overfit it. If loss doesn't go to ~0, your model has a bug, not a data problem.
- `assert torch.isfinite(loss).all(), 'NaN!'` catches NaNs at the source.
- `torch.autograd.set_detect_anomaly(True)` — slow but pinpoints the op that produced NaN.
- `nvidia-smi` in another terminal — watch GPU memory and utilization.
- For shape mismatches, write the expected shape next to every tensor as a comment. Future-you will thank you.

---

## 13. PyTorch's quiet ecosystem

You'll often reach for:

- **`torch.func`** — `vmap`, `grad`, JAX-style functional transforms.
- **`torch.nn.functional` (F)** — stateless versions of `nn.*`. Useful in custom forward methods.
- **`torchvision`, `torchaudio`** — datasets and pre-built models.
- **`einops`** (`pip install einops`) — readable reshapes: `rearrange(x, 'b t (h d) -> b h t d', h=8)` beats fiddling with `view` and `permute`.

---

## Going deeper

- The official [PyTorch tutorials](https://pytorch.org/tutorials/) — start with "60 minute blitz."
- Andrej Karpathy's `nanoGPT` — the single best ~600-line PyTorch codebase to read.
- The PyTorch source for `nn.Linear` and `nn.MultiheadAttention` — surprisingly readable.

Next: **[03-gpu-computing.md](./03-gpu-computing.md)** — what's actually happening when you call `.to('cuda')`.
