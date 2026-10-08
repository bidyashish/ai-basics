# 11 · Building Qwen from Scratch

> **TL;DR** Qwen 2.5 / Qwen 3 are pre-norm transformers with RMSNorm, GQA, RoPE, and SwiGLU — the same template from chapter 10, with a few specific hyperparameter choices. In this chapter we build a full **Qwen-shaped** model in ~250 lines of PyTorch, load real Qwen weights into it, run a forward pass, and generate text. **No `transformers` library tricks — every layer is yours.**

## 1. Why Qwen as the reference

Qwen has three useful properties:

1. **Open weights**, permissive license, available in many sizes from 0.5B to 235B.
2. **Mainstream architecture** — what you learn here ports directly to Llama, Mistral, Gemma, DeepSeek-base.
3. **Excellent quality** — probably the strongest small-model family in 2026.

We'll target **Qwen2.5-0.5B-Instruct** as the implementation reference because it's small enough to download and run on a laptop CPU. Same code works for Qwen3-4B, just with bigger config.

---

## 2. The configuration

Qwen2.5-0.5B's `config.json` (paraphrased, the fields you actually use):

```python
@dataclass
class QwenConfig:
    vocab_size:         int = 151_936
    hidden_size:        int = 896           # D
    intermediate_size:  int = 4_864         # F (SwiGLU inner)
    num_hidden_layers:  int = 24            # L
    num_attention_heads:int = 14            # H_q
    num_key_value_heads:int = 2             # H_kv (group = 7)
    head_dim:           int = 64            # D / H_q
    max_position:       int = 32_768
    rope_theta:         float = 1_000_000.0 # RoPE base
    rms_norm_eps:       float = 1e-6
    tie_word_embeddings:bool = True
```

For Qwen2.5-7B: `D=3584, L=28, H_q=28, H_kv=4, F=18944, V=152064, tie=False`.
For Qwen3-4B: `D=2560, L=36, H_q=32, H_kv=8, F=9728, tie=False`.

---

## 3. The full model in PyTorch

```python
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class QwenConfig:
    vocab_size: int = 151_936
    hidden_size: int = 896
    intermediate_size: int = 4_864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    max_position: int = 32_768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, D, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))
        self.eps = eps

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * h).to(x.dtype)


def precompute_rope(head_dim, max_T, base, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                 # (T, head_dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x, cos, sin):
    # x: (B, H, T, head_dim) — using HF-style "split-half" layout
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :half]
    sin = sin[None, None, :, :half]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class QwenAttention(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg
        self.H_q = cfg.num_attention_heads
        self.H_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        D = cfg.hidden_size
        self.q_proj = nn.Linear(D, self.H_q  * self.head_dim, bias=True)
        self.k_proj = nn.Linear(D, self.H_kv * self.head_dim, bias=True)
        self.v_proj = nn.Linear(D, self.H_kv * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.H_q * self.head_dim, D, bias=False)

    def forward(self, x, cos, sin, past_kv=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.H_q,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.H_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.H_kv, self.head_dim).transpose(1, 2)

        # apply RoPE — slice cos/sin for the current token range
        if past_kv is not None:
            past_T = past_kv[0].size(2)
            cos_q = cos[past_T:past_T + T]
            sin_q = sin[past_T:past_T + T]
        else:
            cos_q = cos[:T]
            sin_q = sin[:T]
        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_q, sin_q)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=(past_kv is None and T > 1),
            enable_gqa=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_kv


class SwiGLU(nn.Module):
    def __init__(self, D, F_):
        super().__init__()
        self.gate_proj = nn.Linear(D, F_, bias=False)
        self.up_proj   = nn.Linear(D, F_, bias=False)
        self.down_proj = nn.Linear(F_, D, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenBlock(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.input_layernorm          = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn                = QwenAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp                      = SwiGLU(cfg.hidden_size, cfg.intermediate_size)

    def forward(self, x, cos, sin, past_kv=None):
        a, new_kv = self.self_attn(self.input_layernorm(x), cos, sin, past_kv)
        x = x + a
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv


class Qwen(nn.Module):
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([QwenBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, ids, kv_cache=None):
        device, dtype = ids.device, self.embed_tokens.weight.dtype
        cos, sin = precompute_rope(self.cfg.head_dim, self.cfg.max_position,
                                   self.cfg.rope_theta, device, dtype)
        x = self.embed_tokens(ids)
        new_cache = []
        for i, layer in enumerate(self.layers):
            past = kv_cache[i] if kv_cache else None
            x, new_kv = layer(x, cos, sin, past)
            new_cache.append(new_kv)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_cache
```

That's the entire model. Roughly 200 lines including config and helpers. Compare with HuggingFace's `modeling_qwen2.py` (1500+ lines): theirs handles many edge cases, but the *math* is identical to what's above.

---

## 4. Loading real Qwen weights

```python
from huggingface_hub import snapshot_download
import safetensors.torch as st
import os, json

ckpt = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')
with open(os.path.join(ckpt, 'config.json')) as f:
    cfg_json = json.load(f)

cfg = QwenConfig(
    vocab_size=cfg_json['vocab_size'],
    hidden_size=cfg_json['hidden_size'],
    intermediate_size=cfg_json['intermediate_size'],
    num_hidden_layers=cfg_json['num_hidden_layers'],
    num_attention_heads=cfg_json['num_attention_heads'],
    num_key_value_heads=cfg_json['num_key_value_heads'],
    head_dim=cfg_json.get('head_dim', cfg_json['hidden_size'] // cfg_json['num_attention_heads']),
    max_position=cfg_json['max_position_embeddings'],
    rope_theta=cfg_json['rope_theta'],
    rms_norm_eps=cfg_json['rms_norm_eps'],
    tie_word_embeddings=cfg_json.get('tie_word_embeddings', True),
)

model = Qwen(cfg).to(torch.bfloat16).to('cuda' if torch.cuda.is_available() else 'cpu')
state = {}
for fn in os.listdir(ckpt):
    if fn.endswith('.safetensors'):
        state.update(st.load_file(os.path.join(ckpt, fn)))

# HuggingFace key names: "model.layers.0.self_attn.q_proj.weight"
# Our names:             "layers.0.self_attn.q_proj.weight"
remap = {k.replace('model.', '', 1): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(remap, strict=False)
print('missing:',   missing)
print('unexpected:', unexpected)
```

A few things to know:

- HuggingFace prefixes everything with `model.`, so we strip it. The `lm_head.weight` is at the top level for untied models.
- Safetensors is preferred over `.bin` (no pickle, faster load).
- Cast to `bfloat16` to fit on small GPUs and run faster.
- For big models, use `snapshot_download` + `safetensors.torch.load_file` and map shards to your layers.

---

## 5. Tokenization (just use the official tokenizer)

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(ckpt)

prompt = tok.apply_chat_template(
    [{'role': 'user', 'content': 'Why is the sky blue?'}],
    tokenize=False, add_generation_prompt=True,
)
ids = tok(prompt, return_tensors='pt').input_ids.to(model.embed_tokens.weight.device)
```

The chat template wraps your message in Qwen's special tokens: `<|im_start|>user ...<|im_end|>\n<|im_start|>assistant\n`.

---

## 6. Greedy generation with a KV cache

```python
@torch.inference_mode()
def generate(model, ids, max_new=200, eos_id=None):
    cache = None
    out = ids[0].tolist()
    cur = ids
    for _ in range(max_new):
        logits, cache = model(cur, kv_cache=cache)
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
        out.append(next_id.item())
        if eos_id is not None and next_id.item() == eos_id:
            break
        cur = next_id           # only feed the latest token next round
    return out

eos = tok.convert_tokens_to_ids('<|im_end|>')
out_ids = generate(model, ids, max_new=200, eos_id=eos)
print(tok.decode(out_ids))
```

The first call fills the cache for the whole prompt. Subsequent calls process **only the new token**, using the cache for context — that's why decode is fast.

---

## 7. Sampling with temperature, top-k, top-p

```python
def sample(logits, temperature=0.7, top_k=20, top_p=0.9):
    logits = logits / temperature
    if top_k:
        v, _ = torch.topk(logits, k=top_k)
        logits[logits < v[..., -1, None]] = -float('inf')
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, idx = probs.sort(descending=True, dim=-1)
        cum = sorted_probs.cumsum(-1)
        keep = cum <= top_p
        keep[..., 0] = True
        sorted_probs = sorted_probs * keep
        sorted_probs /= sorted_probs.sum(-1, keepdim=True)
        out = idx.gather(-1, torch.multinomial(sorted_probs, 1))
        return out
    return torch.multinomial(probs, 1)
```

Plug it into `generate` in place of `argmax`. **For chat / open-ended tasks, sampling is much better than greedy** (less repetition, more interesting outputs). Defaults that work: `temperature=0.7, top_p=0.9`.

For reasoning models (Qwen3 in "thinking" mode), use `temperature=0.6, top_p=0.95` per their model card; the chain-of-thought pattern is fragile.

---

## 8. Verifying you got it right

After loading weights, your model's first-token output for a fixed prompt should match HuggingFace's exactly (within fp tolerance). Quick check:

```python
from transformers import AutoModelForCausalLM
hf = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(model.embed_tokens.weight.device)
hf.eval(); model.eval()

with torch.inference_mode():
    a, _ = model(ids); a = a[0, -1].float()
    b = hf(ids).logits[0, -1].float()
    print('top-5 ours :', a.topk(5).indices.tolist())
    print('top-5 theirs:', b.topk(5).indices.tolist())
    print('cos sim    :', F.cosine_similarity(a, b, dim=0).item())
```

If `cos sim > 0.999` and top-5 match, you're correct. If they diverge, common culprits:

1. **RoPE pair layout** (split-half vs interleaved). Match HF.
2. **`bias=True` on qkv proj for Qwen** but **not** for Llama. Yes, Qwen has bias on Q, K, V (a quirk!).
3. **`o_proj` has no bias.**
4. **Norm computed in wrong dtype** (use fp32 for the variance).
5. **Wrong RoPE base** (Qwen uses 1e6, Llama 3 uses 5e5).
6. **Causal mask** wasn't applied to your prefill.
7. **`enable_gqa=True`** — without it, dimension mismatch.

These are the real bugs. The math is fine; the bookkeeping is the work.

---

## 9. Adapting to other models

The code above is 95% the same for:

- **Llama 3 / 3.1 / 3.2:** set `bias=False` on q/k/v proj, RoPE base = 5e5, RoPE has by-parts scaling for extended context, untied embeddings on 8B+.
- **Mistral:** like Llama, plus optionally a sliding window mask (`is_causal=False`, custom mask).
- **Gemma 2/3:** RMSNorm with `(1 + γ) · x / rms` (note the `+1`!), `final_logit_softcapping`, alternating local/global attention.
- **DeepSeek-V2/V3 (MLA + MoE):** different attention class (MLA), MoE FFN with routing — see chapter 13.
- **Qwen3:** same template as Qwen2.5 but with `head_dim` decoupled from `D / H_q`, often `head_dim=128` regardless of model size; Qwen3 also has thinking-mode tokens `<think>...</think>`.

If you can build Qwen, you can build any of them with ~20 lines of diff.

---

## 10. Performance tweaks

Your hand-rolled model is slower than HuggingFace's optimized one. Easy wins:

- **`model = torch.compile(model, mode='max-autotune')`** — compile the whole forward graph. ~30-100% speedup, especially on H100.
- **Pre-allocate the KV cache** as a single tensor of `(B, H_kv, max_T, head_dim)` and assign in place rather than `torch.cat` each step.
- **Fuse `q_proj`, `k_proj`, `v_proj` into one `qkv_proj`** (matmuls with bigger M dim are faster).
- **Use `bf16` for everything except norm reductions and the loss.**
- For real production: switch to **vLLM** (chapter 9). Don't try to compete with it in PyTorch.

---

## 11. End-to-end demo

```python
prompt = tok.apply_chat_template(
    [{'role':'user','content':'Write a haiku about transformers.'}],
    tokenize=False, add_generation_prompt=True,
)
ids = tok(prompt, return_tensors='pt').input_ids.to('cuda')
out = generate(model, ids, max_new=80, eos_id=tok.convert_tokens_to_ids('<|im_end|>'))
print(tok.decode(out))
```

Output (one possible sample):

```
Lines of code converge,
Attention shifts through the layers—
Meaning blooms from noise.
```

Congratulations: you wrote an LLM by hand and it produces poetry. Now to train one.

---

## 12. The 2026 cheat sheet

- **Use Qwen 2.5 / Qwen 3 as your reference** when implementing from scratch.
- **Match the architecture exactly** to the source weights — quirks like Qwen's biased QKV are real.
- **Verify against HF's logits** before trusting your code.
- **Pre-allocate the KV cache** for serious inference.
- **`torch.compile`** is free speed.
- **Use vLLM in production** — your hand-rolled model is for *understanding*, not for serving.

---

## Going deeper

- The **Llama-from-scratch** repo (Naveen Garg) — gentle, complete walkthrough, easy to follow.
- HuggingFace's `modeling_qwen2.py` and `modeling_llama.py` — the production reference.
- Andrej Karpathy's **`nanoGPT`** (GPT-2 era) and **`llama2.c`** (Llama era) — minimal, readable, instructive.
- The Qwen 2.5 / Qwen 3 technical reports.

Next: **[12-quantization.md](./12-quantization.md)** — make the model small enough to actually serve.
