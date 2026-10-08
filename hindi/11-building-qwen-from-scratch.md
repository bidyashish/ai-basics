# 11 · Qwen को Scratch से Build करना

> **TL;DR** Qwen 2.5 / Qwen 3 RMSNorm, GQA, RoPE, और SwiGLU के साथ pre-norm transformers हैं — chapter 10 से same template, कुछ specific hyperparameter choices के साथ। इस chapter में हम PyTorch के ~250 lines में एक full **Qwen-shaped** model build करते हैं, उसमें real Qwen weights load करते हैं, forward pass run करते हैं, और text generate करते हैं। **कोई `transformers` library tricks नहीं — हर layer आपकी।**

## 1. Reference के रूप में Qwen क्यों

Qwen की तीन useful properties हैं:

1. **Open weights**, permissive license, 0.5B से 235B तक कई sizes में available।
2. **Mainstream architecture** — आप यहां जो सीखते हो वो directly Llama, Mistral, Gemma, DeepSeek-base पर port होता है।
3. **Excellent quality** — probably 2026 में strongest small-model family।

हम implementation reference के रूप में **Qwen2.5-0.5B-Instruct** target करेंगे क्योंकि ये एक laptop CPU पर download और run करने के लिए small enough है। Same code Qwen3-4B के लिए काम करता है, बस bigger config के साथ।

---

## 2. Configuration

Qwen2.5-0.5B का `config.json` (paraphrased, वो fields जो आप actually use करते हो):

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

Qwen2.5-7B के लिए: `D=3584, L=28, H_q=28, H_kv=4, F=18944, V=152064, tie=False`।
Qwen3-4B के लिए: `D=2560, L=36, H_q=32, H_kv=8, F=9728, tie=False`।

---

## 3. PyTorch में Full Model

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

        # apply RoPE — current token range के लिए cos/sin slice करो
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

ये पूरा model है। Roughly 200 lines including config और helpers. HuggingFace के `modeling_qwen2.py` (1500+ lines) से compare करो: theirs कई edge cases handle करता है, लेकिन ऊपर जो *math* है वो identical है।

---

## 4. Real Qwen Weights Load करना

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
# हमारे names:           "layers.0.self_attn.q_proj.weight"
remap = {k.replace('model.', '', 1): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(remap, strict=False)
print('missing:',   missing)
print('unexpected:', unexpected)
```

कुछ चीज़ें जानने worth हैं:

- HuggingFace हर चीज़ को `model.` से prefix करता है, इसलिए हम उसे strip करते हैं। Untied models के लिए `lm_head.weight` top level पर है।
- Safetensors `.bin` से preferred है (no pickle, faster load)।
- Small GPUs पर fit होने और faster run करने के लिए `bfloat16` में cast करो।
- Big models के लिए, `snapshot_download` + `safetensors.torch.load_file` use करो और shards को अपनी layers से map करो।

---

## 5. Tokenization (बस official tokenizer use करो)

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(ckpt)

prompt = tok.apply_chat_template(
    [{'role': 'user', 'content': 'Why is the sky blue?'}],
    tokenize=False, add_generation_prompt=True,
)
ids = tok(prompt, return_tensors='pt').input_ids.to(model.embed_tokens.weight.device)
```

Chat template आपके message को Qwen के special tokens में wrap करता है: `<|im_start|>user ...<|im_end|>\n<|im_start|>assistant\n`।

---

## 6. KV Cache के साथ Greedy Generation

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
        cur = next_id           # next iter only feeds 1 token
    return out

eos = tok.convert_tokens_to_ids('<|im_end|>')
out_ids = generate(model, ids, max_new=200, eos_id=eos)
print(tok.decode(out_ids))
```

First call पूरे prompt के लिए cache fill करता है। Subsequent calls **सिर्फ़ नया token** process करते हैं, context के लिए cache use करते हुए — इसलिए decode fast है।

---

## 7. Temperature, Top-k, Top-p के साथ Sampling

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

`generate` में इसे `argmax` की जगह plug करो। **Chat / open-ended tasks के लिए, sampling greedy से much better है** (less repetition, more interesting outputs)। Defaults जो काम करते हैं: `temperature=0.7, top_p=0.9`।

Reasoning models (Qwen3 "thinking" mode में) के लिए, उनके model card के per `temperature=0.6, top_p=0.95` use करो; chain-of-thought pattern fragile है।

---

## 8. Verify करो कि आपने सही किया

Weights load करने के बाद, fixed prompt के लिए आपके model का first-token output HuggingFace के exactly match करना चाहिए (fp tolerance के अंदर)। Quick check:

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

अगर `cos sim > 0.999` और top-5 match हो, आप correct हो। अगर वो diverge करते हैं, common culprits:

1. **RoPE pair layout** (split-half vs interleaved)। HF match करो।
2. **Qwen के लिए qkv proj पर `bias=True`** लेकिन Llama के लिए **नहीं**। हां, Qwen की Q, K, V पर bias है (एक quirk!)।
3. **`o_proj` के पास कोई bias नहीं है।**
4. **Wrong dtype में compute हुआ norm** (variance के लिए fp32 use करो)।
5. **Wrong RoPE base** (Qwen 1e6 use करता है, Llama 3 5e5 use करता है)।
6. **Causal mask** आपके prefill पर apply नहीं हुआ।
7. **`enable_gqa=True`** — इसके बिना, dimension mismatch।

ये real bugs हैं। Math fine है; bookkeeping work है।

---

## 9. दूसरे Models के लिए Adapt करना

ऊपर का code 95% same है इनके लिए:

- **Llama 3 / 3.1 / 3.2:** q/k/v proj पर `bias=False` set करो, RoPE base = 5e5, RoPE में extended context के लिए by-parts scaling, 8B+ पर untied embeddings।
- **Mistral:** Llama की तरह, plus optionally एक sliding window mask (`is_causal=False`, custom mask)।
- **Gemma 2/3:** RMSNorm `(1 + γ) · x / rms` के साथ (`+1` note करो!), `final_logit_softcapping`, alternating local/global attention।
- **DeepSeek-V2/V3 (MLA + MoE):** different attention class (MLA), MoE FFN routing के साथ — chapter 13 देखो।
- **Qwen3:** Qwen2.5 जैसा same template लेकिन `head_dim` `D / H_q` से decoupled, often model size के regardless `head_dim=128`; Qwen3 के पास thinking-mode tokens `<think>...</think>` भी हैं।

अगर आप Qwen build कर सकते हो, आप ~20 lines diff के साथ इनमें से कोई भी build कर सकते हो।

---

## 10. Performance Tweaks

आपका hand-rolled model HuggingFace के optimized वाले से slower है। Easy wins:

- **`model = torch.compile(model, mode='max-autotune')`** — पूरा forward graph compile करो। ~30-100% speedup, especially H100 पर।
- **Pre-allocate KV cache** as एक single tensor `(B, H_kv, max_T, head_dim)` का और हर step में place में assign करो rather than `torch.cat`।
- **`q_proj`, `k_proj`, `v_proj` को एक `qkv_proj` में fuse करो** (bigger M dim के साथ matmuls faster हैं)।
- **हर चीज़ के लिए `bf16` use करो except norm reductions और loss।**
- Real production के लिए: **vLLM** पर switch करो (chapter 9)। उससे PyTorch में compete करने की कोशिश मत करो।

---

## 11. End-to-end Demo

```python
prompt = tok.apply_chat_template(
    [{'role':'user','content':'Write a haiku about transformers.'}],
    tokenize=False, add_generation_prompt=True,
)
ids = tok(prompt, return_tensors='pt').input_ids.to('cuda')
out = generate(model, ids, max_new=80, eos_id=tok.convert_tokens_to_ids('<|im_end|>'))
print(tok.decode(out))
```

Output (एक possible sample):

```
Lines of code converge,
Attention shifts through the layers—
Meaning blooms from noise.
```

बधाई हो: आपने hand से एक LLM लिखा और ये poetry produce कर रहा है। अब एक train करते हैं।

---

## 12. 2026 Cheat Sheet

- Scratch से implement करते समय **Qwen 2.5 / Qwen 3 को अपने reference के रूप में use करो**।
- Source weights से **architecture exactly match करो** — Qwen का biased QKV जैसी quirks real हैं।
- अपने code पर भरोसा करने से पहले **HF के logits के against verify करो**।
- Serious inference के लिए **KV cache pre-allocate करो**।
- **`torch.compile`** free speed है।
- **Production में vLLM use करो** — आपका hand-rolled model *समझने* के लिए है, *serve* करने के लिए नहीं।

---

## और गहराई से

- **Llama-from-scratch** repo (Naveen Garg) — gentle, complete walkthrough, easy to follow।
- HuggingFace के `modeling_qwen2.py` और `modeling_llama.py` — production reference।
- Andrej Karpathy के **`nanoGPT`** (GPT-2 era) और **`llama2.c`** (Llama era) — minimal, readable, instructive।
- Qwen 2.5 / Qwen 3 technical reports।

Next: **[12-quantization.md](./12-quantization.md)** — model को इतना small बनाओ कि उसे actually serve कर सकें।
