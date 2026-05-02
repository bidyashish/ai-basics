# 07 · Positional Encodings — RoPE और Modern Alternatives

> **TL;DR** Attention अपने आप **permutation-invariant** है: ये नहीं जानता कि कौन सा token पहले आया। हम इसको queries और keys में position encode करके fix करते हैं। **2026 में, RoPE (Rotary Position Embedding) universal default है**, और जब आप context window extend करना चाहो तो **YaRN / NTK-aware scaling** के साथ। Older schemes (sinusoidal, learned absolute, ALiBi) legacy हैं।

## 1. Positions को क्यों Encoding चाहिए

Position `i` पर query `q` और position `j` पर key `k` के बीच attention score `q · k` है। उस dot product में `i` या `j` पर कुछ depend नहीं करता। अपना sentence shuffle करो — same attention pattern, same output। Model को order जानने में help चाहिए।

तीन families fixes exist करते हैं:

1. **Absolute Positional Encoding (APE)।** हर token के embedding में position-dependent vector add करो। Sinusoidal (vanilla transformer) या learned (GPT-2, BERT)।
2. **Relative Positional Encoding (RPE)।** Tokens के बीच का *distance* directly attention logits में inject करो। ALiBi, T5-bias।
3. **Rotary Position Encoding (RoPE)।** Query और key vectors को उनके position के proportional angle से rotate करो ताकि `(q · k)` सिर्फ़ `i - j` पर depend करे। **2026 में dominant choice।**

---

## 2. Sinusoidal (the OG)

Vanilla transformer (Vaswani et al. 2017) ने use किया:

```
PE(pos, 2k)   = sin(pos / 10000^(2k/D))
PE(pos, 2k+1) = cos(pos / 10000^(2k/D))
```

तो हर dim अलग frequency पर oscillate करता है, uniquely position को identify करता है। पहली layer से पहले token embedding में add होता है। Simple, works, deterministic — लेकिन longer contexts पर well extrapolate नहीं करता और relative नहीं है।

```python
def sinusoidal_pe(T, D):
    pos = torch.arange(T).unsqueeze(1)                # (T, 1)
    div = torch.exp(torch.arange(0, D, 2) * -(math.log(10000.0) / D))  # (D/2,)
    pe = torch.zeros(T, D)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe                                          # (T, D)
```

आप ये production के लिए नहीं लिखोगे, लेकिन ये एक useful exercise है।

---

## 3. Learned Absolute (GPT-2, BERT)

बस एक `nn.Embedding(max_T, D)` table position से indexed। Easy। Trained context window में fine काम करता है, longer lengths पर breaks क्योंकि table के बाहर positions unseen हैं।

```python
self.pos_embed = nn.Embedding(max_T, D)
x = self.tok_embed(ids) + self.pos_embed(torch.arange(T, device=ids.device))
```

GPT-2 इसी पर रहा। GPT-3 भी। Modern models context-length reasons की वजह से आगे बढ़ गए हैं।

---

## 4. ALiBi (Press et al. 2022)

**बिल्कुल कोई positional vectors नहीं।** Instead, query और key के बीच distance से attention logits को bias करो:

```
attention_logit(i, j) = q_i · k_j  -  m * (i - j)     (j ≤ i के लिए)
```

`m` एक head-specific slope है (per head different, geometric progression match करने के लिए picked)। Intuition: nearby tokens को ज़्यादा attend करना चाहिए, distant ones कम। Bias gracefully retrain के बिना training context length से बाहर extrapolate करता है।

ALiBi ~18 months के लिए hot था (BLOOM ने इसे use किया)। ये काम करता है लेकिन rarely अब choose किया जाता है क्योंकि **RoPE YaRN के via equally well generalize करता है, cleaner relative-position math के साथ**।

---

## 5. RoPE: Rotation Trick

**Rotary Position Embedding** (Su et al. 2021) अब ubiquitous है: GPT-NeoX, Llama 1-3, Mistral, Mixtral, Qwen, DeepSeek, Phi, Gemma, ChatGLM — सब इसे use करते हैं।

### The Idea

हर query/key vector के `head_dim` को pairs `(x_0, x_1), (x_2, x_3), …` में group करो। हर pair को 2D vector की तरह treat करो और **उसको angle `θ_d × position` से rotate करो**, जहां `θ_d` एक frequency है जो dim index के साथ decrease करती है।

`q_i` और `k_j` दोनों rotate करने के बाद, उनका dot product सिर्फ़ `i - j` (the *relative* position) पर depend करता है। Math:

```
For a 2D vector (x, y) and angle α:
  R(α) · (x, y) = (x cos α - y sin α,  x sin α + y cos α)

If we rotate q at position i and k at position j by α_i and α_j respectively:
  Rq · Rk  = some function of (α_i - α_j)
```

तो rotated dot product naturally relative distance encode करता है।

### Frequencies

```
θ_d = base^(-2d / D)        for d in [0, D/2)
```

`base = 10000` original choice है। `base = 500_000` (Llama 3) या `base = 1_000_000` (Qwen2.5, long-context configs) effective range extend करता है — long-context models `base` ऊपर tune करते हैं।

Pair index 0 fastest spin करता है (high frequency, fine local position capture करता है)। Pair index `D/2 - 1` slowest spin करता है (low frequency, coarse "where in the document" info capture करता है)।

### Implementation

```python
import math, torch

def precompute_rope(head_dim, max_T, base=10000.0, device='cpu', dtype=torch.float32):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)             # (T, head_dim/2)
    cos, sin = freqs.cos(), freqs.sin()
    return cos.to(dtype), sin.to(dtype)          # each (T, head_dim/2)

def apply_rope(x, cos, sin):
    # x: (B, H, T, head_dim) — head_dim must be even
    # cos, sin: (T, head_dim/2) — broadcast-friendly
    x1 = x[..., 0::2]                            # (B, H, T, head_dim/2)
    x2 = x[..., 1::2]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot_x1
    out[..., 1::2] = rot_x2
    return out
```

ये है RoPE। ~10 lines। आप इसे सिर्फ़ `q` और `k` पर apply करते हो — `v` पर नहीं, residual stream पर नहीं। Attention के अंदर:

```python
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
attn = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
```

### एक Subtle Implementation Detail: Pair Layouts

कुछ implementations pairs को `(x_0, x_1, x_2, x_3, …)` में interleave करते हैं — ऊपर दिखाया गया **interleaved** layout। दूसरे head_dim को आधा split करते हैं — HuggingFace Llama / Qwen द्वारा used **half-rotated** layout:

```python
def apply_rope_half(x, cos, sin):
    # x: (B, H, T, head_dim)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[..., :half]; sin = sin[..., :half]    # adjust shape per impl
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

दोनों equivalent attention scores produce करते हैं **अगर आप same layout के साथ train और infer करो**। Mix करो और आपका model garbage outputs करता है। हमेशा check करो कि आपके loaded weights कौन सा expect करते हैं।

---

## 6. Longer Contexts के लिए RoPE Extend करना

RoPE trained context के *अंदर* nicely generalize करता है। उसके बाद, high-frequency rotations alias करती हैं और quality drops। कई recipes exist करते हैं:

### Position Interpolation (PI)

Pretend करो कि नए positions stretch factor `s` से divided पुराने positions हैं:

```
position_used = position_actual / s
```

तो 4k → 16k extending `s = 4` use करता है। Cheap, light fine-tuning चाहिए।

### NTK-aware Scaling

PI *सारी* frequencies compress करता है। NTK-aware (a.k.a. NTK by parts) instead **base** को scale करता है, जो high-frequency detail preserve करता है और सिर्फ़ low-frequency dims stretch करता है:

```
base_new = base * s^(D / (D - 2))
```

Often zero-shot काम करता है (no fine-tune), some quality cost पर।

### YaRN (Peng et al. 2023)

2026 का favorite। NTK-aware scaling को frequency-dependent ramp के साथ combine करता है: low frequencies को aggressively stretch करो, high frequencies को near-original छोड़ो। एक small attention temperature correction add करो। ~100M tokens fine-tuning के साथ, आप एक model को 8k → 128k+ context तक extend कर सकते हो।

```python
# Llama 3 / Qwen2.5 long context द्वारा used pseudo-config
{
  "rope_scaling": {
    "type": "yarn",
    "factor": 16.0,
    "original_max_position_embeddings": 8192,
    "extrapolation_factor": 1.0,
    "attn_factor": 1.0,
    "beta_fast": 32,
    "beta_slow": 1
  }
}
```

`transformers` library ये apply करती है जब cosine/sine tables build करती है।

### Llama-3-style "by-parts" Scaling

Llama 3.1 ने explicit by-parts piecewise scaling introduce किया: high-freq untouched, low-freq fully PI-scaled, medium-freq smoothly transitioned। Practice में NTK-aware से cleaner। Llama-3.1-8B/70B, Qwen 2.5-7B-1M द्वारा adopted।

### इसे Together रखना

अपने khud के model के लिए, context base करो:

- Compute efficiency के लिए **4k-8k पर train करो**।
- Pretraining के last few B tokens में YaRN-rescaled RoPE के साथ **32k-128k पर anneal करो**।
- 1M context पर inference के लिए (Qwen 2.5-1M, GPT-4.1), **dual-chunk attention** + **YaRN** + एक separate long-context fine-tune apply करो।

---

## 7. NoPE: क्या आपको positions चाहिए भी?

Surprising 2024-2025 finding: एक **causal** transformer बिना किसी positional encoding (NoPE) के अभी भी implicitly causal mask से position सीख सकता है। बहुत small models RoPE के बिना worse करते हैं; बहुत large वाले compensate कर सकते हैं। कुछ research models (TransNormer, RWKV-7, retentive) successfully NoPE use करते हैं।

Practice में, NoPE एक research curiosity है — RoPE essentially compute करने के लिए free है और reliable है। RoPE use करो।

---

## 8. Attention में RoPE — Full Mini-block

```python
import torch, torch.nn as nn, math

class CausalSelfAttention(nn.Module):
    def __init__(self, D, H, max_T):
        super().__init__()
        self.H = H
        self.head_dim = D // H
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        cos, sin = precompute_rope(self.head_dim, max_T, base=500_000.0)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

    def forward(self, x):                                 # x: (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.H, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                       # each (B, T, H, head_dim)
        q = q.transpose(1, 2)                             # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos = self.cos[:T]; sin = self.sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # scaled-dot-product attention with causal mask
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)
```

ये हर modern attention block का 95% है। Extra 5% (KV cache, GQA) हम next chapter में cover करते हैं।

---

## 9. Comparison Cheat Sheet

| Scheme | Year | कहां रहता है | Extrapolation | Used by (2026) |
|--------|------|----------------|---------------|----------------|
| Sinusoidal | 2017 | embeddings में added | OK | almost no one |
| Learned APE | 2018 | embeddings में added | bad | GPT-2/3 (legacy) |
| ALiBi | 2021 | logit bias | great | BLOOM (legacy) |
| **RoPE** | 2021 | Q & K rotate करता है | OK; YaRN के साथ great | **everyone** |
| RoPE + YaRN | 2023 | Q & K rotate करता है, scaled | excellent | Llama 3.1+, Qwen 2.5+, Mistral, DeepSeek |
| NoPE | 2023 | nothing | so-so | research only |

---

## 10. एक 2026 Cheat Sheet

- **RoPE use करो।** Training contexts ≥ 8k के लिए `base = 500k-1M` pick करो। Default `10k` 2k-context relics के लिए है।
- Inference / fine-tune time पर extend करने के लिए **YaRN** use करो। `factor = target_T / train_T` set करो।
- **RoPE सिर्फ़ Q और K पर apply करो**, V पर नहीं, residual stream पर नहीं।
- **Pair layout watch करो** — interleaved vs split-half। आप जो checkpoint load करते हो उसे match करो।
- Extrapolation early test करो: 1k पर train करो, 2k पर eval करो। अगर perplexity collapse करती है, RoPE base too small है।

---

## और गहराई से

- Su et al. 2021 — original RoPE paper. Clean derivation.
- Peng et al. 2023 — YaRN paper.
- "Extending Context Window of LLMs via Position Interpolation," Chen et al. 2023.
- HuggingFace `modeling_llama.py` — `apply_rotary_pos_emb` एक बार पढ़ो और पूरी abstraction collapse हो जाती है।
- NTK-aware scaling पर Eleuther.ai blog post — best intuitive write-up।

Next: **[08-attention-mechanisms.md](./08-attention-mechanisms.md)** — transformer का heart।
