# 07 · Positional Encodings — RoPE और Modern Alternatives

> **TL;DR** Attention अपने आप **permutation-invariant** है: ये नहीं जानता कि कौन सा token पहले आया। हम इसको queries और keys में position encode करके fix करते हैं। **2026 में, RoPE (Rotary Position Embedding) universal default है**, और जब आप context window extend करना चाहो तो **YaRN / NTK-aware scaling** के साथ। Older schemes (sinusoidal, learned absolute, ALiBi) legacy हैं और सिर्फ़ एक paragraph पाते हैं; **कुछ layers पर NoPE** (Llama 4, SmolLM3) एकमात्र live alternative है।

## 1. Positions को क्यों Encoding चाहिए

Position `i` पर query `q` और position `j` पर key `k` के बीच attention score `q · k` है। उस dot product में `i` या `j` पर कुछ depend नहीं करता। अपना sentence shuffle करो — same attention pattern, same output। Model को order जानने में help चाहिए।

तीन families fixes exist करते हैं:

1. **Absolute Positional Encoding (APE)।** हर token के embedding में position-dependent vector add करो। Sinusoidal (vanilla transformer) या learned (GPT-2, BERT)। Legacy।
2. **Relative Positional Encoding (RPE)।** Tokens के बीच का *distance* directly attention logits में inject करो। ALiBi, T5-bias। Legacy।
3. **Rotary Position Encoding (RoPE)।** Query और key vectors को उनके position के proportional angle से rotate करो ताकि `(q · k)` सिर्फ़ `i - j` पर depend करे। **2026 में dominant choice।**

---

## 2. Legacy Schemes, एक Paragraph में

तीन पुराने schemes अभी भी old checkpoints और papers में दिखते हैं। **Sinusoidal** (Vaswani 2017) embeddings में fixed `sin`/`cos` waves add करता था; **learned absolute** (GPT-2, BERT) एक `nn.Embedding(max_T, D)` lookup add करता था; **ALiBi** (BLOOM) position vectors skip करके attention logits से per-head slope गुणा `|i - j|` subtract करता था। इनमें से कोई भी 2026 frontier model में used नहीं है: पहले दो अपनी training length से आगे extend नहीं कर सकते, और ALiBi का extrapolation advantage तब गायब हो गया जब YaRN ने RoPE को cleanly extend करना सिखा दिया। आपको इन्हें implement नहीं करना पड़ेगा; बस legacy code पढ़ते समय recognize करना है।

---

## 3. RoPE: Rotation Trick

**Rotary Position Embedding** (Su et al. 2021) अब ubiquitous है: Llama 3/4, Qwen3 / Qwen3.6, DeepSeek-V3, Gemma 3/4, gpt-oss, Kimi K2 — हर current open model इसे use करता है।

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

`base = 10000` original choice है। `base = 500_000` (Llama 3) या `base = 1_000_000` (Qwen3, long-context configs) effective range extend करता है — long-context models `base` ऊपर tune करते हैं।

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

## 4. Longer Contexts के लिए RoPE Extend करना

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
# config.json snippet जो Qwen3 ship करता है 32k native से 131k context तक जाने के लिए
{
  "rope_scaling": {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768
  }
}
```

`transformers` library ये apply करती है जब cosine/sine tables build करती है; `beta_fast`, `beta_slow`, और `attn_factor` के good defaults हैं जो आप rarely touch करते हो। vLLM और SGLang same keys पढ़ते हैं।

### Llama-3-style "by-parts" Scaling

Llama 3.1 ने explicit by-parts piecewise scaling introduce किया: high-freq untouched, low-freq fully PI-scaled, medium-freq smoothly transitioned। Practice में NTK-aware से cleaner। Llama 3.1 से Llama 4 तक used (`config.json` में `"rope_type": "llama3"`)।

### इसे Together रखना

अपने khud के model के लिए, context base करो:

- Compute efficiency के लिए **4k-8k पर train करो**।
- Pretraining के last few B tokens में YaRN-rescaled RoPE के साथ **32k-128k पर anneal करो**।
- 1M context पर inference के लिए (Qwen3.6: 262k native, YaRN factor 4 से 1M), **YaRN** plus एक separate long-context fine-tune apply करो। Frontier APIs (GPT-6 family 1.05M window serve करती है) ये built in ship करती हैं।

---

## 5. NoPE: क्या आपको positions चाहिए भी?

एक **causal** transformer बिना किसी positional encoding (NoPE) के भी causal mask से implicitly position सीख सकता है। 2026 में ये *per layer* use होता है, model-wide नहीं: Llama 4 RoPE layers के बीच NoPE global-attention layers interleave करता है (iRoPE), और SmolLM3 हर 4th layer में RoPE drop करता है। NoPE layers unseen lengths पर better extrapolate करते हैं; RoPE layers short-range precision रखते हैं।

अपने model के लिए: default में हर जगह RoPE। NoPE सिर्फ़ एक quarter layers पर try करो अगर आप long-context extrapolation chase कर रहे हो और ablation afford कर सकते हो।

---

## 6. Attention में RoPE — Full Mini-block

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

## 7. Comparison Cheat Sheet

| Scheme | Year | कहां रहता है | Extrapolation | Used by (2026) |
|--------|------|----------------|---------------|----------------|
| **RoPE** | 2021 | Q & K rotate करता है | OK; YaRN के साथ great | **हर current open model** |
| RoPE + YaRN | 2023 | Q & K rotate करता है, scaled | excellent | Qwen3 / Qwen3.6, Llama 3.1+, DeepSeek-V3, Gemma 3 |
| NoPE (कुछ layers) | 2023 | nothing | best | Llama 4 iRoPE, SmolLM3 |
| Sinusoidal / learned / ALiBi | 2017-2021 | embeddings या logit bias | poor / poor / good | सिर्फ़ legacy |

---

## 8. एक 2026 Cheat Sheet

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
- Llama 4 release blog (iRoPE) और SmolLM3 blog post — practice में NoPE layers RoPE models में कैसे mix होती हैं।
- NTK-aware scaling पर Eleuther.ai blog post — best intuitive write-up।

Next: **[08-attention-mechanisms.md](./08-attention-mechanisms.md)** — transformer का heart।
