# 10 · Building Blocks — RMSNorm, SwiGLU, और Modern Stack

> **TL;DR** एक modern transformer block है **pre-norm + RMSNorm + RoPE के साथ GQA attention → residual → pre-norm + SwiGLU FFN → residual**। Plus tied या untied embeddings, LM head से पहले एक final RMSNorm, और (optionally) QK-norm। ये template Llama 3, Qwen 3, Mistral, DeepSeek, Gemma, और basically 2026 में हर open model द्वारा shared है।

## 1. Reference Block (जो आप copy करोगे)

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.D, eps=cfg.norm_eps)
        self.attn = GQAttention(cfg.D, cfg.H_q, cfg.H_kv, cfg.head_dim,
                                rope_base=cfg.rope_base, max_T=cfg.max_T)
        self.ffn_norm = RMSNorm(cfg.D, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.D, cfg.F)

    def forward(self, x, past_kv=None):
        a, new_kv = self.attn(self.attn_norm(x), past_kv=past_kv)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv
```

ये पूरा block है। इसके twenty-something layers, plus एक embedding और एक head, और आपके पास एक LLM है।

---

## 2. Residual Connections

हर sub-layer (attention, FFN) एक delta compute करता है, replacement नहीं:

```
x_{l+1} = x_l + sublayer(norm(x_l))
```

ये है **residual stream**. ये very deep networks के लिए एक fix के रूप में start हुआ था (He et al. 2015 ResNet) लेकिन transformers में इसकी एक beautiful interpretation है: **हर sublayer एक shared communication channel से read करता है और अपना update वापस write करता है**। Residual stream की information input embedding से LM head तक flow होती है हर layer अपने contributions push करते हुए।

Practical consequences:

- पहली और last layers most work करती हैं; middle layers ज़्यादा redundant हैं।
- अगर आप एक single residual addition ablate करो, model catastrophically break होता है।
- "Logit lens" — किसी भी layer के residual को LM head से project करना — surprisingly readable predictions देता है।

Residuals को skip मत करो।

---

## 3. LayerNorm vs RMSNorm

Attention और FFN दोनों अपने inputs को stable distribution में चाहते हैं। Classic answer **LayerNorm** है (Ba et al. 2016):

```
LN(x) = γ · (x - μ) / sqrt(σ² + ε)  + β
```

ये हर token के vector को independently center और scale करता है। हर dim per दो learnable params: `γ` (scale) और `β` (shift)।

**RMSNorm** (Zhang & Sennrich 2019) mean centering और bias drop करता है:

```
RMSNorm(x) = γ · x / sqrt(mean(x²) + ε)
```

क्यों? Empirically, centering almost कुछ नहीं करता। RMSNorm में fewer ops, fewer params हैं, और ये equally well या better train होता है। **हर major 2024-2026 LLM RMSNorm use करता है।**

```python
class RMSNorm(nn.Module):
    def __init__(self, D, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))
        self.eps = eps
    def forward(self, x):
        # stable variance के लिए fp32 में cast करो, फिर वापस
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * h).to(x.dtype)
```

PyTorch 2.4+ में built-in `nn.RMSNorm` है। उसे use करो।

एक subtle implementation detail: **हमेशा variance fp32 में compute करो** भले ही आपके activations bf16/fp16 हों। Aggressive training runs पर otherwise आपको NaN मिलेगा।

---

## 4. Pre-norm vs Post-norm

Ordering people के सोचने से ज़्यादा matter करती है।

- **Post-norm** (original Vaswani 2017): `x = LN(x + sublayer(x))`। Residual के *बाद* norm।
- **Pre-norm** (modern LLMs में used): `x = x + sublayer(LN(x))`। Sublayer से *पहले* norm; residual stream raw रहता है।

Pre-norm depth पर more stably train होता है — आपको same fragile तरीके से warmup नहीं चाहिए, और gradients residual path से cleaner flow होते हैं। **हर modern LLM pre-norm है।**

Micro-variants हैं:

- **Sandwich-norm** (कुछ 2024 models): pre-norm और एक extra post-sublayer norm. Very deep nets stabilize करने में help करता है।
- **DeepNorm**: residual पर एक scaling 1000-layer training feasible बनाने के लिए। Niche।

99% cases के लिए, pre-norm + RMSNorm correct है।

---

## 5. FFN — और SwiGLU क्यों जीता

हर block में feed-forward network दो linear layers एक non-linearity के साथ है:

```
FFN(x) = W_2 · activation(W_1 · x)
```

`W_1` `D` से `F` को project करता है; `W_2` `F` से वापस `D` को project करता है। Original transformer ने ReLU और `F = 4D` use किया।

### GLU Variants

एक **gated linear unit** (Dauphin 2017) FFN को दो parallel projections में split करता है, उन्हें elementwise multiply करता है, और वापस project करता है। तीन projections:

```
SwiGLU(x) = W_3 · ( silu(W_1 · x)  ⊙  (W_2 · x) )
```

`silu(x) = x · sigmoid(x)`, smooth ReLU। `(W_1 · x)` branch `(W_2 · x)` के साथ elementwise multiply से "gated" है।

Empirically (Shazeer 2020, "GLU Variants Improve Transformer"), SwiGLU ReLU FFN को noticeable margin से beat करता है और GeGLU (SiLU के बजाय GELU use करते हुए) को hair से beat करता है।

पुराने `F = 4D` ReLU FFN के साथ parameter count comparable रखने के लिए, आप `F = (4D × 2/3)` set करते हो क्योंकि SwiGLU में 2 के बजाय 3 matrices हैं: `3 × D × F = 2 × D × 4D` `F ≈ 2.67 D` को solves करता है। Llama 3-8B में `D = 4096` → `F = 14336`। Almost exactly that ratio.

Implementation:

```python
class SwiGLU(nn.Module):
    def __init__(self, D, F):
        super().__init__()
        self.gate_proj = nn.Linear(D, F, bias=False)
        self.up_proj   = nn.Linear(D, F, bias=False)
        self.down_proj = nn.Linear(F, D, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

कई implementations एक single fused `gate_up_proj: nn.Linear(D, 2*F)` use करते हैं और split करते हैं, जो slightly faster है।

`F.silu` `x * sigmoid(x)` है। आप इसे `nn.SiLU()` या `swish` भी call कर सकते हो — same function।

---

## 6. Activation Choices: एक Quick Map

| Activation | Formula | में used |
|------------|---------|---------|
| ReLU | `max(0, x)` | old transformers, some MoE |
| GELU | `x · Φ(x)` | BERT, GPT-2, T5 |
| SiLU / Swish | `x · sigmoid(x)` | SwiGLU के अंदर gate; dense models भी |
| GeGLU | `gelu(W₁x) ⊙ W₂x` | PaLM, some Gemma variants |
| SwiGLU | `silu(W₁x) ⊙ W₂x` | **Llama, Qwen, Mistral, DeepSeek (default)** |
| ReGLU | `relu(W₁x) ⊙ W₂x` | rare |

2026 में काम करने वाले defaults: **`F ≈ 2.67 D` के साथ SwiGLU FFN**।

---

## 7. QK-norm (Optional Stability Trick)

जब आप bf16 के साथ long context पर train करते हो, attention logits explode कर सकते हैं। OLMoE, Gemma 3, Llama 3.5-Int4, और काफी research models में used fix है **QK-norm**:

```
q = RMSNorm(q)
k = RMSNorm(k)
scores = q @ k.T / sqrt(d)
```

RoPE के बाद, लेकिन matmul से पहले। हर head को अपना RMSNorm मिलता है। Cheap, ~free quality, much more stable।

कुछ implementations `LayerNorm` use करते हैं instead — दोनों काम करते हैं। कुछ per-head normalize करते हैं; दूसरे heads के across shared norm use करते हैं।

अगर आप अपना khud का model train कर रहे हो और long context पर intermittent loss spikes देख रहे हो, **QK-norm add करो**।

---

## 8. Logit Soft-cap

Gemma 2 द्वारा final logits को sensible range में रखने के लिए used:

```
logits = soft_cap × tanh(logits / soft_cap)        # soft_cap ≈ 30
```

Same trick sometimes attention scores के अंदर apply होता है। ये एक old training instability के लिए fix था और जब QK-norm present हो तो अब essential नहीं है। Mostly historical।

---

## 9. Embedding और LM Head

सब कुछ model level पर together रखना:

```python
class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.V, cfg.D)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.L)])
        self.final_norm = RMSNorm(cfg.D, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.D, cfg.V, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, ids, kv_cache=None):
        x = self.embed(ids)
        new_cache = []
        for i, layer in enumerate(self.layers):
            past = kv_cache[i] if kv_cache else None
            x, new_kv = layer(x, past)
            new_cache.append(new_kv)
        x = self.final_norm(x)
        return self.lm_head(x), new_cache
```

कुछ details:

- Head से पहले **final RMSNorm** — हर modern model के पास है। उसके बिना, last layer पर gradients badly behave करते हैं।
- Small models (≤1B) के लिए **tied embeddings**, larger के लिए untied।
- Bias हर जगह `False` है — biases add करना modern models को slightly hurt करता है और कुछ params बचाता है।

---

## 10. Initialization

वो सारे weights किस numbers पर start होते हैं?

- **Embeddings:** `nn.init.normal_(std=0.02)`। GPT-2 से standard।
- **Linear weights** (general): `std = 0.02` fine काम करता है। कुछ teams residual projections (attention का `o_proj` और FFN का `down_proj`) पर `1/sqrt(2 × L)` से scale करते हैं depth के साथ activation variance constant रखने के लिए — GPT-2 / nanoGPT trick।
- **Norm weights** (`γ`): `1.0`।
- **Biases**: कोई नहीं।

Pseudocode:

```python
def init_weights(self):
    for name, p in self.named_parameters():
        if 'embed' in name:
            nn.init.normal_(p, mean=0, std=0.02)
        elif p.dim() == 2 and 'norm' not in name:
            nn.init.normal_(p, mean=0, std=0.02)
            if 'o_proj' in name or 'down_proj' in name:
                p.data /= math.sqrt(2 * len(self.layers))
        elif 'norm' in name:
            nn.init.ones_(p)
```

---

## 11. Full Small-model Recipe (Architecturally)

एक **canonical 2026 small LLM** एक specification में:

```python
@dataclass
class Config:
    V:        int = 50_257     # vocab size
    D:        int = 1_024      # model dim
    L:        int = 24         # layers
    H_q:      int = 16         # query heads
    H_kv:     int = 4          # KV heads (GQA group size 4)
    head_dim: int = 64
    F:        int = 2_752      # FFN inner dim, ≈ 2.67 D
    max_T:    int = 4_096
    rope_base:float = 500_000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
```

हर component:

1. Token embedding `(V, D)`।
2. `L` blocks, हर एक:
   - RMSNorm
   - RoPE के साथ GQA attention
   - RMSNorm
   - SwiGLU FFN
3. Final RMSNorm।
4. LM head (embedding से tied)।

बस यही। Training (chapter 14), data (chapter 4), tokenization (chapter 6) add करो, और आपके पास working LLM है। हम इसे **[chapter 11](./11-building-qwen-from-scratch.md)** में glue करते हैं।

---

## 12. दूसरों के Models का Code पढ़ते समय Checklist

जब आप किसी और का transformer देखो (HF, Llama, Qwen), check करो:

- [ ] Pre-norm या post-norm?
- [ ] LayerNorm या RMSNorm?
- [ ] Linears पर bias?
- [ ] FFN: ReLU / GeGLU / SwiGLU?
- [ ] FFN intermediate ratio (F / D)?
- [ ] GQA group size?
- [ ] RoPE base?
- [ ] QK-norm? logit soft-cap?
- [ ] Tied embeddings?
- [ ] Head से पहले final norm?

Almost सारे 2026 models इस list को इस तरह answer करते हैं: pre, RMS, no, SwiGLU, ~2.67, 4 या 8, 500k-1M, sometimes, often-yes-for-small-models, yes।

Small differences spot करना (Qwen 2.5 vs DeepSeek vs Llama 3) ज़्यादातर इस template पर hyperparameter shifts हैं।

---

## और गहराई से

- Zhang & Sennrich 2019 — RMSNorm।
- Shazeer 2020 — "GLU Variants Improve Transformer."
- He et al. 2015 — original ResNet, residuals कहां से आते हैं।
- Karpathy का nanoGPT और Llama-from-scratch tutorials — इस template के लिए easiest references।

Next: **[11-building-qwen-from-scratch.md](./11-building-qwen-from-scratch.md)** — एक full Qwen-style model assembling।
