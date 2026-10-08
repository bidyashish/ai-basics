# 10 · Building Blocks — RMSNorm, SwiGLU, और Modern Stack

> **TL;DR** एक modern transformer block है **pre-norm + RMSNorm + RoPE के साथ GQA attention → residual → pre-norm + SwiGLU FFN → residual**। Plus tied या untied embeddings, LM head से पहले एक final RMSNorm, और (optionally) QK-norm। ये template Llama 3/4, Qwen3, DeepSeek-V3, Gemma 3/4, gpt-oss, और basically 2026 में हर open model द्वारा shared है। एकमात्र frontier-only addition **depth recurrence** (looped blocks) है, GPT-6 Astra के लिए reported और §11 में covered।

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

## 3. RMSNorm (एकमात्र norm जो आपको चाहिए)

Attention और FFN दोनों अपने inputs को stable distribution में चाहते हैं। Original answer **LayerNorm** था (center, scale, learned shift)। **RMSNorm** (Zhang & Sennrich 2019) centering और shift drop करता है:

```
RMSNorm(x) = γ · x / sqrt(mean(x²) + ε)
```

Empirically centering कुछ useful नहीं करता; RMSNorm में fewer ops, fewer params हैं, और ये equally well या better train होता है। **हर current LLM RMSNorm use करता है।** LayerNorm सिर्फ़ vision encoders और legacy code में बचा है।

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

PyTorch में built-in `nn.RMSNorm` है। उसे use करो।

एक subtle implementation detail: **हमेशा variance fp32 में compute करो** भले ही आपके activations bf16/fp16 हों। Aggressive training runs पर otherwise आपको NaN मिलेगा।

---

## 4. Pre-norm (plus एक optional second norm)

हर current LLM **pre-norm** है: `x = x + sublayer(RMSNorm(x))`। Norm sublayer से *पहले* बैठता है और residual stream raw रहता है, तो gradients depth के through cleanly flow करते हैं और warmup बहुत कम fragile है। Post-norm (`x = LN(x + sublayer(x))`, original 2017 layout) legacy है।

एकमात्र live variant: Gemma 3/4 और OLMo 2 residual add से पहले **sublayer output पर एक second RMSNorm** add करते हैं (sometimes sandwich norm कहा जाता है)। ये almost कुछ cost नहीं करता और big runs में most loss spikes हटा देता है। Plain pre-norm से start करो; अगर training unstable हो तो second norm add करो।

---

## 5. FFN — और SwiGLU क्यों जीता

हर block में feed-forward network दो linear layers एक non-linearity के साथ है:

```
FFN(x) = W_2 · activation(W_1 · x)
```

`W_1` `D` से `F` को project करता है; `W_2` `F` से वापस `D` को project करता है। 2017 transformer ने ReLU और `F = 4D` use किया; अब कोई नहीं करता।

### GLU Variants

एक **gated linear unit** (Dauphin 2017) FFN को दो parallel projections में split करता है, उन्हें elementwise multiply करता है, और वापस project करता है। तीन projections:

```
SwiGLU(x) = W_3 · ( silu(W_1 · x)  ⊙  (W_2 · x) )
```

`silu(x) = x · sigmoid(x)`, smooth ReLU। `(W_1 · x)` branch `(W_2 · x)` के साथ elementwise multiply से "gated" है।

Empirically (Shazeer 2020, "GLU Variants Improve Transformer"), SwiGLU ReLU FFN को noticeable margin से beat करता है और GeGLU (SiLU के बजाय GELU use करते हुए) को hair से beat करता है।

पुराने `F = 4D` ReLU FFN के साथ parameter count comparable रखने के लिए, आप `F = (4D × 2/3)` set करते हो क्योंकि SwiGLU में 2 के बजाय 3 matrices हैं: `3 × D × F = 2 × D × 4D` `F ≈ 2.67 D` को solves करता है। Real models round up करते हैं: Llama 3-8B `D = 4096` पर `F = 14336` (3.5×) use करता है और Qwen3-8B `F = 12288` (3.0×); 2.67 figure parameter-matched floor है, law नहीं।

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
| SwiGLU | `silu(W₁x) ⊙ W₂x` | **Llama, Qwen3, DeepSeek-V3, gpt-oss, Kimi K2 (default)** |
| GeGLU | `gelu(W₁x) ⊙ W₂x` | Gemma 3/4 |
| SiLU / Swish | `x · sigmoid(x)` | SwiGLU के अंदर gate |

ReLU और GELU FFNs (BERT, GPT-2) legacy हैं; आप इन्हें सिर्फ़ old checkpoints में मिलोगे। 2026 में काम करने वाला default: **`F ≈ 2.67-3.5 D` के साथ SwiGLU FFN**।

---

## 7. QK-norm (Optional Stability Trick)

जब आप bf16 के साथ long context पर train करते हो, attention logits explode कर सकते हैं। Qwen3, OLMo 2, Gemma 3, और most 2025-2026 releases में used fix है **QK-norm**:

```
q = RMSNorm(q)
k = RMSNorm(k)
scores = q @ k.T / sqrt(d)
```

RoPE के बाद, लेकिन matmul से पहले। हर head को अपना RMSNorm मिलता है। Cheap, ~free quality, much more stable।

कुछ implementations `LayerNorm` use करते हैं instead — दोनों काम करते हैं। कुछ per-head normalize करते हैं; दूसरे heads के across shared norm use करते हैं।

अगर आप अपना khud का model train कर रहे हो और long context पर intermittent loss spikes देख रहे हो, **QK-norm add करो**।

---

## 8. Embedding और LM Head

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

## 9. Initialization

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

## 10. Full Small-model Recipe (Architecturally)

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

## 11. Looped / Recurrent-depth Blocks (GPT-6 pattern)

ऊपर सब कुछ हर block को **एक बार** run करता है। एक **looped transformer** (या "recurrent depth") blocks की *same* stack को कई बार run करता है, output को वापस input के रूप में feed करके। Parameters fixed रहते हैं; effective depth multiply होती है।

```
x = prelude(x)                     # कुछ ordinary blocks
for pass in range(n_loops):        # हर pass same weights
    x = core(x)                    # blocks की shared stack
x = coda(x)                        # कुछ ordinary blocks + final norm
```

2026 क्यों care करता है:

- **The Information ने report किया (Sept 2026) कि OpenAI का GPT-6 Astra recurrent depth use करता है।** GPT-6.1 Sol के लिए एक Microsoft Foundry catalog note, जो 6 Oct 2026 को screenshots के रूप में circulate हुआ, कहता है: "GPT-6.1-Sol uses the same base model weights as GPT-6 Sol with two inference passes instead of three." OpenAI ने architecture publish नहीं किया है, तो इसे reported मानो, confirmed नहीं। OpenAI के chief scientist Jakub Pachocki ने September में ज़रूर कहा कि उनके frontier models की compute-graph depth, "including Astra, is within a factor of two of GPT-4."
- **Published open results real हैं।** Geiping et al. 2025 एक 3.5B recurrent-depth model train करते हैं जिसके 4 shared blocks test time पर 4-32 बार unroll हो सकते हैं; Ouro (ByteDance, 2025) 1.4B/2.6B models ship करता है 4 recurrent steps के साथ जो 12B dense models match करते हैं; Nanbeige4.2-3B 22 blocks दो बार run करता है; Mixture-of-Recursions एक router को per token loop count pick करने देता है।
- **आप parameters save करते हो, compute नहीं।** `n_loops × C` blocks का compute और KV cache, सिर्फ़ `C` blocks के weights के साथ। एक looped model store, download, और per parameter train करने में cheaper है; same effective depth के conventional model से serve करने में cheaper *नहीं* है।

इस chapter के `TransformerBlock` के top पर एक minimal implementation:

```python
class LoopedGPT(nn.Module):
    """prelude -> (shared core) x n_loops -> coda, Geiping et al. 2025 style।"""
    def __init__(self, cfg, n_prelude=2, n_core=4, n_coda=2, n_loops=4):
        super().__init__()
        self.embed   = nn.Embedding(cfg.V, cfg.D)
        self.prelude = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_prelude)])
        self.core    = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_core)])
        self.coda    = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_coda)])
        self.inject  = nn.Linear(2 * cfg.D, cfg.D, bias=False)   # हर pass state को input के साथ mix करता है
        self.final_norm = RMSNorm(cfg.D, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.D, cfg.V, bias=False)
        self.n_loops = n_loops

    def forward(self, ids, n_loops=None):
        n_loops = n_loops or self.n_loops          # test time पर ज़्यादा passes = ज़्यादा latent 'thinking'
        e = self.embed(ids)
        for blk in self.prelude:
            e, _ = blk(e)
        s = torch.zeros_like(e)                     # recurrent state
        for _ in range(n_loops):                    # weights reuse होते हैं, activations नहीं
            s = self.inject(torch.cat([s, e], dim=-1))
            for blk in self.core:
                s, _ = blk(s)
        for blk in self.coda:
            s, _ = blk(s)
        return self.lm_head(self.final_norm(s))
```

Papers से training notes: `n_loops` को per batch randomly sample करो (Geiping 32 के around log-normal Poisson use करते हैं) ताकि model किसी भी depth पर काम करे; memory flat रखने के लिए सिर्फ़ last few passes के through backprop करो; एक learned **exit gate** (Ouro, Universal Transformers) add करो अगर आप चाहते हो कि model decide करे कि एक token को कितने passes चाहिए। KV cache: `core` के हर pass को अपनी cache entries चाहिए, तो एक serving engine 3-pass model को `P + 3C + K` layers की तरह treat करता है।

कब use करें: जब आप parameter-bound हो (memory, download size, training data per parameter) और compute spend कर सकते हो। अगर आप serving-cost-bound हो, एक conventional deeper model simpler है और equally fast।

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
- [ ] QK-norm?
- [ ] हर block एक pass, या shared stack कई बार looped?
- [ ] Tied embeddings?
- [ ] Head से पहले final norm?

Almost सारे 2026 open models इस list को इस तरह answer करते हैं: pre, RMS, no, SwiGLU, ~3, 4 या 8, 500k-1M, usually, one pass, often-yes-for-small-models, yes।

Small differences spot करना (Qwen3 vs DeepSeek-V3 vs Llama 4) ज़्यादातर इस template पर hyperparameter shifts हैं।

---

## और गहराई से

- Zhang & Sennrich 2019 — RMSNorm।
- Shazeer 2020 — "GLU Variants Improve Transformer."
- He et al. 2015 — original ResNet, residuals कहां से आते हैं।
- Karpathy का **nanochat** (2025) और HuggingFace का `modeling_qwen3.py` — इस template के लिए easiest readable references।
- Geiping et al. 2025 — "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (arXiv 2502.05171)। सबसे clean recurrent-depth recipe।
- Zhu et al. 2025 — Ouro, "Scaling Latent Reasoning via Looped Language Models" (arXiv 2510.25741); Bae et al. 2025 — Mixture-of-Recursions (arXiv 2507.10524)।
- Sebastian Raschka, "GPT-6 Astra, Looped Transformers, and Hidden Reasoning" (Sept 2026) — https://magazine.sebastianraschka.com/p/gpt-6-astra-looped-transformers-and

Next: **[11-building-qwen-from-scratch.md](./11-building-qwen-from-scratch.md)** — एक full Qwen-style model assembling।
