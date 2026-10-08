# 08 · Attention Mechanisms — Transformer का Heart

> **TL;DR** Attention एक soft, differentiable lookup है। हर **query** vector decide करता है कि softmax use करते हुए हर **key/value** pair से कितना read करना है। Self-attention layer में हर token query भी है और key/value भी, इसलिए tokens एक-दूसरे से "बात" करते हैं। Multi-head attention कई smaller heads के साथ इसे parallel में करता है। Causal masking tokens को सिर्फ़ past देखने के लिए force करती है। **2026 में, आप `F.scaled_dot_product_attention` (जो hood के नीचे Flash Attention 2/3 use करता है) call करते हो और कभी inner loop खुद नहीं लिखते।**

## 1. Intuition

एक meeting imagine करो जहां सबके पास है:
- एक **question** जिसका वो answer चाहते हैं (`q`)।
- एक **subject जिसमें वो expert हैं** (`k`)।
- एक **answer जो वो दे सकते हैं अगर पूछा जाए** (`v`)।

हर person बाकी सब के "subject" को देखता है, उन्हें weight करता है उसके अपने "question" के relevant होने के basis पर, और उनके "answers" को accordingly weighted combine करता है। यही attention है।

Mathematically, `Q, K, V` matrices के साथ (per token एक row):

```
attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d) ) V
```

- `Q Kᵀ` relevance scores का matrix है: `score[i, j]` = "query `i` को key `j` के बारे में कितना care करना चाहिए?"
- `sqrt(d)` से divide करो scores को sensible range में रखने के लिए (otherwise softmax saturate करता है)।
- Softmax row-wise → weights जो 1 तक sum होते हैं।
- `V` से multiply करो → हर query के लिए values का weighted sum।

बस यही है। बाकी सब window dressing है।

---

## 2. Self vs Cross Attention

- **Self-attention:** `Q`, `K`, `V` **same** sequence से आते हैं। आज हर transformer LLM में used।
- **Cross-attention:** `Q` एक sequence से (decoder), `K, V` दूसरे से (encoder)। Encoder-decoder models (T5, mBART, Whisper) में और image-conditioned text models में used। Modern decoder-only LLMs इसे use नहीं करते।

दोनों same operation हैं; सिर्फ़ QKV का source differ होता है।

---

## 3. Dimension से Multi-head तक

`D` dim के साथ एक single attention head wasteful है — इसे एक जगह हर तरह का relationship encode करना होगा। **Multi-head attention** `D` को `H` heads में split करता है, हर एक `head_dim = D / H` का, और हर एक पर attention separately run करता है, फिर concatenate करता है। हर head specialize कर सकता है।

Shapes:

```
input x:        (B, T, D)
Q, K, V:        (B, T, D)            linear projections के बाद
reshape:        (B, T, H, head_dim)
transpose:      (B, H, T, head_dim)
attention out:  (B, H, T, head_dim)
back-shape:     (B, T, D)
output proj:    (B, T, D)
```

**Heads क्यों help करते हैं:** different heads different patterns track करते हैं (positional, syntactic, semantic, copy-style, etc.)। Empirically, 8-128 heads sweet-spot territory है; 64 या 128 का head_dim Tensor Cores के लिए best है।

---

## 4. Causal Mask

Language modeling के लिए हम next token predict करते हैं, तो position `i` को positions `> i` नहीं देखनी चाहिए। हम ये achieve करते हैं **softmax से पहले attention logits के upper triangle में `-∞` add करके**। Softmax के बाद वो entries zero बन जाती हैं — token `i` future को entirely ignore करता है।

```python
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float('-inf'))
```

Equivalently, `F.scaled_dot_product_attention(..., is_causal=True)` ये kernel में करता है।

Encoder-only models (BERT) में आप कोई mask use नहीं करते। Encoder-decoder में आपके पास सिर्फ़ decoder side पर causal mask है।

---

## 5. Simplest Implementation (इसे ship मत करो)

```python
import torch, torch.nn as nn, math

class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H, self.D = H, D
        self.head_dim = D // H
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)

    def forward(self, x):                                # x: (B, T, D)
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)           # each (B, T, D)

        # split into heads
        q = q.view(B, T, self.H, self.head_dim).transpose(1, 2)  # (B, H, T, d)
        k = k.view(B, T, self.H, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.H, self.head_dim).transpose(1, 2)

        # attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = scores.softmax(dim=-1)
        out = attn @ v                                    # (B, H, T, d)

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.D)
        return self.proj(out)
```

इसे दो बार पढ़ो। अपने head में shapes trace करो। **अगर आप इसे scratch से लिख सकते हो, आप transformers का 80% समझते हो।**

लेकिन — `O(T²)` memory और `O(T²)` compute। `T=8192` के लिए, score matrix per head per sample fp32 में 256 MB है। इसे ship मत करो।

---

## 6. Flash Attention (2026 का Default)

Flash Attention (Dao et al. 2022) और इसके successors (FA2 2023, FA3 2024) **mathematically same operation** हैं लेकिन एक tiled, IO-aware kernel use करते हैं जो:

- Full `(T, T)` matrix HBM में कभी materialize नहीं करता।
- Online softmax use करता है (एक streaming-friendly normalization)।
- Softmax + dropout + masking + matmul को एक kernel में fuse करता है।

Memory `O(T)` बन जाती है। Speed: naive के ऊपर ~2-3×। **Long context (32k+) के लिए, FA only option है।**

आप FA खुद नहीं लिखते। आप call करते हो:

```python
out = torch.nn.functional.scaled_dot_product_attention(
    q, k, v,                          # (B, H, T, head_dim)
    is_causal=True,                   # LM के लिए
    dropout_p=0.0,
    enable_gqa=True,                  # auto-handles GQA (chapter 9)
)
```

PyTorch FA2/FA3 पर dispatch करता है जब shapes और dtypes allow करते हैं। FA backend को force करने के लिए:

```python
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

More flexibility (custom masks, sliding windows, sparse patterns) के लिए, **`torch.nn.attention.flex_attention`** (PyTorch 2.5+) use करो — एक JIT-compiled API जो आपके custom score-modifier को Flash-style kernel में compile करता है:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=T, KV_LEN=T)
out = flex_attention(q, k, v, block_mask=block_mask)
```

Flex Attention एक 2025 superpower है। अगर आपने कभी `+ -1e9 * mask` line लिखी है और memory को blow up होते देखा है, ये आपकी problem solve करता है।

---

## 7. Sliding-window Attention (Mistral-style)

कुछ models attention को past के `W` tokens के window तक bound करते हैं:

```
position i positions [max(0, i - W + 1), i] पर attend करता है
```

Mistral 7B (W=4096), Gemma 2/3 (mixed local + global), Llama 4 long-context layers द्वारा used। Long context पर compute बचाता है, little quality loss के साथ क्योंकि most useful information local है।

Flex Attention में:

```python
def sliding(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx < W)
```

एक common 2025-2026 architecture pattern: **interleave** sliding-window layers with full-context layers (e.g., Gemma 2 1:1 alternating local/global use करता है)। आप quadratic blow-up के across all layers के बिना long-range connectivity पाते हो।

---

## 8. Multi-query (MQA) और Grouped-query (GQA): Preview

Inference memory बचाने के लिए, **MQA** एक shared K/V per layer use करता है (H queries के साथ), और **GQA** एक K/V को `H_q / H_kv` queries के group के बीच share करता है। ये inference के लिए इतना important है कि इसे अपना chapter मिलता है — **[09-kv-cache-mqa-gqa.md](./09-kv-cache-mqa-gqa.md)** देखो।

Code में, बस fewer K/V heads use करो:

```python
self.q_proj = nn.Linear(D, H_q * head_dim, bias=False)
self.k_proj = nn.Linear(D, H_kv * head_dim, bias=False)
self.v_proj = nn.Linear(D, H_kv * head_dim, bias=False)
# H_q = 32, H_kv = 8 का मतलब GQA group size 4 के साथ
```

फिर `F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)` call करो — PyTorch automatically query heads के across K/V tile करता है।

---

## 9. Multi-Head Latent Attention (DeepSeek-V2/V3)

एक 2024 invention जो top-tier models के लिए 2026 architecture standard बन गई है। **MLA** K और V को एक tiny shared latent space में project करता है (`d_kv ≈ 512` regardless of `H`), सिर्फ़ latent store करता है, और K/V on the fly reconstruct करता है:

- **KV cache size** MHA versus ~90% drop करती है।
- Quality MHA पर on par है, often same KV-cache budget पर GQA से better।
- More complex to implement; numerical stability को care चाहिए।

DeepSeek-V2/V3, MiniMax-Text-01 द्वारा used। जानने worth है कि exists; hand-rolled small LLMs के लिए, GQA simpler है। हम chapter 9 में briefly touch करते हैं।

---

## 10. Biases / Temperatures के साथ Attention (optional)

कुछ 2025 models attention के अंदर small wrinkles add करते हैं:

- **QK-norm** (Llama-3.5-Int4, OLMoE, Gemma 3): RoPE के *बाद*, dot product से पहले Q और K को RMSNorm apply करो। Long-context training stabilize करता है।
- **Attention temperature** (e.g., Gemma 2 logit soft-cap): softmax(`x`) को `softmax(soft_cap * tanh(x / soft_cap))` से replace करो। Runaway logits prevent करने के लिए attention scores को cap करता है।
- **Attention sinks** (StreamingLLM): long context पर softmax को anchor करने के लिए हमेशा tokens 0-3 को K/V cache में रखो।

ये standard attention block के top पर 1-2 line tweaks हैं।

---

## 11. अपनी Attention Sanity-check करो

किसी भी attention implementation के लिए correctness test:

```python
B, H, T, d = 2, 4, 8, 16
q = torch.randn(B, H, T, d)
k = torch.randn(B, H, T, d)
v = torch.randn(B, H, T, d)

# reference (eager)
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
ref = (q @ k.transpose(-2, -1) / math.sqrt(d)).masked_fill(mask, float('-inf')).softmax(-1) @ v

fast = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
print(torch.allclose(ref, fast, atol=1e-5))     # should be True
```

ये आपका canary है। अगर refactor इसे break करे, रुक जाओ।

---

## 12. Attention का Compute और Memory Cost

| Resource | Standard (MHA) | Flash | Sliding (W) |
|----------|----------------|-------|-------------|
| Compute | `O(B H T² d)` | `O(B H T² d)` | `O(B H T W d)` |
| Memory (act'n) | `O(B H T²)` | `O(B H T)` | `O(B H T W)` |
| KV cache (inference) | `O(L H T d)` | same | same |

Flash Attention **memory** shrink करता है, compute नहीं। Sliding दोनों shrink करता है — long context पर। MLA **KV cache memory** dramatically shrink करता है।

Inference के लिए, KV cache (chapter 9) often memory को orders of magnitude से dominate करता है। **इसलिए GQA / MQA / MLA exist करते हैं।**

---

## 13. 2026 Cheat Sheet

- training और inference दोनों के लिए **`F.scaled_dot_product_attention` को default करो**।
- LM के लिए **`is_causal=True` use करो**; kernel को mask build करने दो।
- **`enable_gqa=True` use करो** ताकि same kernel MQA/GQA handle करे।
- **FlexAttention use करो** जब आपको sliding windows, document boundaries, ALiBi-like biases, या sparse patterns चाहिए।
- **head_dim = 64 या 128**; `D` `H` से divisible।
- **Attention के अंदर RoPE apply करो**, Q/K linear projections के बाद, dot product से पहले।
- **Inference पर, GQA ratio 4-8** standard है। अगर आपको really long context चाहिए, MLA देखो।
- अगर आप long context पर attention divergence देखते हो, **Soft-cap logits या QK-norm**।

---

## और गहराई से

- Vaswani et al. 2017 — original "Attention Is All You Need" paper।
- Dao et al. 2022 / 2023 / 2024 — Flash Attention, FA2, FA3।
- Karpathy का `nanoGPT` — `model.py` में cleanest hand-rolled attention है जो आप देख सकते हो।
- DeepSeek-V2 paper — MLA का readable derivation।
- PyTorch 2.5+ docs on `torch.nn.attention.flex_attention` — custom attention का future।

Next: **[09-kv-cache-mqa-gqa.md](./09-kv-cache-mqa-gqa.md)** — सबसे ज़्यादा matter करने वाली inference optimizations।
