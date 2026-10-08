# 13 · Mixture of Experts — Slow हुए बिना Scale करो

> **TL;DR** एक **Mixture of Experts (MoE)** हर FFN को `E` experts के pool और एक tiny **router** से replace करता है जो हर token के लिए top `k` experts पick करता है। Total parameters `E×` grow करते हैं, लेकिन per token compute सिर्फ़ `k×` grow होता है। सही से किया जाए तो, dense model से **4×** parameters वाला MoE **same inference speed** रखता है लेकिन quality में much bigger dense model match करता है। **2026 में, MoE default frontier-model architecture है: DeepSeek-V3, Mixtral, Qwen3-MoE, Llama 4, GPT-OSS-120B सब इसे use करते हैं।**

## 1. MoE क्यों Exist करता है

Dense scaling की एक wall है: हर token model के **हर parameter** को use करता है। Smarter होने के लिए, आपको proportionally more compute spend करना है। MoE इसको break करता है parameters exist तो हैं लेकिन per token सिर्फ़ एक fraction use होता है।

MoE जो deal offer करता है:

- More parameters → more knowledge / capability।
- Per token same compute → same speed।
- Cost: more memory (आप सारे experts store करते हो), more complex training, distributed training में communication overhead।

8 में से 2 experts pick करने वाले **router** के लिए (`top-k=2`, `E=8`), MoE roughly **2 experts के worth FFN** के साथ dense model जैसा compute रखता है, लेकिन 8 store करता है।

---

## 2. Experts कहां Place करें?

**FFN।** हर transformer block में, FFN को MoE FFN से replace किया जाता है। Attention dense रहती है (model के across replicated)। FFN क्यों? क्योंकि FFN वो है जहां ज़्यादातर parameters और compute रहते हैं (per layer `8 D²`), और routing वहां cleanly scale होता है।

कुछ research papers attention को MoE-ify करते हैं (`MoA`), लेकिन ये अभी standard नहीं है।

---

## 3. Simplest MoE Forward Pass

```python
class MoEFFN(nn.Module):
    def __init__(self, D, F, E, top_k):
        super().__init__()
        self.E, self.top_k = E, top_k
        self.gate = nn.Linear(D, E, bias=False)
        self.experts = nn.ModuleList([SwiGLU(D, F) for _ in range(E)])

    def forward(self, x):                     # x: (B, T, D)
        B, T, D = x.shape
        x_flat = x.view(-1, D)                # (BT, D)

        scores = self.gate(x_flat)             # (BT, E)
        top_w, top_idx = scores.topk(self.top_k, dim=-1)
        top_w = top_w.softmax(dim=-1)          # chosen experts के लिए weights

        out = torch.zeros_like(x_flat)
        for e in range(self.E):
            mask = (top_idx == e).any(dim=-1)  # expert e को route करने वाले tokens
            if not mask.any(): continue
            tokens_e = x_flat[mask]
            y_e = self.experts[e](tokens_e)
            # weighted contribution
            for k in range(self.top_k):
                hits = (top_idx[:, k] == e) & mask
                if hits.any():
                    out[hits] += top_w[hits, k:k+1] * self.experts[e](x_flat[hits])
        return out.view(B, T, D)
```

ये illustrative है, production नहीं: expert calls unbalanced हैं (small-batched matmuls hurt करते हैं) और loops slow हैं। Real implementations tokens को expert के द्वारा एक big batch per expert में gather करते हैं और सारे expert FFNs को grouped GEMMs के via parallel run करते हैं (या "permute → batch matmul → unpermute" pattern)।

---

## 4. Router और Options

एक **router** decide करता है हर token कहां जाए। Common designs:

- **Linear softmax router** (Shazeer 2017, Switch Transformer): एक single `nn.Linear(D, E)`, softmax, top-k। Standard।
- **Sigmoid router**: per-expert sigmoid scores; multiple experts को independently activate होने देता है। DeepSeek-V3 द्वारा used।
- **Cosine router**: token और expert embeddings के बीच cosine similarity। Training stabilize करता है लेकिन rarely real win।
- **Hash router** (no learning): बस token ID hash करो। Surprisingly competitive baseline।
- **Expert-choice routing**: experts top tokens pick करते हैं instead of tokens experts pick करना। Load balance में help करता है लेकिन autoregressive decoding में left-to-right dependency break करता है (LMs के लिए use मत करो)।

Top-1 (Switch Transformer) simplest है। Top-2 (Mixtral) quality के लिए sweet spot है। Top-4 to top-8 (DeepSeek-V3 shared + routed experts के साथ) high end है।

---

## 5. Load Balancing: Big Training Pain

Routers naturally collapse करते हैं — वो सारे tokens को same few experts पर route करना like करते हैं (जो overtrained हो जाते हैं, और भी ज़्यादा pick होते हैं)। आपको load-balancing pressure चाहिए।

### Auxiliary Loss (Switch Transformer / Mixtral style)

Loss में एक term add करो जो uneven expert use को penalize करे:

```
fraction_to_expert_e = (1/N) * Σ I[token routed to e]
mean_router_prob_e   = (1/N) * Σ p(e | token)

loss_balance = α · E · Σ_e (fraction_to_expert_e × mean_router_prob_e)
```

`α` ≈ 0.01-0.001। Term minimized होता है जब दोनों uniform हों।

### Capacity Factor

Implementation में, हर expert की एक **capacity** होती है (max tokens जो वो handle कर सकता है): `capacity = (BT / E) × cf`, `cf ≈ 1.25-2.0` के साथ। Overflow tokens drop हो जाते हैं (उनका FFN output zero है, सिर्फ़ residual flow करता है)। ये किसी एक expert पर worst-case batch size cap करता है।

### Auxiliary-loss-free Routing (DeepSeek 2024)

DeepSeek-V3 ने एक clever alternative introduce किया: expert load का EMA maintain करो और router scores में एक **per-expert bias** add करो जो traffic को under-used experts की ओर push करे। Gradient में कोई auxiliary loss नहीं — bias autograd के बाहर update होता है। aux-loss से better quality; **2026 frontier favorite**।

```
score_e = router_logit(token, e) + bias_e
score_e द्वारा chosen top_k
update: bias_e -= γ अगर expert e over-loaded है, += γ अगर under-loaded
```

`γ` एक small step है (e.g., `1e-3`)।

---

## 6. Shared vs Routed Experts

DeepSeek-V2/V3 ने **shared experts** introduce किए: कुछ experts (e.g., 1-2) हर token के लिए *हमेशा* run होते हैं, top-k routed वालों के अलावा। Reasoning:

- कुछ computation हर token के लिए generally useful है (frequent patterns, basic syntax)।
- Routed experts को niches पर specialize होने देना better काम करता है जब एक "common" pathway हमेशा available हो।

DeepSeek-V3 architecture: `1 shared expert + top-8 of 256 fine-grained routed experts`। 256 experts usual से smaller हैं (हर एक "normal FFN का 1/8" है), letting more specialization per parameter। ये **fine-grained MoE** style अब high end पर standard है।

---

## 7. MoE की Economics

| Model | Total params | Active params (per token) | Quality reference |
|-------|--------------|---------------------------|--------------------|
| Mixtral-8×7B | 47 B | 13 B | ~Llama 2-70B |
| Mixtral-8×22B | 141 B | 39 B | ~Llama 3-70B |
| Qwen3-30B-A3B | 30 B | 3 B | ~Qwen2.5-14B |
| DeepSeek-V3 | 671 B | 37 B | frontier-class |
| Llama 4 Scout (rumored) | ~100 B | ~17 B | ~Llama 3.1-70B |
| GPT-OSS-120B | 120 B | ~5 B | ~Llama 3.1-70B |

Numbers एक story बताते हैं: MoE आपको per active param **2-5×** quality देता है, **memory की cost पर**। अगर आप 671 B model store कर सकते हो, DeepSeek-V3 37 B dense की speed पर run होता है और hundred-billion-class dense की quality पर। Memory दिए जाने पर magic।

---

## 8. Practice में MoE के Hard Parts

### Memory

आप **सारे** experts store करते हो। एक 671 B MoE को 671 B params worth HBM चाहिए, even though decode per token सिर्फ़ 37 B touch करता है। इसलिए MoE serving cluster का काम है:

- **Expert parallelism**: experts को GPUs के across shard करो (e.g., 8 GPUs प्रत्येक experts का 1/8 hold करते हैं)।
- हर MoE layer पर **GPUs के across token routing**: एक `all-to-all` collective।
- DeepSeek का open-source `DeepEP` और Megablocks इसे fast बनाते हैं।

### Communication

हर MoE layer एक all-to-all (tokens को सही GPU पर send करो) और एक reverse all-to-all (outputs collect करो) trigger करता है। Fast NVLink island पर ये OK है; nodes के across, ये runtime को dominate कर सकता है। Bigger expert parallelism = more comms। GPUs और topology carefully pick करो।

### Imbalance

Load-balancing tricks के साथ भी, कुछ experts hot हो जाते हैं, कुछ cold। आप training के दौरान GPU utilization swings देखते हो। `meco`, `dropless MoE` जैसे tools help करते हैं।

### Inference Complexity

vLLM, SGLang, और TensorRT-LLM MoE को support करते हैं। लेकिन quantization, KV cache, batching सब routing के साथ subtle ways में interact करते हैं। अगर आप scale पर नहीं चला रहे, बस dense model use करो।

---

## 9. Mixtral / Qwen3-MoE-style Implementation Sketch

यहां batched expert execution use करते हुए एक more realistic implementation:

```python
class MoEFFN(nn.Module):
    def __init__(self, D, F, E, top_k):
        super().__init__()
        self.E, self.top_k, self.D, self.F = E, top_k, D, F
        self.gate = nn.Linear(D, E, bias=False)
        # सारे experts के weights को batched matmul के लिए 3D tensors के रूप में pack करो
        self.w1 = nn.Parameter(torch.empty(E, D, F))    # gate_proj
        self.w3 = nn.Parameter(torch.empty(E, D, F))    # up_proj
        self.w2 = nn.Parameter(torch.empty(E, F, D))    # down_proj
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w3, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(-1, D)                              # (N, D)  N = B*T
        scores = self.gate(x)                          # (N, E)
        top_w, top_idx = scores.topk(self.top_k, dim=-1)
        top_w = F.softmax(top_w, dim=-1)               # (N, k)

        # gather: हर token अपने top-k experts पर भेजा गया
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e in range(self.E):
                mask = (top_idx[:, k] == e)
                if not mask.any(): continue
                xe = x[mask]                            # (n_e, D)
                gate = xe @ self.w1[e]
                up   = xe @ self.w3[e]
                h    = F.silu(gate) * up
                out_e = h @ self.w2[e]
                out[mask] += top_w[mask, k:k+1] * out_e
        return out.view(B, T, D)
```

Real efficiency के लिए, inner loops को **`torch.bmm` over a permuted token tensor** से replace करो (एक big batched matmul जहां हर "batch slot" एक expert है)। Megablocks और नया PyTorch `torch._scaled_mm` + `grouped_gemm` ये hardware-friendly form में deliver करते हैं।

---

## 10. Distillation: Cheaper Alternative

Scratch से MoE train करना expensive और finicky है। एक common 2026 path: **एक frontier MoE train करो, फिर उसके outputs को एक small dense model में distill करो** (Llama 3 ने Llama 3.1-405B के outputs use किए छोटे siblings को improve करने के लिए)। आप dense-inference cost पर most quality lift पाते हो।

अगर आपके पास cluster-scale नहीं है, distillation > अपना MoE roll करना every time।

---

## 11. Multi-Token Prediction (MTP) — जानने worth Adjacent Trick

DeepSeek-V3 ने एक small auxiliary head add किया जो सिर्फ़ next token नहीं predict करता बल्कि उसके बाद वाला token भी (and sometimes one more)। Training के दौरान ये free regularizer है; inference के दौरान, auxiliary head की predictions **speculative decoding के लिए draft tokens** के रूप में use हो सकती हैं। 1.8× तक faster generation no quality loss के साथ। 2026 में कई open MoE models MTP heads के साथ ship करते हैं।

---

## 12. क्या आपको MoE Use करना चाहिए?

| आप build कर रहे हैं... | MoE use करें? |
|---------------------|----------|
| Single GPU पर 100M-3B model | **नहीं।** बस dense use करो। Routing overhead worth नहीं है। |
| 7-30B chat model | **Probably no।** Dense Qwen2.5 / Llama 3 excellent है और simpler। |
| 70B+ frontier-competing model | **हां।** MoE only way है frontier dense model quality match करने का acceptable inference cost पर। |
| Strong networking वाले cluster पर running | **हां** अगर scale demand करे। |
| Edge / single GPU पर running | **नहीं।** Memory bottleneck है और MoE उसे worse बनाता है। |

इस guide के ज़्यादातर readers के लिए: **MoE को समझो, लेकिन dense बनाओ**। MoE उठाओ जब आपके पास cluster और data हो उसे justify करने के लिए।

---

## 13. 2026 Cheat Sheet

- MoE = per layer top-k routed FFN; total params grow करते हैं, active params नहीं।
- **Top-2** standard है; **fine-grained** (small experts, top-8 of 256) नया best है।
- **Shared experts** (always-on) बहुत help करते हैं।
- **Auxiliary-loss-free routing** modern load-balancer of choice है।
- **Memory cost है** — MoE को cluster-scale serving (vLLM + EP) चाहिए।
- **MTP** MoE base models के लिए free decode speedup है।
- Small-scale projects को MoE-ify मत करो।

---

## और गहराई से

- Shazeer et al. 2017 — original MoE paper (top-k softmax router)।
- Fedus et al. 2021 — Switch Transformer (top-1 routing, capacity factor)।
- Mixtral 8×7B technical report (2023)।
- DeepSeek-V2 / V3 papers — fine-grained MoE, shared experts, aux-loss-free routing, MLA, MTP. Dense reading लेकिन excellent।
- Megablocks (Stanford, 2022) — kernel work जिसने MoE training को fast बनाया।
- DeepEP — DeepSeek की all-to-all kernel library, 2025 में open-sourced।

Next: **[14-training-small-language-models.md](./14-training-small-language-models.md)** — सब कुछ together रखना।
