# 13 · Mixture of Experts — Scale Without Slowing Down

> **TL;DR** A **Mixture of Experts (MoE)** replaces each FFN with a pool of `E` experts and a tiny **router** that, for each token, picks the top `k` experts to run. Total parameters grow `E×`, but compute per token grows only `k×`. Done right, an MoE with **4×** the parameters of a dense model has the **same inference speed** but matches a much bigger dense model in quality. **In 2026, MoE is the default frontier-model architecture: DeepSeek-V3, Mixtral, Qwen3-MoE, Llama 4, GPT-OSS-120B all use it.**

## 1. Why MoE exists

Dense scaling has a wall: every token uses **every parameter** of the model. To get smarter, you must spend proportionally more compute. MoE breaks this by making the model **sparse**: parameters exist, but only a fraction are used per token.

The deal MoE offers:

- More parameters → more knowledge / capability.
- Same compute per token → same speed.
- Cost: more memory (you store all experts), more complex training, communication overhead in distributed training.

For a **router** that picks 2 of 8 experts (`top-k=2`, `E=8`), an MoE has roughly the same compute as a dense model with **2 experts' worth** of FFN, but stores 8.

---

## 2. Where do you put the experts?

**The FFN.** In every transformer block, the FFN is replaced with an MoE FFN. Attention stays dense (replicated across the model). Why FFN? Because the FFN is where most of the parameters and compute live (`8 D²` per layer), and routing scales cleanly there.

Some research papers MoE-ify attention (`MoA`), but it's not yet a standard.

---

## 3. The simplest MoE forward pass

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
        top_w = top_w.softmax(dim=-1)          # weights for the chosen experts

        out = torch.zeros_like(x_flat)
        for e in range(self.E):
            mask = (top_idx == e).any(dim=-1)  # tokens routing to expert e
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

This is illustrative, not production: the expert calls are unbalanced (small-batched matmuls hurt) and the loops are slow. Real implementations gather tokens by expert into one big batch per expert and run all expert FFNs in parallel via grouped GEMMs (or the "permute → batch matmul → unpermute" pattern).

---

## 4. The router and its options

A **router** decides where each token goes. Common designs:

- **Linear softmax router** (Shazeer 2017, Switch Transformer): a single `nn.Linear(D, E)`, softmax, top-k. Standard.
- **Sigmoid router**: per-expert sigmoid scores; lets multiple experts independently activate. Used by DeepSeek-V3.
- **Cosine router**: cosine similarity between token and expert embeddings. Stabilizes training but rarely a real win.
- **Hash router** (no learning): just hash the token ID. Surprisingly competitive baseline.
- **Expert-choice routing**: experts pick top tokens instead of tokens picking experts. Helps with load balance but breaks left-to-right dependency in autoregressive decoding (don't use for LMs).

Top-1 (Switch Transformer) is the simplest. Top-2 (Mixtral) is the sweet spot for quality. Top-4 to top-8 (DeepSeek-V3 with shared + routed experts) is the high end.

---

## 5. Load balancing: the big training pain

Routers naturally collapse — they like routing all tokens to the same few experts (which become overtrained, getting picked even more). You need a load-balancing pressure.

### Auxiliary loss (Switch Transformer / Mixtral style)

Add to the loss a term that penalizes uneven expert use:

```
fraction_to_expert_e = (1/N) * Σ I[token routed to e]
mean_router_prob_e   = (1/N) * Σ p(e | token)

loss_balance = α · E · Σ_e (fraction_to_expert_e × mean_router_prob_e)
```

`α` ≈ 0.01-0.001. The term is minimized when both are uniform.

### Capacity factor

In implementation, each expert has a **capacity** (max tokens it can handle): `capacity = (BT / E) × cf`, with `cf ≈ 1.25-2.0`. Overflow tokens are dropped (their FFN output is zero, only residual flows). This caps the worst-case batch size on any one expert.

### Auxiliary-loss-free routing (DeepSeek 2024)

DeepSeek-V3 introduced a clever alternative: maintain an EMA of expert load and **add a per-expert bias** to the router scores that pushes traffic toward under-used experts. No auxiliary loss in the gradient — the bias is updated outside autograd. Better quality than aux-loss; **the 2026 frontier favorite**.

```
score_e = router_logit(token, e) + bias_e
top_k chosen by score_e
update: bias_e -= γ if expert e is over-loaded, += γ if under-loaded
```

`γ` is a small step (e.g., `1e-3`).

---

## 6. Shared vs routed experts

DeepSeek-V2/V3 introduced **shared experts**: a few experts (e.g., 1-2) are *always* run for every token, in addition to the top-k routed ones. The reasoning:

- Some computation is generally useful for every token (frequent patterns, basic syntax).
- Letting routed experts specialize on niches works better when a "common" pathway is always available.

DeepSeek-V3 architecture: `1 shared expert + top-8 of 256 fine-grained routed experts`. The 256 experts are smaller than usual (each is "1/8 of a normal FFN"), letting more specialization per parameter. This **fine-grained MoE** style is now standard at the high end.

---

## 7. The economics of MoE

| Model | Total params | Active params (per token) | Quality reference |
|-------|--------------|---------------------------|--------------------|
| Mixtral-8×7B | 47 B | 13 B | ~Llama 2-70B |
| Mixtral-8×22B | 141 B | 39 B | ~Llama 3-70B |
| Qwen3-30B-A3B | 30 B | 3 B | ~Qwen2.5-14B |
| DeepSeek-V3 | 671 B | 37 B | frontier-class |
| Llama 4 Scout (rumored) | ~100 B | ~17 B | ~Llama 3.1-70B |
| GPT-OSS-120B | 120 B | ~5 B | ~Llama 3.1-70B |

The numbers tell a story: MoE buys you **2-5×** quality per active param, **at the cost of memory**. If you can store a 671 B model, DeepSeek-V3 runs at the speed of a 37 B dense and the quality of a hundred-billion-class dense. Magic, given the memory.

---

## 8. The hard parts of MoE in practice

### Memory

You store **all** experts. A 671 B MoE needs 671 B params worth of HBM, even though decode only touches 37 B per token. This is why MoE serving is the cluster's job:

- **Expert parallelism**: shard experts across GPUs (e.g., 8 GPUs each hold 1/8 of the experts).
- **Token routing across GPUs** at every MoE layer: an `all-to-all` collective.
- DeepSeek's open-source `DeepEP` and Megablocks make this fast.

### Communication

Every MoE layer triggers an all-to-all (send tokens to the right GPU) and a reverse all-to-all (collect outputs). On a fast NVLink island this is OK; across nodes, it can dominate runtime. Bigger expert parallelism = more comms. Pick GPUs and topology carefully.

### Imbalance

Even with load-balancing tricks, some experts get hot, some cold. You see GPU utilization swings during training. Tools like `meco`, `dropless MoE` help.

### Inference complexity

vLLM, SGLang, and TensorRT-LLM support MoE. But quantization, KV cache, batching all interact with routing in subtle ways. If you're not running at scale, just use a dense model.

---

## 9. Mixtral / Qwen3-MoE-style implementation sketch

Here is a more realistic implementation using batched expert execution:

```python
class MoEFFN(nn.Module):
    def __init__(self, D, F, E, top_k):
        super().__init__()
        self.E, self.top_k, self.D, self.F = E, top_k, D, F
        self.gate = nn.Linear(D, E, bias=False)
        # Pack all experts' weights as 3D tensors for batched matmul
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

        # gather: each token sent to its top-k experts
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

For real efficiency, replace the inner loops with **`torch.bmm` over a permuted token tensor** (one big batched matmul where each "batch slot" is an expert). Megablocks and the new PyTorch `torch._scaled_mm` + `grouped_gemm` deliver this in hardware-friendly form.

---

## 10. Distillation: the cheaper alternative

Training MoE from scratch is expensive and finicky. A common 2026 path: **train a frontier MoE, then distill its outputs into a small dense model** (Llama 3 used Llama 3.1-405B's outputs to improve smaller siblings). You get most of the quality lift at dense-inference cost.

If you don't have cluster-scale, distillation > rolling your own MoE every time.

---

## 11. Multi-Token Prediction (MTP) — adjacent trick worth knowing

DeepSeek-V3 added a small auxiliary head that predicts not just the next token but also the token after. During training this is a free regularizer; during inference, the predictions of the auxiliary head can be used as **draft tokens for speculative decoding**. Up to 1.8× faster generation with no quality loss. Many open MoE models in 2026 ship with MTP heads.

---

## 12. Should you use MoE?

| You're building... | Use MoE? |
|---------------------|----------|
| A 100M-3B model on a single GPU | **No.** Just use dense. Routing overhead isn't worth it. |
| A 7-30B chat model | **Probably no.** Dense Qwen2.5 / Llama 3 is excellent and simpler. |
| A 70B+ model competing with frontier | **Yes.** MoE is the only way to match frontier dense model quality at acceptable inference cost. |
| Running on a cluster with strong networking | **Yes** if scale demands. |
| Running on edge / single GPU | **No.** Memory is the bottleneck and MoE makes it worse. |

For most readers of this guide: **understand MoE, but build dense**. Pick up MoE when you have the cluster and the data to justify it.

---

## 13. The 2026 cheat sheet

- MoE = top-k routed FFN per layer; total params grow, active params don't.
- **Top-2** is standard; **fine-grained** (small experts, top-8 of 256) is the new best.
- **Shared experts** (always-on) help a lot.
- **Auxiliary-loss-free routing** is the modern load-balancer of choice.
- **Memory is the cost** — MoE needs cluster-scale serving (vLLM + EP).
- **MTP** is a free decode speedup for MoE base models.
- Don't MoE-ify small-scale projects.

---

## Going deeper

- Shazeer et al. 2017 — original MoE paper (top-k softmax router).
- Fedus et al. 2021 — Switch Transformer (top-1 routing, capacity factor).
- Mixtral 8×7B technical report (2023).
- DeepSeek-V2 / V3 papers — fine-grained MoE, shared experts, aux-loss-free routing, MLA, MTP. Dense reading but excellent.
- Megablocks (Stanford, 2022) — the kernel work that made MoE training fast.
- DeepEP — DeepSeek's all-to-all kernel library, open-sourced in 2025.

Next: **[14-training-small-language-models.md](./14-training-small-language-models.md)** — putting it all together.
