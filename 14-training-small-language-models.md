# 14 · Training Small Language Models — The Complete Pipeline

> **TL;DR** A modern small-LLM run is: **pre-tokenized binary shards** streamed via `IterableDataset` → **bf16 model with `torch.compile`** → **AdamW or Muon** optimizer → **FSDP2** for multi-GPU memory → **cosine schedule with warmup and anneal phase** → **eval every N steps** on perplexity + downstream benchmarks → **checkpoint to disk frequently** → after pretraining, **SFT on chat data** → **DPO / KTO / RLVR** for preference & reasoning. **W&B logs everything** so you can debug from charts.

---

## 1. The end-to-end shape

```
[clean text data]
   ↓ tokenize, shard, pack
[binary token shards on S3 / disk]
   ↓ stream
[DataLoader (B, T)]
   ↓ forward
[model, bf16]
   ↓ cross-entropy
[loss]
   ↓ backward
[gradients]
   ↓ clip, scale
[AdamW step]
   ↓ schedule
[next step]
   ⤴
   periodically:
   [validation perplexity, downstream benchmarks, checkpoint]
```

After `~10⁹-10¹³` such loops, you have a base model. Then:

```
base model → SFT (instruction tuning) → DPO/KTO (preference) → optional RLVR (verifiable rewards) → final model
```

We walk through each piece.

---

## 2. Choose the size

- **125M-300M**: laptop / single-GPU training. Trains in hours-days. Useful for *learning*, not for production.
- **500M-1.5B**: single-node (8 GPUs) training in 1-2 weeks. Great target for a personal project; matches 2023's 7B in many tasks.
- **3B-8B**: cluster training, ~64-256 H100 for 2-4 weeks. Can match early Llama-3 territory.
- **30B+**: only if you have access to a cluster with hundreds of H100/B200 and a few months.

This chapter focuses on the **0.5B-3B** band — the sweet spot for hobby / lab / startup work.

---

## 3. Final architecture choice (for our worked example)

```python
@dataclass
class Config:
    V:        int = 49_152      # tokenizer-dependent
    D:        int = 1_536
    L:        int = 28
    H_q:      int = 12
    H_kv:     int = 4
    head_dim: int = 128
    F:        int = 4_096
    max_T:    int = 4_096
    rope_base:float = 500_000.0
    norm_eps: float = 1e-6
    tie_emb:  bool = True
    dropout:  float = 0.0       # modern LLMs: dropout 0
```

≈ 1.1 B params. Architecture as in chapter 11.

---

## 4. Data pipeline

Use a **pre-tokenized**, **pre-packed** binary corpus per chapter 4.

```python
class TokenStream(IterableDataset):
    def __init__(self, shard_glob, block_size):
        self.shards = sorted(glob.glob(shard_glob))
        self.block_size = block_size

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rank = info.id if info else 0
        nworkers = info.num_workers if info else 1
        random.seed(42 + rank)
        my_shards = self.shards[rank::nworkers]
        random.shuffle(my_shards)
        for s in my_shards:
            arr = np.load(s, mmap_mode='r')
            n = (len(arr) - 1) // self.block_size
            for i in np.random.permutation(n):
                a = i * self.block_size
                x = torch.from_numpy(arr[a:a+self.block_size].astype(np.int64))
                y = torch.from_numpy(arr[a+1:a+1+self.block_size].astype(np.int64))
                yield x, y
```

Wrap in DataLoader with `num_workers=4-8`, `pin_memory=True`. For multi-GPU, each rank reads disjoint shards.

For huge datasets, prefer **`mosaicml-streaming`** — handles S3, sharding, deterministic shuffling, mid-epoch resume.

---

## 5. Training step

```python
def train_step(model, batch, opt, scaler, grad_clip=1.0):
    x, y = batch
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    opt.zero_grad(set_to_none=True)
    loss.backward()                          # bf16 doesn't need GradScaler

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return loss.item(), grad_norm.item()
```

Notes:
- bf16 for activations; **fp32 for the optimizer state** (AdamW default).
- Gradient clipping at `1.0` is the LLM standard. Saves runs from rare spikes.
- `set_to_none=True` is faster than zero-fill.

---

## 6. Optimizer choices

### AdamW (the default for ~5 years)

```python
opt = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
    fused=True,                  # PyTorch 2.x — single CUDA kernel
)
```

`weight_decay = 0.1` is the LLM standard (note: weight decay should NOT apply to embeddings, biases, or norm weights — apply via `param_groups`).

### Muon (the 2024-2026 newcomer, especially for matmul layers)

Muon (Jordan et al. 2024) is a momentum optimizer that, instead of Adam-style per-parameter scaling, applies an **orthogonalization** to the update of each weight matrix. Empirically:

- Trains **2× faster** in compute on dense LLM pretraining.
- Better than AdamW on big runs, comparable on small.
- Used by **Moonshot/Kimi K2** (the first frontier-scale Muon-trained model, 2024-2025) and adopted by several open-source recipes in 2026.

```python
from muon import Muon

# Muon for hidden matmul weights, AdamW for everything else (embed, head, norm)
muon_params = [p for n, p in model.named_parameters() if p.ndim == 2 and 'embed' not in n and 'head' not in n]
adamw_params = [p for n, p in model.named_parameters() if p.ndim != 2 or 'embed' in n or 'head' in n]
muon = Muon(muon_params, lr=0.02, momentum=0.95)
adamw = torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
```

Worth experimenting with on a research run. The free version (`KellerJordan/Muon` on GitHub) is solid.

### Soap, Lion, Sophia, etc.

Various 2nd-order-ish optimizers exist. **Soap** (2024) shows promise in recent papers. Mostly research-only as of 2026; AdamW + Muon cover production.

### Optimizer state in 8-bit

`bitsandbytes.optim.AdamW8bit` cuts optimizer memory ~4×. Used in QLoRA fine-tuning, sometimes pretraining. Tiny quality cost.

---

## 7. Learning-rate schedule

The 2026 standard for pretraining: **warmup → cosine decay**, often with a final **constant or linear-anneal** "anneal phase" on extra-clean data.

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

WARMUP = 2_000
TOTAL  = 200_000

warmup = LinearLR(opt, start_factor=1e-5, end_factor=1.0, total_iters=WARMUP)
cosine = CosineAnnealingLR(opt, T_max=TOTAL - WARMUP, eta_min=3e-5)
sched = SequentialLR(opt, [warmup, cosine], milestones=[WARMUP])
```

**Anneal phase** (Llama-3 / SmolLM-2 style):
- Last 5-10% of pretraining tokens.
- Replace cosine with a linear decay to `eta_min ≈ 0`.
- Use a high-quality data mix (curated educational, math, code).
- Often gives a 1-3% benchmark bump.

```python
# After cosine phase, manually drop LR linearly to ~0 while feeding clean data
```

**WSD** (warmup-stable-decay, Hu et al. 2024) is an even simpler alternative gaining popularity in 2026: warmup → flat → fast linear decay at the end. Lets you stop training "early" without throwing away progress.

---

## 8. Multi-GPU: DDP, FSDP, ZeRO

You almost always need multiple GPUs. Pick a strategy by model size.

### DDP (DistributedDataParallel)

Simple replication: every GPU holds the full model, data is split, gradients are all-reduced after backward. Works when full model fits on each GPU.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
dist.init_process_group(backend='nccl')
model = MyModel().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])
```

### FSDP2 (Fully Sharded Data Parallel, the 2026 default)

Shards parameters, gradients, and optimizer state across GPUs. Like ZeRO Stage 3 but native to PyTorch and more efficient since FSDP2 (PyTorch 2.4+).

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for layer in model.layers:
    fully_shard(layer, mp_policy=mp)
fully_shard(model, mp_policy=mp)
```

**FSDP2 is what Llama 3 / Qwen 3 / OLMo / SmolLM are trained with.** It's now stable and well-supported.

### Tensor parallelism (TP)

Shard each weight matrix across GPUs; each GPU computes a slice of the matmul, all-reduce to combine. Used for very wide layers (FFN, attention output) when even FSDP isn't enough. Megatron-LM and `torch.distributed.tensor` support this.

For sub-7B models on one node: **DDP or FSDP2.** For 7-70B multi-node: **FSDP2 + maybe TP for the FFN.** For frontier MoE: **EP + TP + PP** (expert + tensor + pipeline parallelism). You won't write the latter; you'll use Megatron, NeMo, or Colossal-AI.

---

## 9. Activation checkpointing

Trades compute for memory. During forward, don't store all intermediate activations; during backward, recompute them.

```python
from torch.utils.checkpoint import checkpoint
def forward(x):
    for layer in self.layers:
        x = checkpoint(layer, x, use_reentrant=False)
    return self.norm(x)
```

Cost: ~30% extra compute. Benefit: 5-10× less activation memory. Standard for any training where you'd otherwise OOM.

---

## 10. Gradient accumulation

If you want effective batch size B but only B' fits, accumulate gradients across B/B' micro-batches before stepping:

```python
ACCUM = 4
for i, batch in enumerate(loader):
    loss = compute_loss(model, batch)
    (loss / ACCUM).backward()
    if (i + 1) % ACCUM == 0:
        clip_grad(model)
        opt.step(); sched.step(); opt.zero_grad()
```

In FSDP2, wrap with `model.set_requires_gradient_sync(False)` for non-final micro-steps to skip the all-reduce — doubles throughput.

---

## 11. Effective batch size and tokens-per-step

Scaling laws talk about tokens, not steps. Useful identity:

```
tokens_per_step = effective_batch_size × sequence_length
                = B_per_GPU × N_GPUs × accumulation × T

total_tokens   = tokens_per_step × num_steps
```

A common 1B-class run:

```
B = 4 per GPU  ×  64 GPUs  ×  accum 8  ×  T 4096  =  ~8.4 M tokens / step
1.5 T tokens  →  ~180k steps
```

---

## 12. Checkpointing and resuming

```python
def save(step, model, opt, sched, scaler, path):
    ck = dict(
        step=step,
        model=model.state_dict(),
        opt=opt.state_dict(),
        sched=sched.state_dict(),
        scaler=scaler.state_dict(),
    )
    torch.save(ck, path)
```

Save every ~1000 steps. Use **safetensors** for the final model; pickled `.pt` for in-progress checkpoints (faster, includes optimizer state).

For FSDP2, use `torch.distributed.checkpoint` — it shards the checkpoint files across ranks for fast multi-GPU saves/loads.

---

## 13. Evaluation during pretraining

What to log:

- **Train loss** every step.
- **Validation loss** every 500-2000 steps on a held-out shard.
- **Perplexity** on diverse domains (web, code, math) — quick to compute.
- **Downstream tasks** every 5-10k steps:
  - **MMLU** (knowledge), **HellaSwag** (commonsense), **PIQA** (physical reasoning), **ARC** (reasoning), **GSM8K** (math), **HumanEval** (code).
  - Use **lm-eval-harness** (EleutherAI, the standard).
  - These are noisy at small scale; smooth across multiple checkpoints.

```bash
lm_eval --model hf --model_args pretrained=./checkpoint-100000 \
        --tasks mmlu,hellaswag,arc_easy,arc_challenge,gsm8k \
        --batch_size auto
```

If your model rate of improvement on these stalls long before training tokens budget is reached, your data is the problem.

---

## 14. Logging — what and why (W&B)

```python
import wandb
wandb.init(project='small-llm', config=cfg)

# every step
wandb.log({
    'train/loss': loss,
    'train/grad_norm': gnorm,
    'train/lr': sched.get_last_lr()[0],
    'train/tokens_per_sec': tps,
}, step=step)

# every eval
wandb.log({'eval/loss': eval_loss, 'eval/ppl': math.exp(eval_loss)}, step=step)
```

Detailed W&B reading is in **[15-reading-training-logs.md](./15-reading-training-logs.md)** — chartchasing is half the work.

---

## 15. From base to chat: SFT

After pretraining, the base model **completes** text but doesn't follow instructions. We **supervised fine-tune** (SFT) on instruction-response pairs.

```python
# data: list of {messages: [{role, content}, ...]}
def make_examples(tokenizer, msgs):
    text = tokenizer.apply_chat_template(msgs, tokenize=False)
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=4096)
    ids = enc.input_ids[0]
    # labels: train on assistant tokens only — mask user/system with -100
    labels = ids.clone()
    # use the chat template's known offsets to find assistant ranges, set others to -100
    return ids, labels
```

Datasets: Tülu 3 SFT mix, OpenHermes-2.5, Llama-Nemotron post-training, your own. Train **1-3 epochs** with **LR 2e-5 to 1e-5** (much lower than pretraining), no weight decay on the new data, AdamW or Muon. The full stack is otherwise the same.

LoRA / QLoRA SFT is also extremely common — small, cheap, and quality is close to full fine-tuning. Use `peft + trl + accelerate`:

```python
from trl import SFTTrainer, SFTConfig
cfg = SFTConfig(output_dir='out', per_device_train_batch_size=2,
                gradient_accumulation_steps=8, num_train_epochs=2,
                learning_rate=2e-5, bf16=True, packing=True, max_seq_length=4096)
trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg, tokenizer=tok)
trainer.train()
```

---

## 16. Preference / alignment: DPO, KTO, RLVR

After SFT, the model can chat but doesn't necessarily prefer good answers over mediocre ones. Three common 2026 paths:

### DPO (Direct Preference Optimization)

Train on `(prompt, chosen, rejected)` triples. The model learns to assign higher probability to chosen than rejected without an explicit reward model.

```python
from trl import DPOTrainer, DPOConfig
cfg = DPOConfig(output_dir='out-dpo', beta=0.1, learning_rate=5e-7,
                per_device_train_batch_size=2, num_train_epochs=1,
                bf16=True, max_length=4096)
DPOTrainer(model=model, ref_model=ref, args=cfg, train_dataset=ds, tokenizer=tok).train()
```

Datasets: UltraFeedback, HelpSteer3, your own preference data. **DPO is the default in 2026 for chat alignment.**

### KTO / IPO / SimPO

Variants of DPO with different loss formulations. **KTO** uses unpaired good/bad ratings (no need for both at once); **SimPO** drops the reference model entirely. Try them if DPO underperforms.

### RLVR (Reinforcement Learning with Verifiable Rewards)

For **reasoning models** (math, code, logic): train with PPO/GRPO using a programmatic reward (does the output run? does it equal the answer?). DeepSeek-R1 popularized this in early 2025; OpenR1, Tülu 3 with verifiable rewards, AceMath, etc., followed.

```python
# Pseudocode — frameworks: trl GRPOTrainer, OpenR1
prompts = [...]                                # math problems
for batch in loader:
    samples = model.generate(batch.prompts, n=8)        # 8 candidate answers each
    rewards = check_correctness(samples, batch.answers) # 0/1 from a verifier
    grpo_step(model, samples, rewards)                  # group-relative policy optimization
```

GRPO (DeepSeek) is the most efficient algorithm; use **`trl.GRPOTrainer`** as a starting point. Expect very long training runs (RL is sample-inefficient).

---

## 17. Distillation (cheap quality lift)

If a strong teacher model exists, train a small student to **match the teacher's logits** (KL loss) instead of predicting from one-hot labels. Often a 30%+ improvement at no extra data cost.

```python
with torch.no_grad():
    teacher_logits = teacher(x).logits / T
student_logits = student(x).logits / T
loss = F.kl_div(F.log_softmax(student_logits, -1),
                F.softmax(teacher_logits, -1), reduction='batchmean')
```

Cosmopedia, Llama 3 distillation, Phi family — all exploit this.

---

## 18. Final eval and shipping

Before declaring victory:

1. **Comprehensive lm-eval-harness suite** with all relevant benchmarks.
2. **Manual chat tests** — a sanity test set of 50-100 queries, judged by you or another LLM.
3. **Latency / throughput benchmarks** with vLLM at target batch sizes.
4. **Quantize** (chapter 12) to AWQ-int4 / FP8.
5. **Compare quantized model** against bf16 to confirm <1% quality drop.
6. **Decontamination report** — confirm benchmarks weren't leaked.

For shipping: write a **model card** (intended use, license, eval results, limitations) and publish to HuggingFace.

---

## 19. Common training failures

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Loss diverges in first 1k steps | LR too high, init weird | Lower LR, longer warmup |
| Loss plateaus immediately | Tokenizer / data bug | Print 5 batches, sanity check |
| Gradual NaN at long sequences | bf16 attention overflow | Add QK-norm, lower LR |
| Validation goes up while train goes down | Overfit | More data, fewer epochs |
| GPU util < 40% | Data loader bottleneck | More workers, simpler pipeline |
| Loss "saw-tooths" up and down | Bad data shard, high grad noise | Shuffle better, lower LR |
| Sudden spike in train loss | Bad batch (often spam) | Skip-on-spike or filter data |
| Model says "yes I am Claude" | Forgot to filter benchmark / leaked SFT data | Decontaminate again |

Most of these are visible in the W&B charts — see the next file.

---

## 20. The 2026 cheat sheet

- **bf16 + AdamW + cosine + 1-2k warmup + WSD or anneal.**
- **FSDP2** for multi-GPU, **DDP** if model fits on one.
- **`torch.compile(model)`** — free 30-100% speedup.
- **Activation checkpointing** to save memory.
- **Gradient clip 1.0**, **weight decay 0.1**, **`betas=(0.9, 0.95)`**, **`eps=1e-8`**.
- **Tokenize once, shard once, stream forever.**
- **Data anneal phase** in last 5-10% with curated data.
- **Eval with lm-eval-harness** every 5k steps.
- **W&B logs** for everything.
- **SFT + DPO** for chat. **GRPO/RLVR** for reasoning.
- **Quantize before shipping.**
- **vLLM** to serve.

---

## Going deeper

- **`nanoGPT`** (Karpathy) — the smallest readable production-grade trainer. Read first.
- **`llm.c`** (Karpathy) — pure C/CUDA training, the educational gold standard for performance.
- **OLMo** (AI2) and **SmolLM-2** (HuggingFace) — fully open recipes, including data and code.
- **DeepSeek-V3 technical report** — best modern frontier training writeup.
- **Tülu 3 technical report** (AI2 2024) — the canonical post-training pipeline.
- **`torchtitan`** — PyTorch's reference distributed-training library (FSDP2 + TP + activation ckpt).

Next: **[15-reading-training-logs.md](./15-reading-training-logs.md)** — making sense of W&B charts.
