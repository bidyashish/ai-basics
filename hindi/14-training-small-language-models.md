# 14 · Small Language Models Train करना — Complete Pipeline

> **TL;DR** एक modern small-LLM run है: **pre-tokenized binary shards** `IterableDataset` के via streamed → **bf16 model `torch.compile` के साथ** → **AdamW या Muon** optimizer → multi-GPU memory के लिए **FSDP2** → **warmup और anneal phase के साथ cosine schedule** → perplexity + downstream benchmarks पर हर N steps **eval** → frequent **checkpoint to disk** → pretraining के बाद, chat data पर **SFT** → preference और reasoning के लिए **DPO / KTO / RLVR**। **W&B सब कुछ log करता है** ताकि आप charts से debug कर सको।

---

## 1. End-to-end Shape

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

`~10⁹-10¹³` ऐसे loops के बाद, आपके पास base model है। फिर:

```
base model → SFT (instruction tuning) → DPO/KTO (preference) → optional RLVR (verifiable rewards) → final model
```

हम हर piece walk through करते हैं।

---

## 2. Size Choose करो

- **125M-300M**: laptop / single-GPU training. Hours-days में train होता है। *सीखने* के लिए useful, production के लिए नहीं।
- **500M-1.5B**: single-node (8 GPUs) training 1-2 weeks में। Personal project के लिए great target; कई tasks में 2023's 7B match करता है।
- **3B-8B**: cluster training, ~64-256 H100 2-4 weeks के लिए। Early Llama-3 territory match कर सकता है।
- **30B+**: सिर्फ़ अगर आपके पास hundreds of H100/B200 के साथ cluster access और कुछ months हों।

ये chapter **0.5B-3B** band पर focus करता है — hobby / lab / startup work के लिए sweet spot।

---

## 3. Final Architecture Choice (हमारे worked example के लिए)

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

≈ 1.1 B params। Architecture chapter 11 के अनुसार।

---

## 4. Data Pipeline

Chapter 4 के per **pre-tokenized**, **pre-packed** binary corpus use करो।

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

`num_workers=4-8`, `pin_memory=True` के साथ DataLoader में wrap करो। Multi-GPU के लिए, हर rank disjoint shards read करता है।

Huge datasets के लिए, **`mosaicml-streaming`** prefer करो — S3, sharding, deterministic shuffling, mid-epoch resume handle करता है।

---

## 5. Training Step

```python
def train_step(model, batch, opt, scaler, grad_clip=1.0):
    x, y = batch
    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    opt.zero_grad(set_to_none=True)
    loss.backward()                          # bf16 को GradScaler नहीं चाहिए

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return loss.item(), grad_norm.item()
```

Notes:
- Activations के लिए bf16; **optimizer state के लिए fp32** (AdamW default)।
- `1.0` पर gradient clipping LLM standard है। Rare spikes से runs बचाता है।
- `set_to_none=True` zero-fill से faster है।

---

## 6. Optimizer Choices

### AdamW (~5 साल के लिए default)

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

`weight_decay = 0.1` LLM standard है (note: weight decay embeddings, biases, या norm weights पर apply नहीं होना चाहिए — `param_groups` के via apply करो)।

### Muon (2024-2026 newcomer, especially matmul layers के लिए)

Muon (Jordan et al. 2024) एक momentum optimizer है जो, Adam-style per-parameter scaling के बजाय, हर weight matrix के update पर एक **orthogonalization** apply करता है। Empirically:

- Dense LLM pretraining पर compute में **2× faster** train होता है।
- Big runs पर AdamW से better, small पर comparable।
- **Moonshot/Kimi K2** द्वारा used (पहला frontier-scale Muon-trained model, 2024-2025) और 2026 में कई open-source recipes ने adopt किया।

```python
from muon import Muon

# hidden matmul weights के लिए Muon, बाकी सब (embed, head, norm) के लिए AdamW
muon_params = [p for n, p in model.named_parameters() if p.ndim == 2 and 'embed' not in n and 'head' not in n]
adamw_params = [p for n, p in model.named_parameters() if p.ndim != 2 or 'embed' in n or 'head' in n]
muon = Muon(muon_params, lr=0.02, momentum=0.95)
adamw = torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
```

Research run पर experiment करने worth है। Free version (GitHub पर `KellerJordan/Muon`) solid है।

### Soap, Lion, Sophia, etc.

Various 2nd-order-ish optimizers exist करते हैं। **Soap** (2024) recent papers में promise दिखाता है। 2026 तक mostly research-only; AdamW + Muon production cover करते हैं।

### 8-bit में Optimizer State

`bitsandbytes.optim.AdamW8bit` optimizer memory ~4× cut करता है। QLoRA fine-tuning में used, sometimes pretraining। Tiny quality cost।

---

## 7. Learning-rate Schedule

Pretraining के लिए 2026 standard: **warmup → cosine decay**, often एक final **constant या linear-anneal** "anneal phase" extra-clean data पर।

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

WARMUP = 2_000
TOTAL  = 200_000

warmup = LinearLR(opt, start_factor=1e-5, end_factor=1.0, total_iters=WARMUP)
cosine = CosineAnnealingLR(opt, T_max=TOTAL - WARMUP, eta_min=3e-5)
sched = SequentialLR(opt, [warmup, cosine], milestones=[WARMUP])
```

**Anneal phase** (Llama-3 / SmolLM-2 style):
- Pretraining tokens का last 5-10%।
- Cosine को linear decay से replace करो `eta_min ≈ 0` तक।
- एक high-quality data mix use करो (curated educational, math, code)।
- Often 1-3% benchmark bump देता है।

```python
# cosine phase के बाद, LR को manually linearly ~0 तक drop करो clean data feed करते हुए
```

**WSD** (warmup-stable-decay, Hu et al. 2024) एक even simpler alternative है जो 2026 में popularity gain कर रहा है: warmup → flat → end पर fast linear decay। आपको progress throw किए बिना training "early" stop करने देता है।

---

## 8. Multi-GPU: DDP, FSDP, ZeRO

आपको almost हमेशा multiple GPUs चाहिए। Model size से एक strategy pick करो।

### DDP (DistributedDataParallel)

Simple replication: हर GPU full model hold करता है, data split होता है, backward के बाद gradients all-reduced होते हैं। तब काम करता है जब full model हर GPU पर fit हो।

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
dist.init_process_group(backend='nccl')
model = MyModel().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])
```

### FSDP2 (Fully Sharded Data Parallel, 2026 का default)

Parameters, gradients, और optimizer state को GPUs के across shard करता है। ZeRO Stage 3 की तरह लेकिन PyTorch के लिए native और FSDP2 (PyTorch 2.4+) से ज़्यादा efficient।

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for layer in model.layers:
    fully_shard(layer, mp_policy=mp)
fully_shard(model, mp_policy=mp)
```

**FSDP2 वो है जिससे Llama 3 / Qwen 3 / OLMo / SmolLM trained हैं।** ये अब stable और well-supported है।

### Tensor Parallelism (TP)

हर weight matrix को GPUs के across shard करो; हर GPU matmul का slice compute करता है, combine करने के लिए all-reduce। बहुत wide layers (FFN, attention output) के लिए used जब FSDP भी काफी न हो। Megatron-LM और `torch.distributed.tensor` ये support करते हैं।

एक node पर sub-7B models के लिए: **DDP या FSDP2।** 7-70B multi-node के लिए: **FSDP2 + maybe TP for FFN।** Frontier MoE के लिए: **EP + TP + PP** (expert + tensor + pipeline parallelism)। आप latter नहीं लिखेंगे; आप Megatron, NeMo, या Colossal-AI use करोगे।

---

## 9. Activation Checkpointing

Memory के लिए compute trade करता है। Forward के दौरान, सारे intermediate activations store मत करो; backward के दौरान, उन्हें recompute करो।

```python
from torch.utils.checkpoint import checkpoint
def forward(x):
    for layer in self.layers:
        x = checkpoint(layer, x, use_reentrant=False)
    return self.norm(x)
```

Cost: ~30% extra compute। Benefit: 5-10× less activation memory। किसी भी training के लिए standard जहां आप otherwise OOM होते।

---

## 10. Gradient Accumulation

अगर आप effective batch size B चाहते हो लेकिन सिर्फ़ B' fit होता है, stepping से पहले B/B' micro-batches के across gradients accumulate करो:

```python
ACCUM = 4
for i, batch in enumerate(loader):
    loss = compute_loss(model, batch)
    (loss / ACCUM).backward()
    if (i + 1) % ACCUM == 0:
        clip_grad(model)
        opt.step(); sched.step(); opt.zero_grad()
```

FSDP2 में, non-final micro-steps के लिए `model.set_requires_gradient_sync(False)` से wrap करो all-reduce skip करने के लिए — throughput double करता है।

---

## 11. Effective Batch Size और Tokens-per-step

Scaling laws tokens की बात करते हैं, steps की नहीं। Useful identity:

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

## 12. Checkpointing और Resuming

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

हर ~1000 steps save करो। Final model के लिए **safetensors** use करो; in-progress checkpoints के लिए pickled `.pt` (faster, optimizer state include करता है)।

FSDP2 के लिए, `torch.distributed.checkpoint` use करो — ये fast multi-GPU saves/loads के लिए ranks के across checkpoint files shard करता है।

---

## 13. Pretraining के दौरान Evaluation

क्या log करना है:

- **Train loss** हर step।
- **Validation loss** हर 500-2000 steps held-out shard पर।
- **Perplexity** diverse domains (web, code, math) पर — compute करने के लिए quick।
- **Downstream tasks** हर 5-10k steps:
  - **MMLU** (knowledge), **HellaSwag** (commonsense), **PIQA** (physical reasoning), **ARC** (reasoning), **GSM8K** (math), **HumanEval** (code)।
  - **lm-eval-harness** (EleutherAI, standard) use करो।
  - ये small scale पर noisy हैं; multiple checkpoints के across smooth करो।

```bash
lm_eval --model hf --model_args pretrained=./checkpoint-100000 \
        --tasks mmlu,hellaswag,arc_easy,arc_challenge,gsm8k \
        --batch_size auto
```

अगर इन पर improvement की आपकी rate stalls long before training tokens budget reach होते, आपका data problem है।

---

## 14. Logging — क्या और क्यों (W&B)

```python
import wandb
wandb.init(project='small-llm', config=cfg)

# हर step
wandb.log({
    'train/loss': loss,
    'train/grad_norm': gnorm,
    'train/lr': sched.get_last_lr()[0],
    'train/tokens_per_sec': tps,
}, step=step)

# हर eval
wandb.log({'eval/loss': eval_loss, 'eval/ppl': math.exp(eval_loss)}, step=step)
```

Detailed W&B reading **[15-reading-training-logs.md](./15-reading-training-logs.md)** में है — chartchasing आधा work है।

---

## 15. Base से Chat तक: SFT

Pretraining के बाद, base model text **complete** करता है लेकिन instructions follow नहीं करता। हम instruction-response pairs पर **supervised fine-tune** (SFT) करते हैं।

```python
# data: list of {messages: [{role, content}, ...]}
def make_examples(tokenizer, msgs):
    text = tokenizer.apply_chat_template(msgs, tokenize=False)
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=4096)
    ids = enc.input_ids[0]
    # labels: assistant tokens पर ही train करो — user/system को -100 से mask करो
    labels = ids.clone()
    # chat template के known offsets use करो assistant ranges find करने के लिए, बाकी को -100 set करो
    return ids, labels
```

Datasets: Tülu 3 SFT mix, OpenHermes-2.5, Llama-Nemotron post-training, अपना khud का। **1-3 epochs** के लिए train करो **LR 2e-5 to 1e-5** (pretraining से much lower) के साथ, new data पर no weight decay, AdamW या Muon। बाकी stack same है।

LoRA / QLoRA SFT भी extremely common है — small, cheap, और quality full fine-tuning के close है। `peft + trl + accelerate` use करो:

```python
from trl import SFTTrainer, SFTConfig
cfg = SFTConfig(output_dir='out', per_device_train_batch_size=2,
                gradient_accumulation_steps=8, num_train_epochs=2,
                learning_rate=2e-5, bf16=True, packing=True, max_seq_length=4096)
trainer = SFTTrainer(model=model, train_dataset=ds, args=cfg, tokenizer=tok)
trainer.train()
```

---

## 16. Preference / Alignment: DPO, KTO, RLVR

SFT के बाद, model chat कर सकता है लेकिन necessarily good answers को mediocre से prefer नहीं करता। तीन common 2026 paths:

### DPO (Direct Preference Optimization)

`(prompt, chosen, rejected)` triples पर train करो। Model explicit reward model के बिना chosen को rejected से higher probability assign करना सीखता है।

```python
from trl import DPOTrainer, DPOConfig
cfg = DPOConfig(output_dir='out-dpo', beta=0.1, learning_rate=5e-7,
                per_device_train_batch_size=2, num_train_epochs=1,
                bf16=True, max_length=4096)
DPOTrainer(model=model, ref_model=ref, args=cfg, train_dataset=ds, tokenizer=tok).train()
```

Datasets: UltraFeedback, HelpSteer3, अपना preference data। **DPO 2026 में chat alignment के लिए default है।**

### KTO / IPO / SimPO

DPO के variants different loss formulations के साथ। **KTO** unpaired good/bad ratings use करता है (एक साथ दोनों की need नहीं); **SimPO** reference model entirely drop कर देता है। अगर DPO underperform करे तो उन्हें try करो।

### RLVR (Reinforcement Learning with Verifiable Rewards)

**Reasoning models** (math, code, logic) के लिए: programmatic reward (क्या output run होता है? क्या ये answer के equal है?) के साथ PPO/GRPO use करते हुए train करो। DeepSeek-R1 ने इसे early 2025 में popularize किया; OpenR1, Tülu 3 with verifiable rewards, AceMath, etc., followed।

```python
# Pseudocode — frameworks: trl GRPOTrainer, OpenR1
prompts = [...]                                # math problems
for batch in loader:
    samples = model.generate(batch.prompts, n=8)        # हर एक के 8 candidate answers
    rewards = check_correctness(samples, batch.answers) # verifier से 0/1
    grpo_step(model, samples, rewards)                  # group-relative policy optimization
```

GRPO (DeepSeek) most efficient algorithm है; starting point के रूप में **`trl.GRPOTrainer`** use करो। Very long training runs expect करो (RL sample-inefficient है)।

---

## 17. Distillation (Cheap Quality Lift)

अगर एक strong teacher model exists, एक small student को teacher के logits **match करना** (KL loss) train करो instead of one-hot labels से predict करने का। Often no extra data cost पर 30%+ improvement।

```python
with torch.no_grad():
    teacher_logits = teacher(x).logits / T
student_logits = student(x).logits / T
loss = F.kl_div(F.log_softmax(student_logits, -1),
                F.softmax(teacher_logits, -1), reduction='batchmean')
```

Cosmopedia, Llama 3 distillation, Phi family — सब इसको exploit करते हैं।

---

## 18. Final Eval और Shipping

Victory declare करने से पहले:

1. सारे relevant benchmarks के साथ **Comprehensive lm-eval-harness suite**।
2. **Manual chat tests** — 50-100 queries का sanity test set, आपके या किसी और LLM द्वारा judged।
3. Target batch sizes पर vLLM के साथ **Latency / throughput benchmarks**।
4. AWQ-int4 / FP8 तक **Quantize** (chapter 12)।
5. <1% quality drop confirm करने के लिए bf16 के against **quantized model compare** करो।
6. **Decontamination report** — confirm करो कि benchmarks leak नहीं हुए।

Shipping के लिए: एक **model card** लिखो (intended use, license, eval results, limitations) और HuggingFace पर publish करो।

---

## 19. Common Training Failures

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| First 1k steps में loss diverges | LR too high, init weird | LR lower, longer warmup |
| Loss immediately plateaus | Tokenizer / data bug | 5 batches print करो, sanity check |
| Long sequences पर gradual NaN | bf16 attention overflow | QK-norm add करो, lower LR |
| Validation up जाता है while train down जाता है | Overfit | More data, fewer epochs |
| GPU util < 40% | Data loader bottleneck | More workers, simpler pipeline |
| Loss "saw-tooths" up और down | Bad data shard, high grad noise | Better shuffle करो, lower LR |
| Train loss में sudden spike | Bad batch (often spam) | Skip-on-spike या data filter करो |
| Model कहता है "yes I am Claude" | Benchmark / leaked SFT data filter करना भूल गए | Decontaminate again करो |

इनमें से ज़्यादातर W&B charts में visible हैं — अगली file देखो।

---

## 20. 2026 Cheat Sheet

- **bf16 + AdamW + cosine + 1-2k warmup + WSD या anneal।**
- multi-GPU के लिए **FSDP2**, अगर model एक पर fit हो तो **DDP**।
- **`torch.compile(model)`** — free 30-100% speedup।
- Memory बचाने के लिए **Activation checkpointing**।
- **Gradient clip 1.0**, **weight decay 0.1**, **`betas=(0.9, 0.95)`**, **`eps=1e-8`**।
- **एक बार tokenize करो, एक बार shard करो, forever stream करो।**
- Last 5-10% में curated data के साथ **Data anneal phase**।
- हर 5k steps **lm-eval-harness के साथ Eval**।
- हर चीज़ के लिए **W&B logs**।
- Chat के लिए **SFT + DPO**। Reasoning के लिए **GRPO/RLVR**।
- Shipping से पहले **Quantize करो**।
- Serve करने के लिए **vLLM**।

---

## और गहराई से

- **`nanoGPT`** (Karpathy) — smallest readable production-grade trainer। पहले पढ़ो।
- **`llm.c`** (Karpathy) — pure C/CUDA training, performance के लिए educational gold standard।
- **OLMo** (AI2) और **SmolLM-2** (HuggingFace) — fully open recipes, including data और code।
- **DeepSeek-V3 technical report** — best modern frontier training writeup।
- **Tülu 3 technical report** (AI2 2024) — canonical post-training pipeline।
- **`torchtitan`** — PyTorch की reference distributed-training library (FSDP2 + TP + activation ckpt)।

Next: **[15-reading-training-logs.md](./15-reading-training-logs.md)** — W&B charts का sense बनाना।
