# 20 · Fine-Tuning Recipes — Making a Base Model Do What You Want

> **TL;DR** Before you fine-tune, ask: would a better prompt or RAG fix this? **70% of "we need a fine-tune" requests don't actually need one.** When you genuinely do, the 2026 default is **QLoRA-rank-16 SFT** for behavior changes and **DPO** for preference alignment. Full fine-tuning is rare; **LoRA / QLoRA / DoRA** are the workhorses. **GRPO with verifiable rewards** is the new tool for reasoning. This chapter has runnable recipes for all of them and a decision tree for picking the right approach.

---

## 1. The decision tree: should you fine-tune at all?

Before any fine-tune, walk this tree. Most "fine-tuning" projects can be killed at step 2 or 3.

```
1. Can you describe the desired behavior in <100 words?
     YES → try a better system prompt first.
     NO  → continue.

2. Is the issue "the model doesn't know X"?
     YES → use RAG (retrieval). Don't bake facts into weights.
     NO  → continue.

3. Is the issue "the model doesn't follow my output format"?
     YES → use structured output (JSON schema, constrained decoding).
     NO  → continue.

4. Is the issue "I want a smaller / cheaper / private version of a big model"?
     YES → distillation: collect outputs from the big model, SFT a small one.

5. Is the issue "I want a specific style / tone / personality"?
     YES → SFT (LoRA/QLoRA) on 500-5000 example pairs.

6. Is the issue "I want better behavior on disputed / preference questions"?
     YES → DPO (or KTO / SimPO) on preference pairs.

7. Is the issue "I want better reasoning on math / code / tasks with verifiable answers"?
     YES → GRPO / RLVR (reinforcement learning with verifiable rewards).

8. None of the above?
     → you may not need a fine-tune. Profile the problem more carefully.
```

This tree saves teams months. **Fine-tuning is the last resort, not the first.**

---

## 2. The fine-tuning hierarchy

From cheapest/safest to most expensive/risky:

| Method | What changes | Cost | Best for |
|--------|--------------|------|----------|
| **Prompt engineering** | nothing | free | most problems |
| **Few-shot / in-context** | nothing | tokens only | format / style |
| **RAG** | nothing | retrieval | knowledge |
| **Soft prompts / prefix-tuning** | a small prompt embedding | tiny | style / domain |
| **LoRA** | low-rank adapters on key linears | low | most fine-tunes |
| **QLoRA** | LoRA on a 4-bit base | low + memory-efficient | tight GPU budgets |
| **DoRA** | LoRA + magnitude correction | low | slightly better than LoRA |
| **Full SFT** | all weights | high | major behavior shift |
| **DPO / KTO / SimPO** | all weights or LoRA | medium | preference alignment |
| **GRPO / RLVR** | all weights | very high | reasoning / verifiable |
| **Continual pretraining** | all weights, lots of data | very high | new domain knowledge |

Rule of thumb: try one tier above what you think you need. If LoRA fixes it, don't full-SFT.

---

## 3. SFT (supervised fine-tuning) — the bread and butter

SFT shows the model thousands of `(prompt, ideal_response)` pairs. It learns to imitate.

### Data format

The standard 2026 format is OpenAI / HuggingFace `messages`:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a customer support agent for Acme Corp."},
    {"role": "user",      "content": "My order is late."},
    {"role": "assistant", "content": "I'm sorry to hear that. Could you share your order number?"}
  ]
}
```

For a real fine-tune you want **500-50,000** such conversations. Quality matters far more than quantity below 5k examples.

### Critical: mask user / system tokens in the loss

The model should learn to *generate* the assistant turn, not *predict* the user's words. Masking user/system tokens (set their labels to -100) drops the noise from the loss and improves SFT quality dramatically.

`trl.SFTTrainer` does this automatically when you pass `dataset_text_field` properly with chat templates, or use `DataCollatorForCompletionOnlyLM`. Always verify by inspecting `loss_mask` on a sample batch.

### Hyperparameters that work in 2026

```python
# QLoRA SFT — sane defaults
config = SFTConfig(
    output_dir="./out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,    # effective batch = 16
    num_train_epochs=2,                # 1-3, more = overfit
    learning_rate=2e-5,                # lower than pretraining
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    packing=True,                      # pack short examples
    max_seq_length=4096,
    save_strategy="epoch",
    logging_steps=10,
    optim="adamw_torch_fused",
    weight_decay=0.0,                  # NO weight decay for SFT, often
    gradient_checkpointing=True,
)
```

The most common mistakes:
- LR too high (use 1e-5 to 5e-5, not 1e-4 like pretraining).
- Too many epochs (overfit; the model becomes weird outside fine-tune distribution).
- Forgetting to mask user/system in the loss.
- Not using the *correct* chat template for the base model.

---

## 4. LoRA — the workhorse of 2026

LoRA (Hu et al. 2021) freezes the base model and adds **two small low-rank matrices** `A (d × r)` and `B (r × d)` to selected layers. Forward pass becomes:

```
y = (W + α/r · B A) x
```

The base `W` is frozen; only `A` and `B` train. **Tiny number of trainable params, full quality.**

### LoRA hyperparameters

| Parameter | Typical value | Notes |
|-----------|--------------|-------|
| `r` (rank) | 16, 32, 64 | start at 16; larger if more capacity needed |
| `lora_alpha` | 16-64 | often `2 × r`; controls effective LR |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj` (attention) + sometimes `gate_proj, up_proj, down_proj` (FFN) | start with attention only; expand if needed |
| `lora_dropout` | 0.05 | small regularization |
| `bias` | "none" | rarely useful to train biases |

```python
from peft import LoraConfig, get_peft_model
cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, cfg)
model.print_trainable_parameters()   # ~0.5-1% of total
```

### When to expand `target_modules`

- Attention only: cheapest, usually enough for style / persona.
- Attention + FFN: needed for harder tasks (math, code, complex reasoning).
- Attention + FFN + embeddings + lm_head (rare): very domain-specific shifts.

A good rule: start narrow, evaluate, expand only if eval needs more capacity.

---

## 5. QLoRA — LoRA on a 4-bit base

QLoRA (Dettmers et al. 2023) loads the base model in **NF4** (4-bit normal float) and trains LoRA adapters in bfloat16. Cuts memory ~4× vs LoRA at near-identical quality.

A 70B model fits a single 80 GB H100 with QLoRA. A 7B model fits a 16 GB consumer GPU.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B",
    quantization_config=bnb,
    device_map="auto",
)

# then attach LoRA exactly as above
```

The whole QLoRA + SFT pipeline is in **`trl`**:

```python
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(...),
    train_dataset=ds,
    peft_config=lora_cfg,        # peft_config triggers QLoRA when model is 4-bit
    tokenizer=tok,
)
trainer.train()
trainer.save_model("./qwen3-4b-mydomain-qlora")
```

### Saving and serving QLoRA

After training you have small adapter files (~50-500 MB). Two options at inference:

1. **Merge** the adapter into the base weights (`model.merge_and_unload()`), save full model, serve with vLLM.
2. **Keep separate**, load base + adapter at runtime. vLLM, TGI, and Fireworks all support hot-swapping LoRA adapters — useful when you have *many* per-customer fine-tunes.

```bash
# vLLM serving multiple LoRAs on one base
vllm serve Qwen/Qwen3-4B \
    --enable-lora \
    --lora-modules customer_a=./adapters/a customer_b=./adapters/b \
    --max-loras 16
```

---

## 6. DoRA — LoRA's slightly-better successor

DoRA (Liu et al. 2024, "Weight-Decomposed Low-Rank Adaptation") decomposes the weight update into **direction** (LoRA-style low-rank) and **magnitude** (a learnable per-column scalar). Empirically beats LoRA by 1-3 points on most fine-tunes at ~0% extra cost.

```python
cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    use_dora=True,                # the only change
    bias="none", task_type="CAUSAL_LM",
)
```

If `use_dora=True` works in your `peft` version, prefer it as a default. There are weirder variants (PiSSA, VeRA, GaLore) — try if DoRA isn't enough; mostly small improvements.

---

## 7. DPO — direct preference optimization

After SFT, the model can talk; it doesn't yet *prefer* the right answer over the wrong one. DPO trains on `(prompt, chosen, rejected)` triples:

```json
{
  "prompt":   [{"role":"user","content":"Explain ML to a 5-year-old."}],
  "chosen":   [{"role":"assistant","content":"Imagine teaching your dog tricks…"}],
  "rejected": [{"role":"assistant","content":"Machine learning is a subset of AI…"}]
}
```

DPO directly optimizes the policy to assign higher log-prob to chosen than rejected, with a KL penalty that keeps it close to the SFT model.

### Recipe

```python
from trl import DPOTrainer, DPOConfig

cfg = DPOConfig(
    output_dir="./dpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,                # 1-2; more = overfit
    learning_rate=5e-7,                # 10-100× smaller than SFT
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    beta=0.1,                          # KL penalty strength; 0.05-0.5 typical
    max_length=4096,
    max_prompt_length=2048,
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
)

trainer = DPOTrainer(
    model=sft_model,
    ref_model=None,                    # uses SFT model as reference automatically
    args=cfg,
    train_dataset=preference_ds,
    tokenizer=tok,
    peft_config=lora_cfg,              # DPO with LoRA is the default
)
trainer.train()
```

Hyperparameter notes:
- LR is **much** smaller than SFT (5e-7, not 5e-5).
- `beta` controls KL penalty: too small → drifts from SFT, too large → barely changes anything. Sweep `0.05, 0.1, 0.5`.
- One epoch is usually enough; more is over-fitting.
- 5,000-50,000 preference pairs is typical.

### Public preference datasets

- **UltraFeedback** — 64k high-quality pairs.
- **HelpSteer3** — Nvidia's preference dataset.
- **Anthropic HH-RLHF** — older but battle-tested.
- **Chatbot Arena conversations** — real human preferences.
- **PKU-SafeRLHF** — safety-focused.

### Variants worth trying

- **KTO** (Kahneman-Tversky): uses unpaired good/bad ratings; cheaper to label.
- **IPO**: better behavior near deterministic preferences.
- **SimPO**: drops the reference model; smaller memory.
- **ORPO**: combines SFT + preference in one stage.

For most teams: **DPO is the default**. Try variants only if DPO results are weak.

---

## 8. GRPO / RLVR — for reasoning models

DeepSeek-R1 (early 2025) showed that `RL with verifiable rewards` (RLVR) on reasoning tasks can produce frontier-class math/code models from a base model alone. The algorithm of choice in 2026 is **GRPO** (Group Relative Policy Optimization).

### The setup

You need:
- A **prompt set** with verifiable answers (math problems with known solutions, code problems with unit tests).
- A **verifier function** that returns 0 or 1 for each generation.
- A base model strong enough to occasionally produce correct answers.

For each prompt, sample N (usually 8) generations. Score each with the verifier. Compute the *group advantage* — how much better than the group mean — and update the policy.

```python
from trl import GRPOConfig, GRPOTrainer

def reward_fn(samples, **kwargs):
    """Return 0/1 per sample based on whether the math answer is correct."""
    return [verify_math(s, kwargs["solution"]) for s in samples]

cfg = GRPOConfig(
    output_dir="./grpo",
    learning_rate=1e-6,
    num_generations=8,                 # samples per prompt
    max_prompt_length=2048,
    max_completion_length=4096,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    use_vllm=True,                     # accelerate sampling
    beta=0.04,                         # KL to reference
)

trainer = GRPOTrainer(
    model=base_model,
    reward_funcs=[reward_fn],
    args=cfg,
    train_dataset=math_ds,
    tokenizer=tok,
)
trainer.train()
```

### What works in 2026

- **Math**: Numina, AIME-style problems, MATH dataset. Verifier compares numeric answer.
- **Code**: HumanEval-style or LeetCode-style, run unit tests in a sandbox. Verifier returns pass rate.
- **Logic / reasoning**: BIG-Bench style problems with deterministic answers.
- **Tool-use trajectories**: verify the final task succeeded.

### Training tips

- Sample more (N=16+) gives more signal but is slow. N=8 is a good default.
- KL penalty (`beta`) prevents collapse; too low and the model degenerates.
- **Verifier quality is everything**: a bad verifier teaches bad behavior. Test it independently.
- Reward shaping helps: small partial-credit signals (compiles? runs? close to answer?) speed up early training.
- Plot reward over time; expect a slow climb, then a plateau.

DeepSeek-V3 / R1 / R1-Zero technical reports remain the canonical reading on RLVR design.

---

## 9. Distillation — the underrated shortcut

If you want a small model that behaves like a big one, **don't fine-tune from scratch — distill**. Generate a few hundred thousand high-quality outputs from the big model on prompts representative of your distribution, then SFT a smaller model on those.

```python
# 1. generate teacher outputs
teacher = anthropic.Anthropic()
distill_data = []
for prompt in prompts:
    out = teacher.messages.create(model="claude-opus-4-7",
                                   max_tokens=2048,
                                   messages=[{"role":"user","content":prompt}])
    distill_data.append({"prompt": prompt, "completion": out.content[0].text})

# 2. SFT a small model on it
# (standard SFT pipeline as in §3)
```

This is exactly how Cosmopedia, Phi, and many "small but smart" 2026 models are built. ~30-70% of the teacher's quality at 5-20× lower inference cost.

### Logit distillation (better, when you have access)

If you can run the teacher in-house (e.g., a permissively-licensed open model), match the teacher's *logit distribution*, not just its argmax outputs:

```python
loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction="batchmean",
) * (T * T)
```

Temperature `T` (typically 2-4) softens the distributions. Logit distillation transfers far more information per example than text-only.

---

## 10. Continual fine-tuning and catastrophic forgetting

Fine-tuning a model often degrades performance on tasks it was good at before. This is **catastrophic forgetting**. Mitigations:

- **Mix in some general data** (e.g., 10-30% Tülu/UltraChat) with your domain data.
- **Lower LR** for continual pretraining (1e-5 or below).
- **Limit epochs** — 1-2 is usually enough.
- **LoRA-only training** (not merging) — you can hot-swap or remove the adapter.
- **Replay buffer** — keep a small set of "old behavior" examples and mix them in.
- **Compare on multiple benchmarks**, not just your target task.

A common production pattern: weekly LoRA fine-tunes on new user data, monthly evaluation across the full benchmark suite, periodic retraining from the original base if drift accumulates.

---

## 11. End-to-end: fine-tune Qwen 3 on your domain in ~150 lines

A complete recipe — runnable on a single 24 GB consumer GPU.

```python
"""
QLoRA SFT for Qwen 3-4B on a custom JSONL dataset.
Each line is {"messages": [{"role":..., "content":...}, ...]}.
"""
import json, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL = "Qwen/Qwen3-4B"
DATA  = "data/train.jsonl"
OUT   = "out/qwen3-4b-mydomain"

# ─── 1. Tokenizer ───────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# ─── 2. Model in 4-bit ──────────────────────────────────────────────────────
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model)

# ─── 3. LoRA config ─────────────────────────────────────────────────────────
peft_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    use_dora=True,                         # DoRA upgrade
)

# ─── 4. Data ────────────────────────────────────────────────────────────────
ds = load_dataset("json", data_files=DATA, split="train")

def fmt(ex):
    return {"text": tok.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}

ds = ds.map(fmt, remove_columns=ds.column_names)

# ─── 5. Train ───────────────────────────────────────────────────────────────
args = SFTConfig(
    output_dir=OUT,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True, packing=True, max_seq_length=4096,
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    save_strategy="epoch",
    logging_steps=20,
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model, args=args,
    train_dataset=ds, tokenizer=tok,
    peft_config=peft_cfg,
)
trainer.train()
trainer.save_model(OUT)         # adapter only

# ─── 6. Optional: merge for serving ─────────────────────────────────────────
# from peft import AutoPeftModelForCausalLM
# m = AutoPeftModelForCausalLM.from_pretrained(OUT, torch_dtype=torch.bfloat16)
# m.merge_and_unload().save_pretrained(OUT + "-merged")
```

Then DPO if you have preference data, GRPO if you have verifiable tasks. The skeleton is the same.

---

## 12. Common fine-tuning failures and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Loss goes to 0 in 50 steps | Overfitting tiny dataset | More data, fewer epochs, drop LR |
| Loss never drops | LR too low, wrong chat template, frozen layers | Print first batch, verify masking, raise LR |
| Model gets dumber on general tasks | Catastrophic forgetting | Mix general data, fewer epochs |
| Model parrots training data verbatim | Severe overfit | Lower LR, fewer epochs, more data |
| Outputs become repetitive / collapsed | KL penalty too low (DPO/GRPO) | Increase `beta` |
| Out of memory | Batch / sequence too big | Gradient checkpointing, gradient accumulation, smaller `r` |
| QLoRA NaN | bf16 underflow or fp16 mix | Use bf16 throughout, lower LR |
| LoRA + merge gives worse model than no merge | Numerical drift | Save adapter, serve with hot-swap |
| Eval drops after SFT but train loss looks fine | Eval distribution differs from train | Add eval-distribution examples to train |

---

## 13. The 2026 cheat sheet

- **Don't fine-tune unless prompt + RAG + structured output have failed.**
- Default: **QLoRA with DoRA, rank 16, on attention + FFN.**
- LR: **2e-5 for SFT, 5e-7 for DPO, 1e-6 for GRPO.**
- **Mask user/system tokens** in SFT loss.
- **1-3 epochs** for SFT, **1 epoch** for DPO. More = overfit.
- **Mix in 10-30% general data** to fight catastrophic forgetting.
- **Multiple judges from different families** when evaluating.
- **Distillation > fine-tune from scratch** when a strong teacher exists.
- **GRPO** is the new tool for reasoning tasks with verifiable answers.
- **Hot-swap adapters** in vLLM for per-customer fine-tunes.
- **Always evaluate on a held-out set** plus general benchmarks; never on the same data you trained on.

---

## Going deeper

- **`trl` repo and docs** — TRL 0.10+ is the canonical 2026 fine-tuning library.
- **Hu et al. 2021** — original LoRA paper.
- **Dettmers et al. 2023** — QLoRA paper.
- **Liu et al. 2024** — DoRA paper.
- **Rafailov et al. 2023** — DPO paper.
- **DeepSeek-R1 technical report** — RLVR / GRPO at scale.
- **Tülu 3 technical report** — modern post-training pipeline (SFT + DPO + RLVR with verifiable rewards) end-to-end.
- **Hugging Face's "alignment-handbook"** — production-grade scripts.
- **`unsloth` and `axolotl`** — popular community fine-tuning frameworks.

Next: **[21-embeddings-and-rag.md](./21-embeddings-and-rag.md)** — when retrieval beats fine-tuning.
