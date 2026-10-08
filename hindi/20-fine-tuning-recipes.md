# 20 · Fine-Tuning Recipes — Base Model से वो काम करवाना जो आप चाहते हो

> **TL;DR** Fine-tune करने से पहले, पूछो: क्या एक better prompt या RAG इसे fix करेगा? **70% "हमें fine-tune चाहिए" requests को actually एक की need नहीं होती।** जब आपको genuinely चाहिए, 2026 default behavior changes के लिए **QLoRA-rank-16 SFT** और preference alignment के लिए **DPO** है। Full fine-tuning rare है; **LoRA / QLoRA / DoRA** workhorses हैं। **Verifiable rewards के साथ GRPO** reasoning के लिए new tool है। ये chapter सबके लिए runnable recipes और सही approach pick करने का decision tree रखता है।

---

## 1. Decision Tree: क्या आपको Fine-tune करना भी चाहिए?

किसी भी fine-tune से पहले, इस tree को walk करो। ज़्यादातर "fine-tuning" projects step 2 या 3 पर killed किए जा सकते हैं।

```
1. क्या आप desired behavior को <100 words में describe कर सकते हो?
     YES → पहले एक better system prompt try करो।
     NO  → continue।

2. क्या issue "model X नहीं जानता" है?
     YES → RAG (retrieval) use करो। Facts को weights में bake मत करो।
     NO  → continue।

3. क्या issue "model मेरा output format follow नहीं करता" है?
     YES → structured output (JSON schema, constrained decoding) use करो।
     NO  → continue।

4. क्या issue "मुझे एक बड़े model का smaller / cheaper / private version चाहिए" है?
     YES → distillation: big model से outputs collect करो, small वाले को SFT करो।

5. क्या issue "मुझे एक specific style / tone / personality चाहिए" है?
     YES → 500-5000 example pairs पर SFT (LoRA/QLoRA)।

6. क्या issue "मुझे disputed / preference questions पर better behavior चाहिए" है?
     YES → preference pairs पर DPO (या KTO / SimPO)।

7. क्या issue "मुझे verifiable answers वाले math / code / tasks पर better reasoning चाहिए" है?
     YES → GRPO / RLVR (verifiable rewards के साथ reinforcement learning)।

8. ऊपर का कुछ नहीं?
     → आपको fine-tune की need नहीं हो सकती। Problem को carefully और profile करो।
```

ये tree teams के months बचाता है। **Fine-tuning last resort है, first नहीं।**

---

## 2. Fine-tuning Hierarchy

Cheapest/safest से most expensive/risky तक:

| Method | क्या change होता है | Cost | इसके लिए best |
|--------|--------------|------|----------|
| **Prompt engineering** | nothing | free | ज़्यादातर problems |
| **Few-shot / in-context** | nothing | tokens only | format / style |
| **RAG** | nothing | retrieval | knowledge |
| **Soft prompts / prefix-tuning** | एक small prompt embedding | tiny | style / domain |
| **LoRA** | key linears पर low-rank adapters | low | ज़्यादातर fine-tunes |
| **QLoRA** | 4-bit base पर LoRA | low + memory-efficient | tight GPU budgets |
| **DoRA** | LoRA + magnitude correction | low | LoRA से slightly better |
| **Full SFT** | सारे weights | high | major behavior shift |
| **DPO / KTO / SimPO** | सारे weights या LoRA | medium | preference alignment |
| **GRPO / RLVR** | सारे weights | very high | reasoning / verifiable |
| **Continual pretraining** | सारे weights, lots of data | very high | new domain knowledge |

Rule of thumb: जो आप सोचते हो उससे एक tier ऊपर try करो। अगर LoRA fix करे, full-SFT मत करो।

---

## 3. SFT (Supervised Fine-tuning) — Bread and Butter

SFT model को thousands of `(prompt, ideal_response)` pairs दिखाता है। ये imitate करना सीखता है।

### Data Format

2026 standard format OpenAI / HuggingFace `messages` है:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a customer support agent for Acme Corp."},
    {"role": "user",      "content": "My order is late."},
    {"role": "assistant", "content": "I'm sorry to hear that. Could you share your order number?"}
  ]
}
```

A real fine-tune के लिए आप **500-50,000** ऐसे conversations चाहते हो। Quality 5k examples से नीचे quantity से कहीं ज़्यादा matter करती है।

### Critical: Loss में User / System Tokens Mask करो

Model को assistant turn *generate* करना सीखना चाहिए, user के words को *predict* नहीं। User/system tokens mask करना (उनके labels को -100 set करना) loss से noise drop करता है और SFT quality को dramatically improve करता है।

`trl.SFTTrainer` ये automatically करता है जब आप `dataset_text_field` को chat templates के साथ properly pass करते हो, या `DataCollatorForCompletionOnlyLM` use करो। हमेशा एक sample batch पर `loss_mask` inspect करके verify करो।

### 2026 में काम करने वाले Hyperparameters

```python
# QLoRA SFT — sane defaults
config = SFTConfig(
    output_dir="./out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,    # effective batch = 16
    num_train_epochs=2,                # 1-3, more = overfit
    learning_rate=2e-5,                # pretraining से कम
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    packing=True,                      # short examples pack करो
    max_seq_length=4096,
    save_strategy="epoch",
    logging_steps=10,
    optim="adamw_torch_fused",
    weight_decay=0.0,                  # SFT के लिए often NO weight decay
    gradient_checkpointing=True,
)
```

Most common mistakes:
- LR too high (1e-5 to 5e-5 use करो, pretraining जैसे 1e-4 नहीं)।
- Too many epochs (overfit; model fine-tune distribution के बाहर weird हो जाता है)।
- Loss में user/system mask करना भूलना।
- Base model के लिए *correct* chat template use न करना।

---

## 4. LoRA — 2026 का Workhorse

LoRA (Hu et al. 2021) base model को freeze करता है और selected layers में **दो small low-rank matrices** `A (d × r)` और `B (r × d)` add करता है। Forward pass बन जाता है:

```
y = (W + α/r · B A) x
```

Base `W` frozen है; सिर्फ़ `A` और `B` train होते हैं। **Trainable params की tiny number, full quality।**

### LoRA Hyperparameters

| Parameter | Typical value | Notes |
|-----------|--------------|-------|
| `r` (rank) | 16, 32, 64 | 16 से start करो; अगर ज़्यादा capacity चाहिए तो larger |
| `lora_alpha` | 16-64 | often `2 × r`; effective LR control करता है |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj` (attention) + sometimes `gate_proj, up_proj, down_proj` (FFN) | सिर्फ़ attention से start करो; need हो तो expand करो |
| `lora_dropout` | 0.05 | small regularization |
| `bias` | "none" | rarely useful biases train करना |

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

### `target_modules` कब Expand करें

- सिर्फ़ Attention: cheapest, usually style / persona के लिए enough।
- Attention + FFN: harder tasks (math, code, complex reasoning) के लिए चाहिए।
- Attention + FFN + embeddings + lm_head (rare): very domain-specific shifts।

A good rule: narrow start करो, evaluate करो, सिर्फ़ तब expand करो जब eval ज़्यादा capacity needs करे।

---

## 5. QLoRA — 4-bit Base पर LoRA

QLoRA (Dettmers et al. 2023) base model को **NF4** (4-bit normal float) में load करता है और LoRA adapters को bfloat16 में train करता है। LoRA vs near-identical quality पर memory ~4× cut करता है।

A 70B model एक single 80 GB H100 पर QLoRA के साथ fits है। एक 7B model 16 GB consumer GPU पर fits है।

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

# फिर LoRA exactly वैसे attach करो जैसे ऊपर
```

पूरी QLoRA + SFT pipeline **`trl`** में है:

```python
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(...),
    train_dataset=ds,
    peft_config=lora_cfg,        # peft_config triggers QLoRA जब model 4-bit है
    tokenizer=tok,
)
trainer.train()
trainer.save_model("./qwen3-4b-mydomain-qlora")
```

### QLoRA Saving और Serving

Training के बाद आपके पास small adapter files (~50-500 MB) हैं। Inference पर दो options:

1. Adapter को base weights में **Merge** करो (`model.merge_and_unload()`), full model save करो, vLLM के साथ serve करो।
2. **Separate रखो**, runtime पर base + adapter load करो। vLLM, TGI, और Fireworks सब LoRA adapters का hot-swapping support करते हैं — useful जब आपके पास *कई* per-customer fine-tunes हों।

```bash
# vLLM एक base पर multiple LoRAs serve करते हुए
vllm serve Qwen/Qwen3-4B \
    --enable-lora \
    --lora-modules customer_a=./adapters/a customer_b=./adapters/b \
    --max-loras 16
```

---

## 6. DoRA — LoRA का Slightly-better Successor

DoRA (Liu et al. 2024, "Weight-Decomposed Low-Rank Adaptation") weight update को **direction** (LoRA-style low-rank) और **magnitude** (एक learnable per-column scalar) में decompose करता है। ज़्यादातर fine-tunes पर LoRA को 1-3 points ~0% extra cost पर beat करता है empirically।

```python
cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    use_dora=True,                # only change
    bias="none", task_type="CAUSAL_LM",
)
```

अगर `use_dora=True` आपके `peft` version में काम करे, default के रूप में prefer करो। Weirder variants हैं (PiSSA, VeRA, GaLore) — अगर DoRA काफी न हो तो try करो; mostly small improvements।

---

## 7. DPO — Direct Preference Optimization

SFT के बाद, model बात कर सकता है; ये अभी ग़लत answer पर right answer *prefer* नहीं करता। DPO `(prompt, chosen, rejected)` triples पर train करता है:

```json
{
  "prompt":   [{"role":"user","content":"Explain ML to a 5-year-old."}],
  "chosen":   [{"role":"assistant","content":"Imagine teaching your dog tricks…"}],
  "rejected": [{"role":"assistant","content":"Machine learning is a subset of AI…"}]
}
```

DPO directly policy को optimize करता है chosen को rejected से higher log-prob assign करने के लिए, एक KL penalty के साथ जो इसे SFT model के close रखता है।

### Recipe

```python
from trl import DPOTrainer, DPOConfig

cfg = DPOConfig(
    output_dir="./dpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,                # 1-2; more = overfit
    learning_rate=5e-7,                # SFT से 10-100× smaller
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
    ref_model=None,                    # SFT model को reference के रूप में automatically use करता है
    args=cfg,
    train_dataset=preference_ds,
    tokenizer=tok,
    peft_config=lora_cfg,              # LoRA के साथ DPO default है
)
trainer.train()
```

Hyperparameter notes:
- LR SFT से **much** smaller है (5e-7, 5e-5 नहीं)।
- `beta` KL penalty control करता है: too small → SFT से drift, too large → barely changes anything। `0.05, 0.1, 0.5` sweep करो।
- One epoch usually enough है; more over-fitting है।
- 5,000-50,000 preference pairs typical है।

### Public Preference Datasets

- **UltraFeedback** — 64k high-quality pairs।
- **HelpSteer3** — Nvidia का preference dataset।
- **Anthropic HH-RLHF** — older but battle-tested।
- **Chatbot Arena conversations** — real human preferences।
- **PKU-SafeRLHF** — safety-focused।

### Worth Trying Variants

- **KTO** (Kahneman-Tversky): unpaired good/bad ratings use करता है; cheaper to label।
- **IPO**: deterministic preferences के पास better behavior।
- **SimPO**: reference model drop करता है; smaller memory।
- **ORPO**: SFT + preference को एक stage में combine करता है।

Most teams के लिए: **DPO default है**। Variants सिर्फ़ तब try करो जब DPO results weak हों।

---

## 8. GRPO / RLVR — Reasoning Models के लिए

DeepSeek-R1 (early 2025) ने दिखाया कि reasoning tasks पर `verifiable rewards के साथ RL` (RLVR) एक base model से frontier-class math/code models produce कर सकता है alone। 2026 में choice का algorithm **GRPO** (Group Relative Policy Optimization) है।

### Setup

आपको चाहिए:
- A **prompt set** verifiable answers के साथ (known solutions वाले math problems, unit tests वाले code problems)।
- A **verifier function** जो हर generation के लिए 0 या 1 return करे।
- A base model strong enough that occasionally correct answers produce करे।

हर prompt के लिए, N (usually 8) generations sample करो। हर एक को verifier से score करो। *Group advantage* compute करो — group mean से कितना better — और policy update करो।

```python
from trl import GRPOConfig, GRPOTrainer

def reward_fn(samples, **kwargs):
    """Math answer correct है या नहीं इस basis पर per sample 0/1 return करो।"""
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
    use_vllm=True,                     # sampling accelerate करो
    beta=0.04,                         # reference के लिए KL
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

### 2026 में क्या काम करता है

- **Math**: Numina, AIME-style problems, MATH dataset। Verifier numeric answer compare करता है।
- **Code**: HumanEval-style या LeetCode-style, sandbox में unit tests run करो। Verifier pass rate return करता है।
- **Logic / reasoning**: deterministic answers वाले BIG-Bench style problems।
- **Tool-use trajectories**: verify final task succeed किया।

### Training Tips

- More sample (N=16+) ज़्यादा signal देता है लेकिन slow है। N=8 good default है।
- KL penalty (`beta`) collapse prevent करता है; too low और model degenerate हो जाता है।
- **Verifier quality everything है**: bad verifier bad behavior teach करता है। Independently test करो।
- Reward shaping helps: small partial-credit signals (compiles? runs? answer के close?) early training speed up करते हैं।
- Reward over time plot करो; slow climb expect करो, फिर एक plateau।

DeepSeek-V3 / R1 / R1-Zero technical reports RLVR design पर canonical reading रहते हैं।

---

## 9. Distillation — Underrated Shortcut

अगर आप एक small model चाहते हो जो एक big वाले की तरह behave करे, **scratch से fine-tune मत करो — distill करो**। Big model से आपके distribution के representative prompts पर कुछ hundred thousand high-quality outputs generate करो, फिर उन पर एक smaller model SFT करो।

```python
# 1. teacher outputs generate करो
teacher = anthropic.Anthropic()
distill_data = []
for prompt in prompts:
    out = teacher.messages.create(model="claude-opus-4-7",
                                   max_tokens=2048,
                                   messages=[{"role":"user","content":prompt}])
    distill_data.append({"prompt": prompt, "completion": out.content[0].text})

# 2. उस पर एक small model SFT करो
# (chapter 20 में standard SFT pipeline)
```

Cosmopedia, Phi, और कई "small but smart" 2026 models exactly इस तरह बनते हैं। 5-20× lower inference cost पर teacher की quality का ~30-70%।

### Logit Distillation (Better, जब आपके पास Access हो)

अगर आप teacher in-house run कर सकते हो (e.g., एक permissively-licensed open model), teacher के *logit distribution* को match करो, सिर्फ़ उसके argmax outputs को नहीं:

```python
loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction="batchmean",
) * (T * T)
```

Temperature `T` (typically 2-4) distributions को soften करता है। Logit distillation per example text-only से far ज़्यादा information transfer करता है।

---

## 10. Continual Fine-tuning और Catastrophic Forgetting

एक model को fine-tune करना often उन tasks पर performance degrade करता है जो ये पहले अच्छा था। ये **catastrophic forgetting** है। Mitigations:

- अपने domain data के साथ **कुछ general data mix करो** (e.g., 10-30% Tülu/UltraChat)।
- Continual pretraining के लिए **Lower LR** (1e-5 या नीचे)।
- **Limit epochs** — 1-2 usually enough है।
- **LoRA-only training** (merging नहीं) — आप adapter को hot-swap या remove कर सकते हो।
- **Replay buffer** — "old behavior" examples का एक small set रखो और उनमें mix करो।
- सिर्फ़ अपने target task पर नहीं, **multiple benchmarks पर compare करो**।

A common production pattern: नए user data पर weekly LoRA fine-tunes, full benchmark suite के across monthly evaluation, अगर drift accumulate हो तो original base से periodic retraining।

---

## 11. End-to-end: एक Custom Domain पर Qwen 3 को ~150 lines में Fine-tune करो

A complete recipe — single 24 GB consumer GPU पर runnable।

```python
"""
Qwen 3-4B के लिए QLoRA SFT एक custom JSONL dataset पर।
हर line {"messages": [{"role":..., "content":...}, ...]} है।
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
trainer.save_model(OUT)         # सिर्फ़ adapter

# ─── 6. Optional: serving के लिए merge ─────────────────────────────────────
# from peft import AutoPeftModelForCausalLM
# m = AutoPeftModelForCausalLM.from_pretrained(OUT, torch_dtype=torch.bfloat16)
# m.merge_and_unload().save_pretrained(OUT + "-merged")
```

फिर अगर आपके पास preference data हो तो DPO, अगर verifiable tasks हों तो GRPO। Skeleton same है।

---

## 12. Common Fine-tuning Failures और Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| 50 steps में Loss 0 हो जाती है | Tiny dataset overfitting | More data, fewer epochs, drop LR |
| Loss never drops | LR too low, wrong chat template, frozen layers | First batch print करो, masking verify करो, LR raise करो |
| Model general tasks पर dumber होता है | Catastrophic forgetting | General data mix करो, fewer epochs |
| Model training data verbatim parrot करता है | Severe overfit | Lower LR, fewer epochs, more data |
| Outputs repetitive / collapsed बनते हैं | KL penalty too low (DPO/GRPO) | `beta` increase करो |
| Out of memory | Batch / sequence too big | Gradient checkpointing, gradient accumulation, smaller `r` |
| QLoRA NaN | bf16 underflow या fp16 mix | Throughout bf16 use करो, lower LR |
| LoRA + merge no merge से worse model देता है | Numerical drift | Adapter save करो, hot-swap के साथ serve करो |
| SFT के बाद eval drops while train loss looks fine | Eval distribution train से differ | Train में eval-distribution examples add करो |

---

## 13. 2026 Cheat Sheet

- **Fine-tune मत करो जब तक prompt + RAG + structured output fail न हों।**
- Default: **Attention + FFN पर QLoRA DoRA के साथ, rank 16।**
- LR: **SFT के लिए 2e-5, DPO के लिए 5e-7, GRPO के लिए 1e-6।**
- SFT loss में **user/system tokens mask करो**।
- SFT के लिए **1-3 epochs**, DPO के लिए **1 epoch**। More = overfit।
- Catastrophic forgetting fight करने के लिए **10-30% general data mix करो**।
- Evaluating के समय **different families से Multiple judges**।
- जब strong teacher exist हो **scratch से fine-tune > Distillation**।
- Verifiable answers वाले reasoning tasks के लिए **GRPO** new tool है।
- Per-customer fine-tunes के लिए vLLM में **adapters Hot-swap करो**।
- **हमेशा एक held-out set पर evaluate करो** plus general benchmarks; कभी same data पर नहीं जिस पर train किया।

---

## और गहराई से

- **`trl` repo और docs** — TRL 0.10+ canonical 2026 fine-tuning library है।
- **Hu et al. 2021** — original LoRA paper।
- **Dettmers et al. 2023** — QLoRA paper।
- **Liu et al. 2024** — DoRA paper।
- **Rafailov et al. 2023** — DPO paper।
- **DeepSeek-R1 technical report** — scale पर RLVR / GRPO।
- **Tülu 3 technical report** — modern post-training pipeline (SFT + DPO + verifiable rewards के साथ RLVR) end-to-end।
- **Hugging Face's "alignment-handbook"** — production-grade scripts।
- **`unsloth` और `axolotl`** — popular community fine-tuning frameworks।

Next: **[21-embeddings-and-rag.md](./21-embeddings-and-rag.md)** — जब retrieval fine-tuning को beat करती है।
