# 25 · Model Releases, Versioning, और Model के अंदर Look करना

> **TL;DR** Model versions एक four-tier ladder follow करती हैं: **major (X → X+1) = fresh pretrain**, often new architecture और tokenizer के साथ; **minor (X → X.5) = continual pretrain** existing checkpoint के top पर new/refreshed data के साथ; **point (X.5 → X.6/X.7) = सिर्फ़ post-training** (SFT/DPO/GRPO refresh, no pretraining); **patch (X.7 → X.7.1) = targeted bug/safety fix**। हर tier data pipeline के different stage पर focus करता है और different tools use करता है। Model के अंदर, आप उसका mind read कर सकते हो: **forward hooks** activations capture करते हैं, **TransformerLens / NNsight** हर layer और head को inspectable बनाते हैं, **logit-lens** intermediate activations को LM head के through project करता है ये देखने के लिए "model layer 17 पर answer के बारे में क्या सोच रहा है?", और **activation patching** forward passes के बीच activations copy करके isolate करता है कौन से neurons कौन सा behavior cause करते हैं। **एक single head, layer, या weight ablate करना** model में क्या matter करता है find करने की most reliable single-line technique है।

---

## 1. Release Ladder

A real production model family एक model नहीं ship करती — एक *cadence* ship करती है। हर version step different money cost करता है, different time लेता है, और pipeline के different parts change करता है। उन सबको same treat करना सबसे expensive mistake है।

| Tier | Version step | क्या change होता है | क्या change NHI होता | Compute cost | Calendar time |
|------|--------------|---------------|----------------------|--------------|---------------|
| **Major** | X → X+1 | architecture, tokenizer, **random init से fresh pretrain** | brand identity | $$$$ (10⁵-10⁷ GPU-h) | months |
| **Minor** | X → X.5 | checkpoint के top पर **continual pretrain**, new data mix, more tokens, sometimes added modality / longer context | architecture, tokenizer | $$ (~5-20% of major) | weeks |
| **Mid-train / anneal release** | X.5 → X.5b | extra-clean / domain-specific data पर एक **second pretrain phase**; same arch और starting point के weights | tokenizer | $ (~1-5% of major) | days-weeks |
| **Point** | X.5 → X.6 / X.7 | **सिर्फ़ post-training** — new SFT mix, new DPO/GRPO data, updated chat template | post-training के initialization पर base model weights | $$ (pretrain से orders of magnitude cheaper) | days |
| **Patch** | X.7 → X.7.1 | safety/bug examples पर targeted SFT, prompt template fix | almost everything | $ | hours-days |

आप ये exact ladder हर public family में देखोगे:

- **Anthropic Claude:** 3 → 3.5 → 3.7 → 4 → 4.5 → 4.6 → 4.7 — major bumps new pretrains हैं, .5 / .6 / .7 post-training-heavy refreshes (occasional mid-train uplifts के साथ)।
- **OpenAI GPT:** 4 → 4o → 4.1 → 5 → 5.5 — `4 → 4o` native multimodality के साथ new pretrain था; `4 → 4.1` continual pretrain + post-training था; `4.1 → 5` fresh pretrain था।
- **Meta Llama:** 3 → 3.1 → 3.2 → 3.3 → 4 — `3.1` continual pretrain के via context extend किया; `3.2` visual data पर continual pretrain के via vision add किया; `3.3` post-training था; `4` new architecture के साथ new pretrain था।
- **Qwen:** 2 → 2.5 → 3 → 3.5 → 3.6 — most majors fresh pretrains हैं; `3.6` एक hybrid-attention overhaul था (chapter 16)।
- **DeepSeek:** V2 → V2.5 → V3 → V3.1 → V3.2 — `V3` ने MLA + MoE fine-grained experts introduce किए और fresh pretrain था।
- **Gemma:** 2 → 2.5 → 3 → 3n → 4 — `3n` edge के लिए new architecture था; `4` एक new design पर fresh pretrain था (chapter 24)।

ऊपर layer किए गए कुछ और release patterns:

- **Distillation children**: एक tiny model big वाले से distilled, major के साथ released (Llama-3-8B Llama 3-70B से distilled; Phi family much larger teachers से)।
- **Long-context variants**: extra continual pretrain + YaRN tuning से produced एक `-1M` या `-256K` variant। Same base, different context।
- **Reasoning variants**: base के top पर GRPO से trained एक `-Reasoner` या `-Thinking` variant, sibling के रूप में released।
- **Multimodal extensions**: multimodal data पर continual pretrain के via vision/audio added।

जब आप एक release plan करो, **पहले decide करो आप कौन से tier में हो** — tooling, calendar, और risk profile completely different हैं।

---

## 2. Major Release: Fresh Pretrain from Scratch

आप ये तब करते हो जब:

- Architecture meaningfully change होता है (MoE introduced, hybrid attention, new positional encoding, new tokenizer)।
- Data mix में fundamental rewrite है (new domain dominance, multilingual focus, much higher quality threshold)।
- Compute budget allow करता है (ये 10⁵-10⁷ GPU-hour runs हैं, $1M-$1B+)।

**Pretrain pipeline random init से start होती है।** हर weight reset हो जाता है; कुछ carry forward नहीं होता। ये only way है *truly* एक new architecture incorporate करने का।

Stages और उनका focus:

| Stage | किस पर focus | Tools |
|-------|------------------|-------|
| **Architecture design** | layer count, hidden dim, FFN ratio, attention type, RoPE base | paper search, scaling-law fit |
| **Tokenizer** | vocab size, BPE merges, multilingual coverage | `tokenizers` (HuggingFace) |
| **Data mix** | domain से shards, quality filtering, scale पर dedup, decontamination | `datatrove`, `mosaicml-streaming`, MinHash, classifier filters |
| **Optimizer & schedule** | AdamW vs Muon, warmup, cosine, anneal phase | `torchtitan`, custom |
| **Distributed training** | FSDP2 + TP + EP + PP, fault tolerance, checkpointing | `torchtitan`, `Megatron-LM`, `NeMo`, `DeepSpeed`, `Levanter` |
| **Mid-flight monitoring** | loss curves, grad norm, checkpoints पर eval | W&B, TensorBoard |
| **Mid-train annealing** | curated mix पर tokens का last 5-10% | same trainer, बस config flip |
| **Post-training** | SFT, DPO, GRPO | `trl`, alignment-handbook |
| **Evaluation** | full benchmark suite, internal evals | `lm-evaluation-harness`, `inspect-ai` |

Calendar: architecture-design lock से release तक 2-6 months। इसमें से ज़्यादातर actual pretrain run है।

---

## 3. Minor Release: Continual Pretrain

A minor release **existing major के checkpoint से start होती है** और new data पर training continue करती है। Architecture, tokenizer, और weights सब carry forward होते हैं। आप effectively *more* training कर रहे हो, new training नहीं।

कब करना है:

- आपके पास **fresh data** है जो original mix में नहीं था (new web crawl, new domain, new language)।
- आप **context extend** करना चाहते हो (long docs पर YaRN + continual pretrain के via 8K → 128K)।
- आप **एक modality add** करना चाहते हो (multimodal data पर continual pretrain के via text → text+vision)।
- आप **specialize** करना चाहते हो (general → coding-focused, medical-focused)।
- आप data wall hit कर चुके हो और tokens-per-param higher push करना चाहते हो।

क्या carry forward होता है, क्या नहीं:

```
Carry forward होता है:
   weights, optimizer state (sometimes), tokenizer, architecture

Reset होता है (या fresh start):
   learning-rate schedule (small warmup, lower peak LR)
   data mix (potentially base से different)
   evaluation cadence
```

Critical decisions:

- **LR**: continual pretrain LR original peak LR का 10-30% है (e.g., 3e-4 base → 5e-5 to 1e-4 continual)।
- **Tokens budget**: original pretrain token count का 5-30%।
- **Anneal phase**: very common। Most "X.5" releases essentially "X + better data पर एक long anneal phase" हैं।
- **Risk: catastrophic forgetting.** अगर आपका new data too narrow है, model general capabilities भूल जाता है। एक specialty target करते समय भी 30-60% general data mix करो।

A common 2026 mid-train recipe:

```python
# base के against continual pretrain config diff के लिए pseudocode
config = base_config.copy()
config.lr_peak = 5e-5                   # base 3e-4 का 1/6
config.lr_warmup_steps = 1000           # short warmup
config.lr_schedule = "linear_decay"     # end तक ~0 तक decay
config.total_tokens = 200_000_000_000   # original 4T का ~5%
config.data_mix = {                     # heavy specialty + general
    "math": 0.30,
    "code": 0.20,
    "general_web_filtered": 0.30,
    "instruction_traces": 0.20,
}
config.start_from_checkpoint = "base/step_1500000.pt"
```

ये आपको एक "X.5" देता है — same model, more capable, no architecture surgery।

---

## 4. Point Release: Sirf Post-training

Cheapest, most frequent kind। आप base model checkpoint लेते हो और उस पर SFT + DPO/GRPO run करते हो। **No pretraining।** Single-digit thousands of GPU-hours cost करता है। अगर आपके पास data है तो हर 2-4 weeks में point release ship कर सकते हो।

Focus कहां है:

| Sub-stage | किस पर focus | Tools |
|-----------|------------------|-------|
| **SFT data curation** | quality, diversity, refusal coverage, format conformity | `datatrove`, manual review, judge-filtering |
| **SFT** | LoRA / QLoRA / full, hyperparameters | `trl`, `axolotl`, `unsloth`, `alignment-handbook` |
| **Preference data** | UltraFeedback, HelpSteer3, अपना | curation pipelines |
| **DPO / KTO / SimPO** | beta sweep, pair quality | `trl.DPOTrainer` |
| **GRPO / RLVR** (reasoning) | verifier quality, reward shaping | `trl.GRPOTrainer`, OpenR1 |
| **Constitutional revisions** | Anthropic-style self-critique | scripted |
| **Eval gauntlet** | prior version vs win-rate, golden set पर regression | `promptfoo`, LangSmith, internal harness |

यहीं से production में ज़्यादातर "हमने coding पर model 5% smarter बनाया" wins आती हैं। Point releases को overlook मत करो।

---

## 5. Patch Release

Targeted, surgical fixes:

- एक specific safety failure mode → 200-500 SFT examples better behavior teach करते हुए।
- A formatting bug (एक corner case में model broken JSON output करता है) → corrected examples पर SFT।
- A refusal regression (model harmless requests decline करता है) → permissive examples पर SFT।
- Post-launch spotted regional/cultural bias।

Patches **same pipeline use करते हैं जो point releases use करते हैं**, बस smaller, more targeted data के साथ। Often existing weights के top पर एक LoRA merged in। Calendar: hours to days।

---

## 6. Per Release Tier Data Pipeline

क्योंकि हर tier पर focus shifts होता है, data pipeline भी होती है। यहां क्या change होता है:

| Pipeline component | Major | Minor | Point | Patch |
|---------------------|-------|-------|-------|-------|
| **Raw crawl / web data** | full reset, multi-TB | new data refresh | — | — |
| **Filtering classifiers** | scratch से retrained | reused या fine-tuned | — | — |
| **Tokenization** | new tokenizer trained | same tokenizer | same | same |
| **Sharding** | full re-shard, billions of files | new shards append | — | — |
| **Deduplication** | whole corpus पर global MinHash | new vs old dedup | — | — |
| **Mid-train mix** | designed | final-stage mix replace | — | — |
| **SFT data** | whole new mix curated | extended | refreshed | targeted additions |
| **Preference pairs** | scratch से | augmented | refreshed | targeted |
| **Eval suite** | full reset, including new benchmarks | extended | extended | spot checks |

Tooling stack इसे mirror करता है:

```
Tier              Primary data tooling
─────────────────────────────────────────────────────────
Major          datatrove, mosaicml-streaming, dedup के लिए Spark,
               Ray Data, fastText classifiers, n-gram decontam
Minor          add: contextual filters, anneal data assembly,
               long-context document selection
Point          alignment-handbook data prep, judge-based filtering,
               preference-data labeling pipelines
Patch          targeted SFT collection, often hand-curated
```

---

## 7. Tool Stack — Role से

जो tool आप अभी जो role play कर रहे हो उस पर depend करता है। हर role का अपना canonical 2026 stack है।

### Pretraining Engineer

- **`torchtitan`** — PyTorch का reference pretraining framework। FSDP2, TP, PP, activation checkpointing, FP8।
- **Megatron-LM / Megatron-Core** (NVIDIA) — frontier scale पर battle-tested।
- **NeMo** (NVIDIA) — production framework, end-to-end including data और post-training।
- **Levanter** (Stanford) — JAX-based, reproducible, hashable runs।
- **`llm.c`** (Karpathy) — pure C/CUDA reference, educational gold।
- **OLMo trainer** (AI2) — fully open recipe।

### Post-training (Fine-tune / SFT / DPO / GRPO) Engineer

- **`trl`** (HuggingFace) — `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `KTOTrainer`। Canonical।
- **`peft`** — LoRA / QLoRA / DoRA / VeRA implementations।
- **`alignment-handbook`** (HuggingFace) — production-grade scripts।
- **`axolotl`** — popular YAML-driven trainer।
- **`unsloth`** — consumer GPUs पर 2-3× faster QLoRA।
- **`OpenRLHF`**, **`OpenR1`** — RL/RLVR pipelines।

### Inference / Serving Engineer

- **vLLM** — production default।
- **SGLang** — prefix-cache-heavy workloads के लिए best।
- **TensorRT-LLM** — best NVIDIA latency।
- **TGI** — HuggingFace का serving stack।
- **`llama.cpp` / Ollama** — CPU + edge।
- **MLX** — Apple Silicon।

### Research / Interpretability Engineer

- **TransformerLens** (Neel Nanda et al.) — mechanistic interpretability के लिए *the* standard। हर activation hook करता है; `model.run_with_cache(...)`।
- **NNsight** (David Bau lab) — context-managed forward passes; `nnsight-server` के via huge models पर काम करता है।
- **`captum`** (Meta) — feature attribution, integrated gradients, layer attributions।
- **`sae_lens`** — feature discovery के लिए sparse autoencoder training & inference।
- **`circuitsvis`** — Jupyter में interactive attention/circuit visualization।
- **`inspectus`** — attention pattern visualization।
- **Goodfire / Transluce / Anthropic's Reach** — commercial / hosted SAE / interpretability tools।

### Observability / Monitoring Engineer

- **W&B (Weights & Biases)** — research और training run tracking। Default।
- **TensorBoard** — local, simple।
- **MLflow** — experiment tracking।
- **Aim** — open-source W&B alternative।
- **LangSmith / Langfuse / Braintrust / Helicone / Phoenix** — *production* LLM observability (traces, judge runs, cost)।
- **Prometheus + Grafana + OpenTelemetry** — serving के लिए infra-side metrics (chapter 17)।
- **`wandb-weave`** (W&B) — newer LLM-specific traces।

### Testing / Eval Engineer

- **`lm-evaluation-harness`** — capability benchmarks।
- **`promptfoo`** — prompt comparison + CI।
- **`inspect-ai`** (UK AISI) — capability + safety eval framework।
- **`deepeval`** — Python-friendly LLM eval।
- **`RAGAS`** — RAG-specific।
- **`pytest`** — आपका existing Python test runner; golden-set eval के साथ combine करो।
- **`Garak`** (NVIDIA) — vulnerability scanning।
- **Arena-Hard-Auto / WildBench** — pairwise judge-based eval।

A typical 2026 product team active use में हर category के 2-4 रखती है।

---

## 8. Model के अंदर Look करना — Mental Model

A trained transformer के पास दो kinds of state हैं:

- **Weights** — आपने disk पर save किए parameters। Inference पर frozen।
- **Activations** — values जो forward pass के दौरान network से *flow* होती हैं। हर input के लिए different।

Debug करने के लिए:

- **Weights inspect करो** जब आप पूछो "model खुद कैसा है?" (e.g., एक layer के projection का norm, baked in attention pattern templates, neuron directions)।
- **Activations inspect करो** जब आप पूछो "इस specific input पर model क्या कर रहा है?" (e.g., layer 17 अगले token के बारे में क्या सोचता है, कौन से heads कहां attend करते हैं)।

दोनों accessible हैं। उन तक पहुंचने का harness guessing और knowing के बीच का difference है।

---

## 9. Forward Hooks — Simplest Possible Debugger

Native PyTorch। तीन lines:

```python
activations = {}
def hook(name):
    def fn(module, inputs, output):
        activations[name] = output.detach().cpu()
    return fn

# attach
handle = model.layers[17].self_attn.register_forward_hook(hook("layer17_attn"))

# run
model(input_ids)

# inspect
print(activations["layer17_attn"].shape)        # (B, T, D)
print(activations["layer17_attn"].std())

# detach
handle.remove()
```

आप **किसी भी** `nn.Module` को hook कर सकते हो — हर linear, हर attention, हर FFN। `register_forward_hook` का output एक handle है जिस पर आप done होने पर `.remove()` call करते हो; उसके बिना, आप GPU memory leak करते हो।

A quick layer-by-layer sanity check के लिए:

```python
acts = {}
hooks = []
for i, layer in enumerate(model.layers):
    hooks.append(layer.register_forward_hook(
        lambda m, _, o, i=i: acts.setdefault(f"l{i}", o.detach().std().item())
    ))
model(input_ids)
for h in hooks: h.remove()
print({k: round(v, 3) for k, v in acts.items()})
# {'l0': 0.21, 'l1': 0.34, ..., 'l31': 1.05}
```

अगर एक layer का output std अपने neighbors से wildly different है, कुछ off है — इस तरह आप seconds में broken layer find करते हो।

---

## 10. TransformerLens — Proper Interpretability Tool

Real interpretability work के लिए, `transformer_lens` standard है। किसी भी HuggingFace model को load करता है, हर internal को named activation के रूप में expose करता है, कुछ भी hook करने देता है।

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B")
tokens = model.to_tokens("The cat sat on the")

logits, cache = model.run_with_cache(tokens)

# हर activation named और accessible है:
cache["blocks.10.attn.hook_pattern"].shape       # (B, H, T, T) — attention pattern
cache["blocks.10.hook_resid_post"].shape         # (B, T, D) — block 10 के बाद residual
cache["blocks.10.attn.hook_q"].shape             # (B, T, H, d) — Q vectors
```

`run_with_cache` forward run करता है और **हर चीज़** internally store करता है। उसके बाद, आप कोई भी layer, कोई भी head, कोई भी neuron inspect कर सकते हो। ये field का workhorse है।

---

## 11. Logit Lens — Model *अब* क्या सोच रहा है?

Trick: सिर्फ़ final layer के output पर LM head run करने के बजाय, **हर** intermediate layer के residual stream पर run करो। आप per-layer prediction पाते हो "अगर मैं यहां stop करूं तो next token क्या होगा?"

```python
# 5 lines में logit lens
final_norm = model.ln_f                  # head से पहले final norm
unembed    = model.unembed.W_U           # (D, V) tied or untied LM head

for layer_idx in range(model.cfg.n_layers):
    resid = cache[f"blocks.{layer_idx}.hook_resid_post"]    # (B, T, D)
    logits_at_layer = final_norm(resid) @ unembed           # (B, T, V)
    top = logits_at_layer[0, -1].argmax().item()
    print(layer_idx, model.to_string(top))
```

आप कुछ ऐसा देखोगे:

```
0  the          (random)
1  the          (still random)
...
8  is           (model converge करना start कर रहा है)
12 mat
16 mat
20 mat          (final answer locked in)
24 mat
```

Layer जहां model "decide" करता है आपको hint देती है उस input के लिए कौन सी layers work करती हैं। कई inputs के across apply करो: आप पाते हो कुछ prompts layer 12 पर decide करते हैं, कुछ layer 28 पर — आपको intuition देता है कि कौन से depths क्या handle करते हैं।

**Tuned lens** (Belrose et al. 2023) supervised upgrade है: हर layer पर final norm + unembed use करने के बजाय, हर layer से next-token predict करने के लिए एक small per-layer linear probe train करो। More accurate, especially middle में। `tuned-lens` package implement करता है।

---

## 12. Ablation: एक चीज़ Change करो ये Find करने के लिए कि वो क्या करती है

The mechanistic interpretability mantra: **एक component क्या करता है जानने के लिए, उसे ablate करो और देखो क्या break होता है।**

### एक Attention Head Ablate करो

```python
def zero_head(module, inputs, output, head_idx):
    # output: (B, T, H, d). head_idx zero करो।
    output[:, :, head_idx, :] = 0
    return output

handle = model.blocks[10].attn.register_forward_hook(
    lambda m, i, o: zero_head(m, i, o, head_idx=3)
)
loss_with_ablation = compute_loss(model, eval_set)
handle.remove()
```

ये हर head पर हर layer के लिए run करो। Heads जिनकी ablation एक specific task पर most hurt करती है वो वो हैं जो *वो* task करते हैं। Anthropic की Indirect Object Identification (IOI) circuit exactly इस तरह discovered हुई थी।

### एक Layer Ablate करो

```python
def skip_layer(module, inputs, output):
    # block के output को उसके input से replace करो — skip करने के equivalent।
    return inputs[0]

handle = model.blocks[15].register_forward_hook(skip_layer)
# evaluate; remove
```

### FFN में एक Neuron Ablate करो

```python
def zero_neuron(module, inputs, output, idx):
    output[..., idx] = 0     # MLP intermediate (B, T, F) है; col idx zero करो
    return output

handle = model.blocks[10].mlp.gate_proj.register_forward_hook(
    lambda m, i, o: zero_neuron(m, i, o, idx=2048)
)
```

### एक Weight Ablate करो (एक Literal Scalar)

```python
# Backup keep करो, modify करो, run करो, restore करो।
W = model.blocks[10].mlp.gate_proj.weight
backup = W[5, 1024].clone()
W.data[5, 1024] = 0.0
# evaluate
W.data[5, 1024] = backup    # restore
```

ये extreme लगता है, लेकिन surgical questions के लिए ("क्या *ये exact* weight matter करता है?") cleanest experiment है।

### Mean Ablation vs Zero Ablation

चीज़ें zero पर set करना convenient है लेकिन unrealistic भी — model training के दौरान कभी pure-zero activations नहीं देखता। **Mean ablation** more honest है: activation को dataset के across उसके mean से replace करो।

```python
mean_acts = []
for batch in dataset:
    out = model.run_with_cache(batch.tokens)[1]["blocks.10.hook_z"]
    mean_acts.append(out.mean(dim=(0,1)))
mean_z = torch.stack(mean_acts).mean(dim=0)   # (H, d)

def mean_ablate(m, i, o):
    o[:, :, 3] = mean_z[3]    # head 3 को dataset mean से replace करो
    return o
```

Mean ablation results usually "इस component के बिना model क्या करता" के closer होते हैं zero ablation से।

---

## 13. Activation Patching — Most Powerful Technique

Zeroing out के बजाय, **एक input से दूसरे पर activation copy करो**। अगर second input अब उस point पर जैसे first किया था वैसे behave करता है, आपने behavior localize कर लिया।

A canonical example: "एक clean run से एक corrupted run पर indirect-object position पर layer 8, head 5 का attention output patch करो।" अगर corrupted run अब right answer पाता है, आपने show कर दिया कि layer 8 में head 5 indirect-object information carry करता है।

```python
# pseudo-code, TransformerLens इसे much cleaner बनाता है
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupt_logits, _         = model(corrupt_tokens)

def patch(module, inputs, output, layer, head, pos, clean_value):
    output[:, pos, head, :] = clean_value
    return output

handle = model.blocks[8].attn.register_forward_hook(
    lambda m, i, o: patch(m, i, o, layer=8, head=5, pos=4,
                           clean_value=clean_cache["blocks.8.attn.hook_z"][0, 4, 5])
)
patched_logits = model(corrupt_tokens)
# patched_logits को clean_logits से compare करो — अगर close है, head 5 culprit था।
```

`transformer_lens.patching` ये one-liners के रूप में provide करता है (`patching.get_act_patch_attn_head_out_all_pos(...)`)।

---

## 14. Sparse Autoencoders (SAEs) — 2024-2026 Revolution

एक transformer में activations *dense* हैं — हर neuron कई चीज़ों पर fire करता है, model जो भी "thing" जानता है वो कई neurons में spread है। **Sparse Autoencoders** एक wider, sparse layer train करते हैं dense activations को individual interpretable features में decompose करने के लिए।

Recipe:
1. एक large dataset पर forward passes run करो, कुछ layer पर activations capture करो।
2. एक SAE train करो: encoder (D → many * D, e.g., 8×D) + ReLU sparsity + decoder (वापस D), reconstruction + sparsity loss के साथ।
3. SAE features interpretable directions हैं: हर एक एक specific semantic concept पर fire करता है।

Anthropic का "Scaling Monosemanticity" work और Goodfire के product-grade interpretability ऐसे ही operate करते हैं।

Tools:
- **`sae_lens`** (Joseph Bloom et al.) — open-source SAE training और analysis।
- **Goodfire** — hosted SAE-based feature search और intervention।
- **Anthropic's Reach (internal)** — hosted with public APIs partial।

एक बार आपके पास SAE features हों, आप कर सकते हो:
- **Search** उस feature के लिए जो "SQL injections handle करने वाले code" पर fire करे।
- **Steer** residual stream में feature direction add करके — model को ऐसे behave करवाते हुए जैसे उसने अभी उस concept के बारे में सोचा हो।
- **Deployment पर Intervene** unwanted features को suppress करने के लिए (safety) या desired ones को amplify करने के लिए।

A 2026-grade interpretability product कुछ ऐसा दिखता है: multiple layers पर SAEs trained, browsable feature catalog, deploy-time intervention API। **ये model debugging का new frontier है।**

---

## 15. इसे Together रखना — एक Real Failure Debug करना

Suppose आपका model occasionally normal chat के बीच में `</think>` output करता है — thinking-mode tokens से एक leakage। आप find करना चाहते हो क्यों।

1. **Reproduce करो।** 50 prompts find करो जो trigger करें।
2. **हर layer का residual norm forward-pass hook करो।** देखो norm एक specific layer पर explode करता है क्या जब bad token आता है। Often yes।
3. **TransformerLens cache + logit lens।** हर layer पर unembed के through residual project करो। देखो `</think>` first कहां top prediction बनता है। Say layer 22।
4. **Layer 22 में हर head ablate करो।** 1-2 heads find करो जिनकी ablation leak remove करती है।
5. **उन heads के attention patterns inspect करो।** वो probably specific positions या token types पर attend करते हैं। `circuitsvis` से visualize करो।
6. **Patching से confirm करो।** Buggy input पर एक clean run से head का output patch in करो — क्या leak जाता है?
7. **Fix।** या तो: leak avoid करते हुए कुछ hundred examples पर SFT, OR (fancy fix) layer 22 पर एक SAE train करो और inference पर relevant feature down steer करो, OR (surgical fix) head के output projection में एक specific weight zero करो।

ये है कैसे 2026 की एक real interpretability team actually production models debug करती है। आपको एक special library की need नहीं; जो चाहिए वो है *mental model* — weights vs activations, hooks, ablate-and-test।

---

## 16. एक New Version Ship करने का Complete Workflow

पूरे chapter को together रखना:

```
[research / data / arch decision]
   │
   ▼
[major release]      —— fresh pretrain, full data pipeline
   │
   ▼
[evaluation gauntlet] —— lm-eval-harness, internal evals, golden sets
   │
   ▼
[mid-train / minor]   —— better data पर continual pretrain, maybe modality add करो
   │
   ▼
[post-training (point release)] —— SFT + DPO/GRPO, alignment-handbook
   │
   ▼
[red-team / safety patches]    —— targeted SFT, content classifiers
   │
   ▼
[shipping (point या patch)]
   │
   ▼
[CI में observability + golden set]  —— LangSmith / Braintrust / promptfoo
   │
   ▼
[interpretability investigation]    —— failures के लिए: hooks, logit lens, SAEs
   │
   ▼
[next iteration]  ── cycle के top पर वापस
```

हर arrow का अपना tooling, calendar, और risk profile है। **Pretraining decisions slow और expensive हैं; post-training fast और cheap है। अपनी problem को right tier से match करो।**

---

## 17. Cheat Sheet

- **Major (X → X+1) = fresh pretrain।** Months। New arch / tokenizer allowed।
- **Minor (X → X.5) = continual pretrain।** Weeks। Same arch और weights।
- **Point (X.5 → X.6 / X.7) = सिर्फ़ post-training।** Days। SFT/DPO/GRPO।
- **Patch (X.7 → X.7.1) = targeted post-training।** Hours-days।
- **Catastrophic forgetting** continual pretrain में biggest risk है। 30-60% general data mix करो।
- **Per role tooling**: pretrain के लिए `torchtitan`/Megatron, post-train के लिए `trl`/`peft`/`axolotl`/`unsloth`, serving के लिए vLLM/SGLang, research के लिए TransformerLens/NNsight, observability के लिए W&B/Langfuse, testing के लिए lm-eval/promptfoo/inspect-ai।
- **PyTorch forward hooks** model के अंदर देखने का simplest तरीका हैं। Three lines।
- **TransformerLens `run_with_cache`** हर activation capture करता है, named और addressable।
- **Logit lens** intermediate residuals को LM head के through project करता है — देखो model हर layer पर क्या सोचता है।
- **Ablation**: एक head/layer/neuron/weight zero करो, देखो क्या breaks। Single most reliable debugging technique।
- **Mean ablation > zero ablation** honest results के लिए।
- **Activation patching** forward passes के बीच activations copy करके behavior localize करता है।
- **Sparse Autoencoders (SAEs)** dense activations को interpretable features में decompose करते हैं — 2026 frontier।

---

## 18. और गहराई से

- **Anthropic's "Scaling Monosemanticity"** और **"Toy Models of Superposition"** — canonical SAE / mechanistic interp papers।
- **TransformerLens docs** at `transformerlensorg.github.io/TransformerLens/` — tutorials पढ़ो।
- **NNsight docs** — modern, scalable variant के लिए।
- **Belrose et al. 2023** — Tuned Lens।
- **Anthropic / OpenAI / DeepMind interpretability research blogs** — best public technical writeups।
- **`sae_lens` GitHub** — open-source SAE training, Anthropic के work का public reproduction।
- **Llama 3 / 3.1 / 3.2 / 3.3 / 4 model cards** — order में पढ़ो ताकि एक real release ladder को public में play out होते देखो।
- **DeepSeek-V2 → V3 → V3.1 release notes** — major-vs-minor decisions explained का clearest example।
- **Karpathy's `llm.c`** — end-to-end pretraining आप पढ़ सकते हो।
- **`torchtitan`** — 2026 में cleanest open-source pretraining framework।
- **MATS / SERI MATS programs और Apollo Research blogs** — community-driven interpretability research।

ये है full release-engineering और debug-the-internals chapter। Together with chapter 14 (training) और chapter 19 (eval), ये production-LLM lifecycle complete करता है: random init से deployed model तक users जो failures find करते हैं उन्हें debug करने तक।
