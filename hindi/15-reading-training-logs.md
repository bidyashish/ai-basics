# 15 · Training Logs पढ़ना — W&B Charts और उनका मतलब

> **TL;DR** Language model train करना 4 days से 3 months का commitment है। आप उसका ज़्यादातर time **charts देखते हुए** spend करोगे। ये file log करने worth metrics list करती है, हर chart कैसा दिखना चाहिए जब चीज़ें सही जा रही हों, और जब वो ग़लत जाएं तो specific shapes का क्या मतलब है। **Weights & Biases (W&B), TensorBoard, और Comet सब same काम करते हैं**; हम W&B example के रूप में use करेंगे क्योंकि ये LLM teams के लिए 2026 default है।

---

## 1. 30 seconds में Setup

```bash
pip install wandb
wandb login
```

```python
import wandb, math, time

wandb.init(
    project='small-llm',
    name=f'qwen-1b-{time.strftime("%Y%m%d-%H%M%S")}',
    config={
        'D': 1536, 'L': 28, 'H_q': 12, 'H_kv': 4, 'lr': 3e-4,
        'batch_size': 256, 'block_size': 4096, 'optimizer': 'AdamW',
        'data': 'fineweb-edu-100B',
    },
    save_code=True,
)
```

फिर loop में:

```python
wandb.log({'train/loss': loss, 'train/lr': lr, ...}, step=global_step)
```

ये पूरा API है। नीचे जो भी है *क्या* log करना है।

---

## 2. Minimum Viable Log Set

हर step हमेशा log करो (cheap):

| Key | क्या | क्यों |
|-----|------|-----|
| `train/loss` | step के batch पर mean cross-entropy | वो only metric जो ultimately matter करता है |
| `train/grad_norm` | `torch.nn.utils.clip_grad_norm_` ये return करता है | spike detection |
| `train/lr` | `sched.get_last_lr()[0]` | confirm करो schedule अपना काम कर रहा है |
| `train/tokens_per_sec` | throughput | regression detection |
| `train/step_time_ms` | per step wall time | hardware health |
| `train/grad_clipped_frac` | clip कितनी बार active था | बताता है `1.0` too tight है क्या |

हर N (e.g., 500) steps:

| Key | क्या |
|-----|------|
| `eval/loss` | held-out cross-entropy |
| `eval/ppl` | `exp(eval/loss)` |
| `eval/by_domain/{web,code,math}/ppl` | per-domain perplexity |
| `eval/mmlu_acc` (हर 5k) | downstream benchmark |
| `eval/gsm8k_acc` (हर 5k) | reasoning benchmark |

Per-parameter health (हर 1k steps):

| Key | क्या |
|-----|------|
| `params/<layer>.weight_norm` | per-layer L2 norm |
| `grads/<layer>.grad_norm` | per-layer gradient L2 |
| `acts/<layer>.std` | activation std (1 batch sample) |

Checkpointing event:

| Key | क्या |
|-----|------|
| `ckpt/saved_at_step` | timestamp marker |
| `ckpt/path` | resume के लिए |

---

## 3. Healthy Loss Curve का Shape

LLM के लिए healthy `train/loss`:

```
              .
              ..
                .
                 ..
                   ...
                      .....
                           ...........__________
^                                                 → smooth, monotonic
|
loss
|
+--------------------- steps -------------------->
warmup     fast drop          long tail
(steep,    (logs:                slowly approaches
spiky)      power-law-ish)        irreducible loss
```

Key features:
- First ~1k steps में **Initial steep drop** — model random से "knows English" तक जाता है।
- **Power-law tail** — log-log axes पर, loss roughly linearly decline करती है। (W&B के पास "log scale" toggle है।) अगर आपका curve power law नहीं है, कुछ off है।
- **Final asymptote** — हर dataset का एक irreducible loss है; आप इसके few % के अंदर पहुंचोगे लेकिन उसे कभी beat नहीं करोगे।

FineWeb-Edu पर 1.1B model के लिए, expect करो:
- step 1k पर `loss < 4.0` (post-warmup)
- step 30k तक `loss ≈ 2.5`
- step 100k तक `loss ≈ 2.2`
- step 200k तक `loss ≈ 2.0`

Different datasets के different floors हैं — code/math/multilingual सब इसे ऊपर push करते हैं।

---

## 4. Loss Chart पर Various Pathologies कैसी दिखती हैं

### Loss inf/NaN तक diverge करता है early

```
loss
|        .'
|        |
|        |   ← NaN
|       /
|      /
|     /
|____/
+----- steps
```
**Causes:** LR too high, fp16 overflow, weight init too big।
**Fix:** LR 10× lower करो, bf16 पर switch करो, longer warmup, lower init std (try 0.02 के बजाय 0.01)।

### Loss Plateau — step 0 से flat

```
loss
|________________
|
+-------- steps
```
**Causes:** model नहीं सीख रहा। Tokenizer mismatch (data integers हैं एक vocab में, model किसी और को expect करता है), LR 0 है, optimizer step नहीं कर रहा (`opt.step()` भूल गए), कहीं `requires_grad=False`।
**Fix:** first batch tokens decoded print करो; verify करो `opt.step()` actually weights change करता है; `loss.backward()` के बाद check करो `param.grad.norm()` nonzero है।

### Saw-tooth

```
loss
|   /\    /\    /\
|  /  \  /  \  /  \
| /    \/    \/    \____
+--------- steps
```
**Causes:** epoch boundary effect (same data revisited होने पर loss jumps), bad shard ordering, saturation के पास LR too high।
**Fix:** epochs के across shards shuffle करो, LR 2× drop करो, ensure `pin_memory + non_blocking` data load overlap करने के लिए।

### Sudden Spike (1-3 steps), recovers

```
loss
|     .
|    /|
|   / |
|__/  |__________
+------- steps
```
**Causes:** rare junk batch (एक ad-spam document), gradient anomaly. Single-batch spikes scale पर normal हैं (~once per 10k steps)।
**Fix:** harmless अगर recover करे। अगर recover न करे और आपको revert करना पड़े: **skip-on-spike** logic implement करो — अगर `loss > median × 5`, optimizer step skip करो। Spike से just पहले एक checkpoint save करो inspection के लिए।

### Validation up जबकि train down

```
       train/loss
          ___________
   loss /
   |  /            eval/loss
   | / .........-----'
   |/  .....----'
   +------ steps
```
**Causes:** overfitting (small datasets / multi-epoch पर more likely)।
**Fix:** more data, fewer epochs, slight dropout (0.05), more regularization. 100B+ tokens पर 1-pass trained LLMs के लिए, ये almost कभी नहीं होता — train और val together track करते हैं।

### Eval Flat जबकि Train falls (memorization)

Train loss ~0 तक जाती है; eval loss high रहती है। मतलब model training data verbatim memorize कर रहा है। **Decontamination** चलाओ और epochs कम करो।

---

## 5. Gradient Norm — Second Most Important Chart

`grad_norm` should:
- **First 100 steps में spike high** (model optimal से far है)।
- एक roughly steady value पर **Settle**, e.g. 0.5 - 2.0।
- जैसे model converge करे **Slowly decrease** हो।

Patterns:

- **Constantly clip threshold (1.0) hit करना** → LR too high है, या training unstable है। LR lower करो।
- **Suddenly explodes** → bad batch आ रहा है। Skipped optimizer steps देखो।
- **~0 तक जाता है** → model ने सीखना बंद कर दिया। या तो आप done हो (rare) या कुछ freeze हो गया।

हमेशा **`grad_norm`** (pre-clip) और **clip ratio** (`grad_norm / max_norm`) दोनों log करो। अगर 50% steps clip करते हैं, आप signal throw कर रहे हो।

---

## 6. Learning-rate Chart — Easiest वाला

```
lr
|        ______________
|       /              \____
|      /                    \____
|     /                          \____
|____/                                \____
+--------- steps
warmup   stable / cosine               anneal
```

क्या देखें:
- Shape match करता है जो आपने configure किया। (आप surprised होंगे कि कितनी बार नहीं करता।)
- LR mid-run unexpectedly spike या drop नहीं करता।
- Anneal phase actually ~0 तक decay करता है (कुछ implementations final phase भूल जाते हैं)।

---

## 7. Throughput / Step Time

दो charts: `train/tokens_per_sec` और `train/step_time_ms`। वो different stories बताते हैं।

- **Stable tokens/sec great है।** Downward drift = कहीं regression है। Downward spikes = network instability (multi-node), data loader stalled, GPU thermal throttling।
- **Step time variance > 30%** = आपका data loader keep up नहीं कर सकता। `num_workers`, `prefetch_factor` बढ़ाओ, dataset code simplify करो, या expensive transforms offline move करो।

Multi-GPU run में, per-GPU throughput देखने के लिए `tokens_per_sec / num_GPUs` log करो। ये scale करने पर roughly constant होना चाहिए।

A typical 1B model 8× H100 पर FSDP2 और bf16 + `torch.compile` के साथ: **~50 000-90 000 tokens/sec**। उसका आधा से कम = कुछ ग़लत है।

---

## 8. Per-layer Norms — Silent Canaries

```python
if step % 1000 == 0:
    for name, p in model.named_parameters():
        if p.grad is not None:
            wandb.log({
                f'params/{name}/norm': p.detach().norm().item(),
                f'grads/{name}/norm': p.grad.detach().norm().item(),
            }, step=step)
```

ये hundreds of lines produce करते हैं लेकिन dividends pay करते हैं:

- **सारे weight norms grow करते हैं** (gradient signal उन तक पहुंच रहा है) — अच्छा।
- **एक layer का weight norm explodes** जबकि बाकी नहीं — init में bug या वो layer unstable है। Inspect करो।
- **एक layer का gradient norm ~0 है** — ये train नहीं हो रहा। Often एक frozen-by-mistake layer या unused parameter।
- **एक layer का gradient norm huge है** जबकि बाकी small हैं — वो layer सारा work ले रही है। Init या LR adjust करो।

Deep models के लिए, **last layer** और **embedding** usual outliers हैं; उन्हें special LR या weight decay groups चाहिए।

---

## 9. Activation Stats — Deep Diagnostic

Forward-hook कुछ layers और उनका output `std` log करो:

```python
acts = {}
def hook(name):
    def fn(_, __, out):
        acts[name] = out.detach().std().item()
    return fn

for i, layer in enumerate(model.layers[::4]):    # हर 4th layer
    layer.register_forward_hook(hook(f'block_{i}'))

# forward के बाद
wandb.log({f'acts/{k}': v for k, v in acts.items()}, step=step)
```

- **Stds layers के across roughly constant रहने चाहिए** (pre-norm + good init ये करता है)।
- **Depth के साथ growing std** → activations explode कर रही हैं। QK-norm या sandwich-norm add करो।
- **Depth के साथ decreasing std** → activations vanish कर रही हैं। Init issue या norms misconfigured।
- **Suddenly 0 तक जाना** → dead layer, residual paths देखो।

---

## 10. Eval Charts — सच

`train/loss` वो है जो आप hour-to-hour देखते हो। `eval/*` वो है जो बताता है कि आपका model *बेहतर हो रहा है या नहीं*।

- **eval/ppl** train/loss को closely follow करना चाहिए।
- **mmlu_acc, hellaswag_acc** smoothly improve होने चाहिए। Small models random के पास start करते हैं (MMLU 5-way पर ~25%) और size और data के depending ~30-50% तक पहुंचते हैं।
- **gsm8k_acc** small scale पर noisy है (often <5% जब तक model ~3B cross न करे)। Panic मत करो।
- **humaneval_pass@1** likewise noisy जब तक आपके पास pretraining में lots of code न हो।

W&B "panel groups" आपको सारे eval metrics एक साथ रखने देते हैं। एक "Pretraining benchmarks" group set up करो और इसे pin करो।

---

## 11. "Compute Spent" Chart

x-axis के रूप में step number के बजाय **cumulative tokens trained** log करो। ये different batch sizes वाले runs के across far more comparable है।

```python
tokens_so_far = step * effective_tokens_per_step
wandb.log({'progress/tokens': tokens_so_far,
           'progress/loss_vs_tokens': loss}, step=tokens_so_far)
```

फिर W&B में अपने loss chart पर "x-axis = progress/tokens" set करो। Suddenly आपके सारे runs एक-दूसरे के साथ comparable हैं — और public scaling-law fits के साथ भी।

---

## 12. Run Comparisons (W&B Superpower)

W&B का point एक single chart नहीं है; ये same axes पर **dozens of runs** overlay करना है। Common patterns:

- **LR sweeping**: `lr ∈ {1e-4, 2e-4, 3e-4, 5e-4, 1e-3}` के साथ 5 runs। Best find करो।
- **Data mix sweeping**: same model, different data ratios। Winner pick करो।
- **Optimizer A vs B**: AdamW vs Muon, देखो per-token कौन जीतता है।
- **एक architectural tweak के साथ vs बिना**: QK-norm on/off; tied embeddings on/off।

Credible comparisons के लिए:
- **एक के अलावा सारे variables constant रखो।**
- जब possible हो **same seed use करो** (W&B by default seed log करता है)।
- जब batch size differ हो तो **steps के बजाय tokens vs plot करो**।
- Large-scale outcomes predict करने के लिए small-scale runs पर **scaling-law line fit करो**।

W&B "Reports" आपको charts pin करने और comments लिखने देते हैं — postmortems के लिए उन्हें use करो। Future-you current-you का thanks करेगा।

---

## 13. Text Samples Logging

हर 1k-5k steps, current model से generate करो और log करो:

```python
def log_samples(model, prompt_set, step, max_new=64):
    model.eval()
    rows = []
    with torch.inference_mode():
        for p in prompt_set:
            ids = tokenize(p).cuda()
            gen = model.generate(ids, max_new_tokens=max_new)
            rows.append([step, p, decode(gen)])
    table = wandb.Table(columns=['step','prompt','completion'], data=rows)
    wandb.log({'samples/text': table}, step=step)
    model.train()
```

A 4-prompt fixed set काफी है। Model को gibberish → broken English → coherent prose → genuinely interesting होते देखना pretraining में most motivating चीज़ है। ये एक bad SFT round spot करने का fastest तरीका भी है।

---

## 14. Alerts और Notifications

Catastrophes के लिए W&B alerts set करो:

```python
if not math.isfinite(loss):
    wandb.alert(title='NaN loss', text=f'step {step}', level=wandb.AlertLevel.ERROR)
```

जब ये fire करेगा आप bed में होंगे। जानना better है।

Long runs के लिए, `wandb_alerts.py` या एक small sidecar के via Slack notifications set करो जो run को poll करे।

---

## 15. W&B के Beyond

- **TensorBoard** — open-source, local. Same idea. Great अगर आपका data network से बाहर नहीं जा सकता।
- **MLflow** — traditional ML के साथ experiment tracking के लिए better; streaming charts पर weaker।
- **Aim** — open-source W&B alternative, fast UI।
- **TrackioAI**, **Comet ML**, **Neptune** — paid alternatives।

2026 में LLM community **research के लिए W&B और cluster sanity checks के लिए TensorBoard** को default करती है। Most large training jobs (Llama, Qwen, OLMo) अपने tech reports में W&B-style training curves publish करते हैं।

---

## 16. Common Dashboard Layout

A panel layout जो long run के एक hour के अंदर ख़ुद के लिए pay कर देता है:

1. **Top row (4 panels)**: `train/loss`, `eval/loss`, `eval/ppl`, `train/grad_norm`।
2. **Second row**: `train/lr`, `train/tokens_per_sec`, `train/step_time_ms`, `train/grad_clipped_frac`।
3. **Third row**: `acts/block_0/std`, `acts/block_<L/2>/std`, `acts/block_<L-1>/std`, *something domain-specific*।
4. **Fourth row**: downstream evals (`mmlu_acc`, `hellaswag_acc`, `gsm8k_acc`, `humaneval_pass@1`)।
5. **Below**: text-sample table।
6. **Side panel**: किसी और run के against run config diff।

ये layout pin करो। हर run के लिए इसे keep करो। Overlay करके runs के across compare करो।

---

## 17. "एक नए run के पहले 30 minutes" Checklist

`train.py` hit करने के बाद:

- [ ] W&B run started, सही project के साथ tagged।
- [ ] First step 60 seconds के अंदर logged।
- [ ] `train/loss` finite है।
- [ ] `train/lr` schedule से match करता है (just past 0)।
- [ ] `train/tokens_per_sec` expected range में है।
- [ ] सारे GPUs near 100% utilization पर (`nvidia-smi`)।
- [ ] कोई memory pressure नहीं (>10% headroom)।
- [ ] 500 steps के बाद, loss meaningfully drop हुई है।
- [ ] Gradient norm stabilize हो गया है।

अगर इनमें से कोई off है, **रुक जाओ और अभी fix करो**। ग़लत start हुआ 4-day run एक 4-day run lost है।

---

## 18. 2026 Cheat Sheet

- **हर step loss log करो**, हर 500-2000 eval, हर 5-10k benchmarks।
- **Steps के बजाय tokens vs plot करो**, cross-run comparability के लिए।
- **Grad norm और grad-clipped fraction watch करो** — वो instability predict करते हैं।
- हर 1k steps **Per-layer norms**; वो silent bugs reveal करते हैं।
- **Text sample** to a W&B table — fastest qualitative signal।
- **एक dashboard layout pin करो** और reuse करो।
- NaN और stalled throughput के लिए **alerts enable करो**।
- एक known-bad event से just पहले **checkpoint snapshot करो** ताकि आप offline debug कर सको।

---

## और गहराई से

- W&B docs का "LLM training" guide।
- **`torchtitan`** logs — एक serious distributed-training setup क्या track करता है इसका public reference।
- OLMo / SmolLM-2 / Llama 3 training reports — वो often अपनी actual W&B charts publish करते हैं, "अच्छा कैसा दिखता है" के लिए excellent calibration।
- **`wandb-sweeps`** small-scale ablations पर hyperparameter optimization के लिए।

That's curriculum का end. **अब कुछ train करो और charts पढ़ो।**
