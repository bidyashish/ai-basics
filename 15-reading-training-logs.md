# 15 · Reading Training Logs — W&B Charts and What They Mean

> **TL;DR** Training a language model is a 4-day to 3-month commitment. You will spend most of that time **looking at charts**. This file lists the metrics worth logging, what each chart should look like when things go right, and what specific shapes mean when they go wrong. **Weights & Biases (W&B), TensorBoard, and Comet all do the same thing**; we'll use W&B as the example because it's the 2026 default for LLM teams.

---

## 1. Setup in 30 seconds

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

Then in the loop:

```python
wandb.log({'train/loss': loss, 'train/lr': lr, ...}, step=global_step)
```

That's the whole API. Everything below is *what* to log.

---

## 2. The minimum viable log set

Always log every step (cheap):

| Key | What | Why |
|-----|------|-----|
| `train/loss` | mean cross-entropy on the step's batch | the only metric that ultimately matters |
| `train/grad_norm` | `torch.nn.utils.clip_grad_norm_` returns this | spike detection |
| `train/lr` | `sched.get_last_lr()[0]` | confirm schedule is doing its thing |
| `train/tokens_per_sec` | throughput | regression detection |
| `train/step_time_ms` | wall time per step | hardware health |
| `train/grad_clipped_frac` | how often clip was active | tells you if `1.0` is too tight |

Every N (e.g., 500) steps:

| Key | What |
|-----|------|
| `eval/loss` | held-out cross-entropy |
| `eval/ppl` | `exp(eval/loss)` |
| `eval/by_domain/{web,code,math}/ppl` | per-domain perplexity |
| `eval/mmlu_acc` (every 5k) | downstream benchmark |
| `eval/gsm8k_acc` (every 5k) | reasoning benchmark |

Per-parameter health (every 1k steps):

| Key | What |
|-----|------|
| `params/<layer>.weight_norm` | per-layer L2 norm |
| `grads/<layer>.grad_norm` | per-layer gradient L2 |
| `acts/<layer>.std` | activation std (sample 1 batch) |

Checkpointing event:

| Key | What |
|-----|------|
| `ckpt/saved_at_step` | timestamp marker |
| `ckpt/path` | for resume |

---

## 3. The shape of a healthy loss curve

A healthy `train/loss` for an LLM:

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
- **Initial steep drop** in the first ~1k steps — model goes from random to "knows English."
- **Power-law tail** — on log-log axes, loss declines roughly linearly. (W&B has a "log scale" toggle.) If your curve isn't a power law, something is off.
- **Final asymptote** — every dataset has an irreducible loss; you'll get within a few % of it but never beat it.

For a 1.1B model on FineWeb-Edu, expect:
- `loss < 4.0` at step 1k (post-warmup)
- `loss ≈ 2.5` by step 30k
- `loss ≈ 2.2` by step 100k
- `loss ≈ 2.0` by step 200k

Different datasets have different floors — code/math/multilingual all push this up.

---

## 4. What various pathologies look like on the loss chart

### Loss diverges to inf/NaN early

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
**Causes:** LR too high, fp16 overflow, weight init too big.
**Fix:** lower LR by 10×, switch to bf16, longer warmup, lower init std (try 0.01 instead of 0.02).

### Loss plateau — flat from step 0

```
loss
|________________
|
+-------- steps
```
**Causes:** model not learning. Tokenizer mismatch (data is integers in one vocab, model expects another), LR is 0, optimizer not stepping (forgot `opt.step()`), `requires_grad=False` somewhere.
**Fix:** print first batch tokens decoded; verify `opt.step()` actually changes weights; check `param.grad.norm()` after `loss.backward()` is nonzero.

### Saw-tooth

```
loss
|   /\    /\    /\
|  /  \  /  \  /  \
| /    \/    \/    \____
+--------- steps
```
**Causes:** epoch boundary effect (loss jumps when same data revisited), bad shard ordering, LR too high near saturation.
**Fix:** shuffle shards across epochs, drop LR by 2×, ensure `pin_memory + non_blocking` to overlap data load.

### Sudden spike (1-3 steps), recovers

```
loss
|     .
|    /|
|   / |
|__/  |__________
+------- steps
```
**Causes:** rare junk batch (one ad-spam document), gradient anomaly. Single-batch spikes are normal at scale (~once per 10k steps).
**Fix:** harmless if it recovers. If it doesn't recover and you have to revert: implement **skip-on-spike** logic — if `loss > median × 5`, skip the optimizer step. Save a checkpoint just before the spike for inspection.

### Validation up while train down

```
       train/loss
          ___________
   loss /
   |  /            eval/loss
   | / .........-----'
   |/  .....----'
   +------ steps
```
**Causes:** overfitting (more likely on small datasets / multi-epoch).
**Fix:** more data, fewer epochs, slight dropout (0.05), more regularization. For LLMs trained 1-pass on 100B+ tokens, this almost never happens — train and val track together.

### Eval flat while train falls (memorization)

Train loss goes to ~0; eval loss stays high. Means the model is memorizing training data verbatim. Run **decontamination** and reduce epochs.

---

## 5. Gradient norm — the second most important chart

`grad_norm` should:
- **Spike high in the first 100 steps** (model is far from optimal).
- **Settle** to a roughly steady value, e.g. 0.5 - 2.0.
- **Slowly decrease** as the model converges.

Patterns:

- **Constantly hitting clip threshold (1.0)** → LR is too high, or training is unstable. Lower LR.
- **Suddenly explodes** → bad batch coming. Watch for skipped optimizer steps.
- **Goes to ~0** → model has stopped learning. Either you're done (rare) or something froze.

Always log **both `grad_norm`** (pre-clip) **and the clip ratio** (`grad_norm / max_norm`). If 50% of steps clip, you're throwing away signal.

---

## 6. Learning-rate chart — the easiest one

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

What to look for:
- The shape matches what you configured. (You'd be surprised how often it doesn't.)
- LR doesn't spike or drop unexpectedly mid-run.
- Anneal phase actually decays to ~0 (some implementations forget the final phase).

---

## 7. Throughput / step time

Two charts: `train/tokens_per_sec` and `train/step_time_ms`. They tell different stories.

- **Stable tokens/sec is great.** Drift downward = a regression somewhere. Spikes downward = network instability (multi-node), data loader stalled, GPU thermal throttling.
- **Step time variance > 30%** = your data loader can't keep up. Increase `num_workers`, `prefetch_factor`, simplify dataset code, or move expensive transforms offline.

In a multi-GPU run, log `tokens_per_sec / num_GPUs` to see per-GPU throughput. It should be roughly constant as you scale.

A typical 1B model on 8× H100 with FSDP2 and bf16 + `torch.compile`: **~50 000-90 000 tokens/sec**. Less than half that = something is wrong.

---

## 8. Per-layer norms — the silent canaries

```python
if step % 1000 == 0:
    for name, p in model.named_parameters():
        if p.grad is not None:
            wandb.log({
                f'params/{name}/norm': p.detach().norm().item(),
                f'grads/{name}/norm': p.grad.detach().norm().item(),
            }, step=step)
```

These produce hundreds of lines but pay dividends:

- **All weight norms grow** (gradient signal is reaching them) — good.
- **One layer's weight norm explodes** while others don't — bug in init or that layer is unstable. Inspect.
- **Gradient norm of a layer is ~0** — it's not training. Often a frozen-by-mistake layer or unused parameter.
- **Gradient norm of a layer is huge** while others are small — that layer is taking all the work. Adjust init or LR.

For deep models, the **last layer** and the **embedding** are the usual outliers; they need special LR or weight decay groups.

---

## 9. Activation stats — the deep diagnostic

Forward-hook a few layers and log their output `std`:

```python
acts = {}
def hook(name):
    def fn(_, __, out):
        acts[name] = out.detach().std().item()
    return fn

for i, layer in enumerate(model.layers[::4]):    # every 4th layer
    layer.register_forward_hook(hook(f'block_{i}'))

# after forward
wandb.log({f'acts/{k}': v for k, v in acts.items()}, step=step)
```

- **Stds should stay roughly constant across layers** (pre-norm + good init does this).
- **Growing std with depth** → activations are exploding. Add QK-norm or sandwich-norm.
- **Decreasing std with depth** → activations are vanishing. Init issue or norms misconfigured.
- **Suddenly going to 0** → dead layer, look at residual paths.

---

## 10. Eval charts — the truth

`train/loss` is what you watch hour-to-hour. `eval/*` is what tells you if your model is *getting better*.

- **eval/ppl** should follow train/loss closely.
- **mmlu_acc, hellaswag_acc** should improve smoothly. Small models start near random (~25% on MMLU 5-way) and reach ~30-50% depending on size and data.
- **gsm8k_acc** is noisy at small scale (often <5% until the model crosses ~3B). Don't panic.
- **humaneval_pass@1** likewise noisy until you have lots of code in pretraining.

W&B "panel groups" let you put all eval metrics together. Set up a "Pretraining benchmarks" group and pin it.

---

## 11. The "compute spent" chart

Log **cumulative tokens trained** as the x-axis instead of step number. It's far more comparable across runs with different batch sizes.

```python
tokens_so_far = step * effective_tokens_per_step
wandb.log({'progress/tokens': tokens_so_far,
           'progress/loss_vs_tokens': loss}, step=tokens_so_far)
```

Then in W&B set "x-axis = progress/tokens" on your loss chart. Suddenly all your runs are comparable to each other — and to public scaling-law fits.

---

## 12. Run comparisons (the W&B superpower)

The point of W&B isn't a single chart; it's overlaying **dozens of runs** on the same axes. Common patterns:

- **Sweeping LR**: 5 runs with `lr ∈ {1e-4, 2e-4, 3e-4, 5e-4, 1e-3}`. Find the best.
- **Sweeping data mix**: same model, different data ratios. Pick the winner.
- **Optimizer A vs B**: AdamW vs Muon, see who wins per-token.
- **With vs without an architectural tweak**: QK-norm on/off; tied embeddings on/off.

For credible comparisons:
- **Hold all but one variable constant.**
- **Use the same seed** when possible (W&B logs seed by default).
- **Plot vs tokens, not steps**, when batch size differs.
- **Fit a scaling-law line** to small-scale runs to predict large-scale outcomes.

W&B "Reports" let you pin charts and write comments — use them for postmortems. Future-you will thank current-you.

---

## 13. Logging text samples

Every 1k-5k steps, generate from the current model and log:

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

A 4-prompt fixed set is enough. Watching the model go from gibberish → broken English → coherent prose → genuinely interesting is the most motivating thing in pretraining. It's also the fastest way to spot a bad SFT round.

---

## 14. Alerts and notifications

Set W&B alerts for catastrophes:

```python
if not math.isfinite(loss):
    wandb.alert(title='NaN loss', text=f'step {step}', level=wandb.AlertLevel.ERROR)
```

You will be in bed when this fires. Better to know.

For long runs, also set Slack notifications via `wandb_alerts.py` or a small sidecar that polls the run.

---

## 15. Beyond W&B

- **TensorBoard** — open-source, local. Same idea. Great if your data can't leave your network.
- **MLflow** — better for experiment tracking with traditional ML; weaker on streaming charts.
- **Aim** — open-source W&B alternative, fast UI.
- **TrackioAI**, **Comet ML**, **Neptune** — paid alternatives.

In 2026 the LLM community defaults to **W&B for research and TensorBoard for cluster sanity checks**. Most large training jobs (Llama, Qwen, OLMo) publish W&B-style training curves in their tech reports.

---

## 16. Common dashboard layout

A panel layout that pays for itself within an hour of a long run:

1. **Top row (4 panels)**: `train/loss`, `eval/loss`, `eval/ppl`, `train/grad_norm`.
2. **Second row**: `train/lr`, `train/tokens_per_sec`, `train/step_time_ms`, `train/grad_clipped_frac`.
3. **Third row**: `acts/block_0/std`, `acts/block_<L/2>/std`, `acts/block_<L-1>/std`, *something domain-specific*.
4. **Fourth row**: downstream evals (`mmlu_acc`, `hellaswag_acc`, `gsm8k_acc`, `humaneval_pass@1`).
5. **Below**: text-sample table.
6. **Side panel**: run config diff vs another run.

Pin this layout. Keep it for every run. Compare across runs by overlaying.

---

## 17. The "first 30 minutes of a new run" checklist

After hitting `train.py`:

- [ ] W&B run started, tagged with the right project.
- [ ] First step logged within 60 seconds.
- [ ] `train/loss` is finite.
- [ ] `train/lr` matches schedule (just past 0).
- [ ] `train/tokens_per_sec` is in expected range.
- [ ] GPUs all near 100% utilization (`nvidia-smi`).
- [ ] No memory pressure (>10% headroom).
- [ ] After 500 steps, loss has dropped meaningfully.
- [ ] Gradient norm has stabilized.

If any of those is off, **stop and fix now**. A 4-day run started wrong is a 4-day run lost.

---

## 18. The 2026 cheat sheet

- **Log loss every step**, eval every 500-2000, benchmarks every 5-10k.
- **Plot vs tokens, not steps**, for cross-run comparability.
- **Watch grad norm and grad-clipped fraction** — they predict instability.
- **Per-layer norms** every 1k steps; they reveal silent bugs.
- **Sample text** to a W&B table — fastest qualitative signal.
- **Pin a dashboard layout** and reuse it.
- **Enable alerts** for NaN and stalled throughput.
- **Snapshot a checkpoint just before a known-bad event** so you can debug offline.

---

## Going deeper

- The W&B docs' "LLM training" guide.
- **`torchtitan`** logs — a public reference for what a serious distributed-training setup tracks.
- The OLMo / SmolLM-2 / Llama 3 training reports — they often publish their actual W&B charts, an excellent calibration for "what good looks like."
- **`wandb-sweeps`** for hyperparameter optimization on small-scale ablations.

That's the end of the curriculum. **Now go train something and read the charts.**
