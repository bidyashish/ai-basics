# 19 · Evaluation — How to Know If Your Model Is Actually Good

> **TL;DR** Evaluation is the under-invested half of LLM engineering. There are three tiers — **capability** (does the model know things?), **quality** (are the outputs good?), and **reliability** (does it stay good after changes?). Use **static benchmarks** (MMLU, GSM8K, HumanEval, SWE-bench) for capability, **LLM-as-judge** for open-ended quality, and your own **golden + regression sets** for reliability. **Eval-driven development** — writing your evals before your prompts — is the single highest-leverage practice for shipping LLM products in 2026. Every team without it ships slower and ships worse.

---

## 1. The three tiers of evaluation

Before any tooling, get clear on what you're trying to measure.

| Tier | Question it answers | Tool |
|------|---------------------|------|
| **Capability** | "Does the model know X?" | static benchmarks (MMLU, GSM8K, HumanEval) |
| **Quality** | "Is this answer good for the user?" | judge models, human raters, win-rate |
| **Reliability** | "Did my last change break anything?" | golden set + regression tests |

A finished AI product needs all three. Capability tells you which model to pick. Quality tells you whether your prompts and harness are good. Reliability tells you you can ship.

---

## 2. Capability benchmarks — the 2026 short list

These are the public benchmarks every model release reports. You don't need all of them; pick the 4-6 most relevant to your product.

| Benchmark | Tests | Notable |
|-----------|-------|---------|
| **MMLU / MMLU-Pro** | broad knowledge & reasoning across 57 / 14 fields | the canonical "general intelligence" test |
| **GPQA Diamond** | graduate-level science questions | harder, less contaminated |
| **BigBench Hard / BBEH** | mixed reasoning tasks | great for diversity |
| **GSM8K / MATH-500 / AIME 2025-26** | math | GSM8K is saturated; AIME is the new bar |
| **HumanEval / MBPP / LiveCodeBench v6** | code generation | LCB is the modern, less-contaminated standard |
| **SWE-bench Verified / SWE-bench Pro** | real-world bug fixes from GitHub | the headline agentic-coding bench |
| **Terminal-Bench 2.0** | shell / agent tool use | newer, tests autonomous CLI use |
| **MMMU / MMMU Pro** | multimodal reasoning | the standard for vision models |
| **MathVista / MATH-Vision** | math with images | long-context multimodal |
| **MRCR-v2** | long-context retrieval (8 needles in 128k) | post-needle-in-haystack standard |
| **OSWorld / GAIA** | autonomous agent on a desktop / web | agent benchmarks |
| **HELM / Open LLM Leaderboard** | aggregator | sanity-check rankings |

**Contamination warning.** Models train on the public web, which contains many benchmarks. Recent alternates (GPQA Diamond, MMLU-Pro, AIME 2025-26, LiveCodeBench v6, SWE-bench Verified) are designed to be harder to leak. Always cross-check on at least one *recent* benchmark; a model that aces only the old ones may have memorized them.

---

## 3. `lm-evaluation-harness` — the standard runner

EleutherAI's `lm-evaluation-harness` (often shortened to `lm-eval`) is the open-source standard for capability evals.

```bash
pip install lm-eval[vllm]

# evaluate any HF model on a battery of tasks
lm_eval --model hf \
        --model_args pretrained=Qwen/Qwen3.6-27B,dtype=bfloat16 \
        --tasks mmlu,hellaswag,arc_challenge,gsm8k,humaneval \
        --batch_size auto

# faster: serve via vLLM, evaluate via OpenAI-compatible API
vllm serve Qwen/Qwen3.6-27B --quantization fp8 &
lm_eval --model local-completions \
        --model_args base_url=http://localhost:8000/v1,model=Qwen/Qwen3.6-27B \
        --tasks mmlu_pro,gpqa,gsm8k,math \
        --batch_size 256
```

The harness handles prompt formatting, few-shot examples, and accuracy computation. The output is reproducible numbers you can compare against published model cards.

When you train your own model, run lm-eval at every checkpoint that crosses an interesting milestone. Plot the eval-loss curve alongside benchmark accuracy to catch divergence early.

---

## 4. LLM-as-judge — the workhorse for open-ended quality

For chat, summarization, writing, code review — anything not multiple-choice — you can't score with a regex. The 2026 standard is **LLM-as-judge**: another LLM (usually a frontier model) reads your model's output and rates it.

### Single-rating

```python
JUDGE_PROMPT = """You are an evaluator. Rate the assistant's response on a 1-5 scale
for {criterion}. 1 = terrible, 5 = excellent. Return only the number.

User question:
{question}

Assistant answer:
{answer}
"""
```

Run this on a few hundred (question, answer) pairs and average. Cheap, biased, useful.

### Pairwise win-rate (more reliable)

Show the judge two answers (yours vs baseline) and ask which is better.

```python
PAIRWISE = """Question: {q}

Answer A: {a}
Answer B: {b}

Which is better? Respond with exactly "A" or "B"."""
```

Always **randomize order** (counter the position bias — judges over-pick "A"). Run with at least two different judge models and average. Pairwise win-rate is the dominant metric in production: "we win 62% against last week's prompt" is exactly what stakeholders want to see.

### MT-Bench / AlpacaEval / Arena-Hard

Public LLM-as-judge benchmarks. **Arena-Hard-Auto** (correlates well with LMSYS Arena) and **WildBench** are the 2026 favorites. Use them as a sanity check against external models.

### Judge biases — what to watch for

- **Length bias**: judges prefer longer answers. Normalize, or use a judge that's been trained against this (the GPT-4 / Claude judges of 2024-2026 are mostly fixed but not perfectly).
- **Position bias**: order matters. Always randomize.
- **Self-preference**: a model judging another from the same family rates it higher. Use a *different family* judge when possible.
- **Style bias**: judges prefer formal, well-structured prose; if your product wants concise replies, build that into the rubric.

A useful sanity check: human-label 100 examples and check the judge's agreement with humans. Aim for ≥ 80% agreement; below that, your rubric needs work.

---

## 5. The golden set and regression tests

This is the eval that protects your product day-to-day.

### Golden set

A curated list of **50-500 representative examples** with known good answers (or judge rubrics). Covers:

- Common questions your users actually ask.
- Edge cases that have broken in the past.
- Adversarial prompts.
- Each major use case (different personas, different intents).

Stored in version control. Reviewed quarterly. **Treated as code.**

### Regression test

Every prompt change, model upgrade, or harness refactor runs the golden set. Output: pass/fail (vs. fixed rubric) or a win-rate vs. the previous version. **CI gate**: don't ship if win-rate < 50% or any high-priority example regresses.

```python
# pseudo-code: run golden set vs baseline
def run_eval(version_a, version_b, golden_set, judge):
    wins = 0
    for example in golden_set:
        a = version_a(example.prompt)
        b = version_b(example.prompt)
        verdict = judge.compare(example.prompt, a, b)
        if verdict == "B": wins += 1
    return wins / len(golden_set)
```

A good golden set is **brutal**. If your eval doesn't catch the bug you introduced, the eval is too easy. Add a failing case every time you find one in production — the set grows organically.

---

## 6. Online evals — the only ones that count for users

Offline evals are proxies. The only true measurement is whether real users get better outcomes. Common online metrics:

- **Thumbs up / thumbs down rate** on responses.
- **Edit distance** between assistant's suggestion and what the user actually shipped (for code completion).
- **Retention** / **session length** / **return rate**.
- **Task completion** for agents (did the user achieve their goal?).
- **Time-to-resolve** for support bots.
- **CSAT** survey on a sample.

Set up A/B testing infrastructure early. Run every meaningful prompt or model change as an A/B for a week before full rollout. **A/B beats intuition every time.** Tools: GrowthBook, Statsig, LaunchDarkly with metric experiments.

---

## 7. Agent-specific evaluation

Agents are harder to evaluate than single-turn responses. Three approaches:

### End-to-end task benchmarks

- **SWE-bench Verified / Pro** — real GitHub bug fixes; success = patch passes the project's tests.
- **OSWorld** — real desktop apps; success = task completed (file created, email sent, etc.).
- **GAIA** — research questions requiring multi-step browsing.
- **AppWorld** — multi-app tool use.
- **BrowseComp** — web browsing benchmark.
- **Terminal-Bench 2.0** — autonomous CLI tasks.

Run these as nightly CI on your agent. Score, regress, fix.

### Trajectory evaluation

Don't just judge the final answer; judge the *trajectory* (sequence of tool calls and decisions). Useful for catching:

- **Wasted tool calls** (the agent searched the web 6 times when it should have used the docs).
- **Wrong tool choice** (used `run_shell` when `read_file` would have done).
- **Loops** — same tool call repeated.
- **Premature giving up**.

LangSmith, Braintrust, and Phoenix all support trajectory inspection. A small custom scorer can catch many issues:

```python
def trajectory_score(traj):
    score = 1.0
    if len(traj) > 30: score -= 0.3                      # too many steps
    if has_loop(traj): score -= 0.5                       # looping
    if any(t.failed for t in traj): score -= 0.2          # tool errors
    if traj[-1].is_giveup_message: score -= 0.4
    return score
```

### Cost-and-latency evaluation

Agents that succeed but take 3 hours and $50 are not shipping. Always log:

- Total tokens (input + output, cached + new).
- Total cost.
- Wall-clock latency.
- Number of tool calls, retries, errors.

Set budgets in CI: any task that completes in 10× the median cost is flagged.

---

## 8. Eval-driven development

A 2026 best practice: **write the eval before the prompt or model change.** It forces you to define "better" before "different."

The loop:

1. Add 5-20 new examples to the golden set covering the *new* behavior.
2. Run eval — many fail (this is correct; the feature isn't built yet).
3. Iterate prompts / fine-tunes / harness until they pass.
4. Run the *full* golden set to check no regressions.
5. Ship.

This is the same discipline as TDD in software, applied to non-determinism. A team with eval-driven development out-ships a team without it 2-3×, easily.

---

## 9. Common eval pitfalls

**Goodhart's law in practice.** As soon as a benchmark becomes the target, it stops being a useful measure. Your model can ace MMLU and still be useless if you over-optimized to it. Mix benchmarks and rotate.

**Train-test contamination.** Benchmarks leak into pretraining data. Always check `train_eval_data_overlap` (Meta's tool, or n-gram overlap). Use *recent* benchmarks (AIME 2025-26, LiveCodeBench v6, SWE-bench Verified) that postdate your model's training cutoff.

**Eval-set overfitting.** If you tune prompts/models against the same set you measure on, the number is meaningless. Always hold out a *separate* test set you don't look at until shipping.

**Brittle string matching.** "The answer is 42." vs "42." Don't fail one. Use lenient matching for numeric/extractive tasks.

**Single-judge bias.** One judge model has its own biases. Use 2+ judges from different families and report both numbers.

**Ignoring tail behavior.** Average is misleading. Track P95 quality and worst-case examples. A model with mean=4.2 / P5=1.0 is *worse* than one with mean=4.0 / P5=3.5 for users.

**Too few examples.** Public papers run with 200-500 prompts and call it done. For production, use 1000+ examples per slice; otherwise variance swamps the signal.

**Expensive judging.** Frontier judges cost money. For high-volume eval, distill the judge: train a smaller classifier on judge labels, use the small one for bulk scoring. Microsoft, Cohere, and Anthropic publish papers on this.

---

## 10. Tooling — the 2026 stack

| Tool | Use |
|------|-----|
| **`lm-evaluation-harness`** | static capability benchmarks |
| **`promptfoo`** | prompt comparison, CI eval, web UI |
| **LangSmith / Langfuse / Braintrust / Helicone / Phoenix** | online eval, judge runs, dashboards, traces |
| **OpenAI Evals** | OpenAI-style eval framework, easy CI |
| **`inspect-ai`** (UK AISI) | safety + capability eval framework |
| **HELM** (Stanford) | meta-benchmark, broad coverage |
| **Arena-Hard-Auto** | LLM-as-judge harness, correlates with LMSYS |
| **`livebench.ai`** | continually-refreshed benchmark to dodge contamination |
| **`olmes`** (AllenAI) | reproducible OLMo-style eval |
| **WildBench / WildEval** | real user prompts, harder than synthetic |
| **Pydantic Logfire / OpenTelemetry GenAI** | structured agent traces |

For a typical product setup: **`promptfoo` for offline / CI**, **LangSmith or Braintrust for online traces**, **`lm-eval` for capability sanity-checks** when picking models. Keep your golden set in a Git-tracked YAML or JSONL; `promptfoo` reads it directly.

---

## 11. A worked eval pipeline

Your daily eval routine for an LLM product:

```
1. CI on every PR
   ├── Static: golden set (50-200 examples) → win-rate vs main branch
   ├── Trajectory checks (for agents)
   └── Latency / cost budget guards

2. Nightly
   ├── Larger golden set (1000+ examples)
   ├── Public benchmark sample (MMLU subset, GSM8K subset)
   ├── Agent benchmarks (SWE-bench Verified subset)
   └── Cost / latency P95 across slices

3. Weekly
   ├── Full golden set + judge (multiple judges)
   ├── User-feedback dashboard review
   ├── Worst-case example deep-dive
   └── Regression analysis (what got worse vs last week)

4. Monthly
   ├── External benchmark refresh (rotate to dodge contamination)
   ├── Re-label 5% of golden set for staleness
   └── Eval-of-eval: human-rate 100 examples, compare to judges
```

---

## 12. The 2026 cheat sheet

- **Three tiers**: capability (benchmarks), quality (judges), reliability (regression).
- **Eval-driven development** — write evals first.
- **Golden set in Git**, treated as code.
- **Pairwise win-rate** beats single-rating for chat.
- **Multiple judges** from different families.
- **Recent benchmarks** to dodge contamination.
- **Track P95**, not just mean.
- **Trajectory + cost evals for agents**, not just final answer.
- **Online A/B beats offline eval** for what users actually want.
- **`lm-eval` + `promptfoo` + Langfuse/Braintrust** is a working stack for most teams.

---

## Going deeper

- **`lm-evaluation-harness`** docs and source — read at least once.
- **Arena-Hard-Auto** paper — modern LLM-as-judge benchmark methodology.
- **Anthropic & OpenAI eval cookbooks** — practical recipes from people who do this at scale.
- **"Beyond accuracy" papers** (Liang et al., HELM) — designing evals that matter.
- **"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"** (Zheng et al. 2023) — the canonical paper on judge biases.
- **`inspect-ai`** UK AISI — open-source safety & capability eval framework.
- **SWE-bench / OSWorld / GAIA** papers — agent eval design.

Next: **[20-fine-tuning-recipes.md](./20-fine-tuning-recipes.md)** — making a base model do exactly what you want.
